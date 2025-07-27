import os
import re
import requests
import argparse
import asyncio
import random
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

import chardet
import aiohttp
import bs4
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from googleapiclient.discovery import build


# --- CLI Args ---
parser = argparse.ArgumentParser(description="Launch online search server.")
parser.add_argument('--api_key', type=str, required=True, help="API key for Google search")
parser.add_argument('--cse_id', type=str, required=True, help="CSE ID for Google search")
parser.add_argument('--topk', type=int, default=3, help="Number of results to return per query")
parser.add_argument('--snippet_only', action='store_true', help="If set, only return snippets; otherwise, return full context.")
args = parser.parse_args()


# --- Config ---
class OnlineSearchConfig:
    def __init__(self, topk: int = 3, api_key: Optional[str] = None, cse_id: Optional[str] = None, snippet_only: bool = False):
        self.topk = topk
        self.api_key = api_key
        self.cse_id = cse_id
        self.snippet_only = snippet_only


# --- Utilities ---
def parse_snippet(snippet: str) -> List[str]:
    segments = snippet.split("...")
    return [s.strip() for s in segments if len(s.strip().split()) > 5]


def sanitize_search_query(query: str) -> str:
    # Remove or replace special characters that might cause issues.
    # This is a basic example; you might need to add more characters or patterns.
    sanitized_query = re.sub(r'[^\w\s]', ' ', query)  # Replace non-alphanumeric and non-whitespace with spaces.
    sanitized_query = re.sub(r'[\t\r\f\v\n]', ' ', sanitized_query) # replace tab, return, formfeed, vertical tab with spaces.
    sanitized_query = re.sub(r'\s+', ' ', sanitized_query).strip() #remove duplicate spaces, and trailing/leading spaces.

    return sanitized_query


def filter_links(search_results: List[Dict]) -> List[str]:
    links = []
    for result in search_results:
        for item in result.get("items", []):
            if "mime" in item:
                continue
            ext = os.path.splitext(item["link"])[1]
            if ext in ["", ".html", ".htm", ".shtml"]:
                links.append(item["link"])
    return links


async def fetch(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> str:
    user_agents = [
        "Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P)...",
        "Mozilla/5.0 AppleWebKit/537.36...",
        "Mozilla/5.0 (compatible; Googlebot/2.1; +https://www.google.com/bot.html)",
    ]
    headers = {"User-Agent": random.choice(user_agents)}

    async with semaphore:
        try:
            async with session.get(url, headers=headers) as response:
                raw = await response.read()
                detected = chardet.detect(raw)
                encoding = detected["encoding"] or "utf-8"
                return raw.decode(encoding, errors="ignore")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return ""


async def fetch_all(urls: List[str], limit: int = 8) -> List[str]:
    semaphore = asyncio.Semaphore(limit)
    timeout = aiohttp.ClientTimeout(total=5)
    connector = aiohttp.TCPConnector(limit_per_host=limit, force_close=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [fetch(session, url, semaphore) for url in urls]
        return await asyncio.gather(*tasks)


# --- Search Engine ---
class OnlineSearchEngine:
    def __init__(self, config: OnlineSearchConfig):
        self.config = config

    def collect_context(self, snippet: str, doc: str) -> str:
        snippets = parse_snippet(snippet)
        ctx_paras = []

        for s in snippets:
            pos = doc.replace("\n", " ").find(s)
            if pos == -1:
                continue
            sta = pos
            while sta > 0 and doc[sta] != "\n":
                sta -= 1
            end = pos + len(s)
            while end < len(doc) and doc[end] != "\n":
                end += 1
            para = doc[sta:end].strip()
            if para not in ctx_paras:
                ctx_paras.append(para)

        return "\n".join(ctx_paras)

    def fetch_web_content(self, search_results: List[Dict]) -> Dict[str, str]:
        links = filter_links(search_results)
        contents = asyncio.run(fetch_all(links))
        content_dict = {}
        for html, link in zip(contents, links):
            soup = bs4.BeautifulSoup(html, "html.parser")
            text = "\n".join([p.get_text() for p in soup.find_all("p")])
            content_dict[link] = text
        return content_dict

    def search(self, search_term: str, num_iter: int = 1) -> List[Dict]:
        service = build('customsearch', 'v1', developerKey=self.config.api_key)
        results = []
        sanitize_search_term = sanitize_search_query(search_term)
        if search_term.isspace():
            return results
        res = service.cse().list(q=sanitize_search_term, cx=self.config.cse_id).execute()
        results.append(res)

        for _ in range(num_iter - 1):
            if 'nextPage' not in res.get('queries', {}):
                break
            start_idx = res['queries']['nextPage'][0]['startIndex']
            res = service.cse().list(q=search_term, cx=self.config.cse_id, start=start_idx).execute()
            results.append(res)

        return results

    def batch_search(self, queries: List[str]) -> List[List[str]]:
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self._retrieve_context, queries))

    def _retrieve_context(self, query: str) -> List[str]:
        
        if self.config.snippet_only:
            search_results = self.search(query)
            contexts = []
            for result in search_results:
                for item in result.get("items", []):
                    title = item.get("title", "")
                    context = ' '.join(parse_snippet(item.get("snippet", "")))
                    if title != "" or context != "":
                        title = "No title." if not title else title
                        context = "No snippet available." if not context else context
                        contexts.append({
                            'document': {"contents": f'\"{title}\"\n{context}'},
                        })
        else:
            content_dict = self.fetch_web_content(search_results)
            contexts = []
            for result in search_results:
                for item in result.get("items", []):
                    link = item["link"]
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    if link in content_dict:
                        context = self.collect_context(snippet, content_dict[link])
                        if title != "" or context != "":
                            title = "No title." if not title else title
                            context = "No snippet available." if not context else context
                            contexts.append({
                                'document': {"contents": f'\"{title}\"\n{context}'},
                            })
        
        return contexts[:self.config.topk]


# --- FastAPI App ---
app = FastAPI(title="Online Search Proxy Server")

class SearchRequest(BaseModel):
    queries: List[str]

config = OnlineSearchConfig(api_key=args.api_key, cse_id=args.cse_id, topk=args.topk, snippet_only=args.snippet_only)
engine = OnlineSearchEngine(config)

@app.post("/retrieve")
def search_endpoint(request: SearchRequest):
    results = engine.batch_search(request.queries)
    return {"result": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
