# pip install -U sentence-transformers
import os
import re
import argparse
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict

import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from retrieval_server import get_retriever, Config as RetrieverConfig
from rerank_server import SentenceTransformerCrossEncoder

app = FastAPI()

def convert_title_format(text):
    # Use regex to extract the title and the content
    match = re.match(r'\(Title:\s*([^)]+)\)\s*(.+)', text, re.DOTALL)
    if match:
        title, content = match.groups()
        return f'\"{title}\"\n{content}'
    else:
        return text

# ----------- Combined Request Schema -----------
class SearchRequest(BaseModel):
    queries: List[str]
    topk_retrieval: Optional[int] = 10
    topk_rerank: Optional[int] = 3
    return_scores: bool = False

# ----------- Reranker Config Schema -----------
@dataclass
class RerankerArguments:
    max_length: int = field(default=512)
    rerank_topk: int = field(default=3)
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    batch_size: int = field(default=32)
    reranker_type: str = field(default="sentence_transformer")

def get_reranker(config):
    if config.reranker_type == "sentence_transformer":
        return SentenceTransformerCrossEncoder.load(
            config.rerank_model_name_or_path,
            batch_size=config.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        raise ValueError(f"Unknown reranker type: {config.reranker_type}")

# ----------- Endpoint -----------
@app.post("/retrieve")
def search_endpoint(request: SearchRequest):
    # Step 1: Retrieve documents
    retrieved_docs = retriever.batch_search(
        query_list=request.queries,
        num=request.topk_retrieval,
        return_score=False
    )

    # Step 2: Rerank
    reranked = reranker.rerank(request.queries, retrieved_docs)

    # Step 3: Format response
    response = []
    for i, doc_scores in reranked.items():
        doc_scores = doc_scores[:request.topk_rerank]
        if request.return_scores:
            combined = []
            for doc, score in doc_scores:
                combined.append({"document": convert_title_format(doc), "score": score})
            response.append(combined)
        else:
            response.append([convert_title_format(doc) for doc, _ in doc_scores])

    return {"result": response}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    # retriever
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--retrieval_topk", type=int, default=10, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')
    # reranker
    parser.add_argument("--reranking_topk", type=int, default=3, help="Number of reranked passages for one query.")
    parser.add_argument("--reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L12-v2", help="Path of the reranker model.")
    parser.add_argument("--reranker_batch_size", type=int, default=32, help="Batch size for the reranker inference.")

    args = parser.parse_args()
    
    # ----------- Load Retriever and Reranker -----------
    retriever_config = RetrieverConfig(
        retrieval_method = args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.retrieval_topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )
    retriever = get_retriever(retriever_config)

    reranker_config = RerankerArguments(
        rerank_topk = args.reranking_topk,
        rerank_model_name_or_path = args.reranker_model,
        batch_size = args.reranker_batch_size,
    )
    reranker = get_reranker(reranker_config)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
