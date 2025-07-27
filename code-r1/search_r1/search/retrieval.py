import json
import os
import warnings
from typing import List, Dict
import functools
from tqdm import tqdm
from multiprocessing import Pool
import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
import argparse
import datasets


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
            'json', 
            data_files=corpus_path,
            split="train",
            num_proc=4)
    return corpus
    

def read_jsonl(file_path):
    data = []
    
    with open(file_path, "r") as f:
        readin = f.readlines()
        for line in readin:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]

    return results


def load_model(
        model_path: str, 
        use_fp16: bool = False
    ):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer


def pooling(
        pooler_output,
        last_hidden_state,
        attention_mask = None,
        pooling_method = "mean"
    ):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path,
                                                use_fp16=use_fp16)

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb


class BaseRetriever:
    """Base object for all retrievers."""

    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

        # self.cache_save_path = os.path.join(config.save_dir, 'retrieval_cache.json')

    def _search(self, query: str, num: int, return_score:bool) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.
        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)
        """
        pass

    def _batch_search(self, query_list, num, return_score):
        pass

    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)
    
    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)


class BM25Retriever(BaseRetriever):
    r"""BM25 retriever based on pre-built pyserini index."""

    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
        
    def _check_contain_doc(self):
        r"""Check if the index contains document content
        """
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score = False) -> List[Dict[str, str]]:
        if num is None:
            num = self.topk
        
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [],[]
            else:
                return []
            
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())['contents'] for hit in hits]
            results = [{'title': content.split("\n")[0].strip("\""), 
                        'text': "\n".join(content.split("\n")[1:]),
                        'contents': content} for content in all_contents]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list, num: int = None, return_score = False):
        # TODO: modify batch method
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num,True)
            results.append(item_result)
            scores.append(item_score)

        if return_score:
            return results, scores
        else:
            return results

def get_available_gpu_memory():
    memory_info = []
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        free_memory = total_memory - allocated_memory
        memory_info.append((i, free_memory / 1e9))  # Convert to GB
    return memory_info


class DenseRetriever(BaseRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            # self.index = faiss.index_cpu_to_all_gpus(self.index)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
             model_name = self.retrieval_method, 
             model_path = config.retrieval_model_path,
             pooling_method = config.retrieval_pooling_method,
             max_length = config.retrieval_query_max_length,
             use_fp16 = config.retrieval_use_fp16
            )
        self.topk = config.retrieval_topk
        self.batch_size = self.config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        batch_size = self.batch_size

        results = []
        scores = []

        for start_idx in tqdm(range(0, len(query_list), batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + batch_size]
            
            # from time import time
            # a = time()
            batch_emb = self.encoder.encode(query_batch)
            # b = time()
            # print(f'################### encode time {b-a} #####################')
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()
            # print(f'################### search time {time()-b} #####################')
            # exit()
            
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            scores.extend(batch_scores)
            results.extend(batch_results)
        
        if return_score:
            return results, scores
        else:
            return results

def get_retriever(config):
    r"""Automatically select retriever class based on config's retrieval method

    Args:
        config (dict): configuration with 'retrieval_method' key

    Returns:
        Retriever: retriever instance
    """
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


def get_dataset(config):
    """Load dataset from config."""
     
    split_path = os.path.join(config.dataset_path, f'{config.data_split}.jsonl')
    return read_jsonl(split_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "Retrieval")

    # Basic parameters
    parser.add_argument('--retrieval_method', type=str)
    parser.add_argument('--retrieval_topk', type=int, default=10)
    parser.add_argument('--index_path', type=str, default=None)
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--dataset_path', default=None, type=str)

    parser.add_argument('--faiss_gpu', default=True, type=bool)
    parser.add_argument('--data_split', default="train", type=str)
    
    parser.add_argument('--retrieval_model_path', type=str, default=None)
    parser.add_argument('--retrieval_pooling_method', default='mean', type=str)
    parser.add_argument('--retrieval_query_max_length', default=256, type=str)
    parser.add_argument('--retrieval_use_fp16', action='store_true', default=False)
    parser.add_argument('--retrieval_batch_size', default=512, type=int)
    
    args = parser.parse_args()

    args.index_path = os.path.join(args.index_path, f'{args.retrieval_method}_Flat.index') if args.retrieval_method != 'bm25' else os.path.join(args.index_path, 'bm25')

    # load dataset
    all_split = get_dataset(args)
    
    input_query = [sample['question'] for sample in all_split[:512]]
    
    # initialize the retriever and conduct retrieval
    retriever = get_retriever(args)
    print('Start Retrieving ...')    
    results, scores = retriever.batch_search(input_query, return_score=True)

    # from IPython import embed
    # embed()
