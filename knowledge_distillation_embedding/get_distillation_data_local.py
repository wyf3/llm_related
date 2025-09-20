from vllm import LLM, SamplingParams
import torch
import torch.nn.functional as F
import json
import tqdm
import argparse
import os

def similarity(emb1: torch.Tensor, emb2: torch.Tensor):
    return F.cosine_similarity(emb1, emb2)

def parse_args():
    parser = argparse.ArgumentParser(description="Process embedding data for knowledge distillation using vLLM")
    

    parser.add_argument("--model_path", type=str, 
                        default="Qwen3-Embedding-8B",
                        help="Path to the embedding model")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4,
                        help="GPU memory utilization ratio")
    

    parser.add_argument("--input_file", type=str, 
                        default="processed_data/processed_train_texts_negative_num_1.json",
                        help="Input JSON file path")
    parser.add_argument("--output_file", type=str, 
                        default="train_data/train_negative_num_1_8b.json",
                        help="Output JSON file path")
    

    parser.add_argument("--task", type=str, default="embed",
                        help="Task type for vLLM")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    print(f"Loading model from {args.model_path}...")
    llm = LLM(
        model=args.model_path, 
        gpu_memory_utilization=args.gpu_memory_utilization, 
        task=args.task
    )
    

    print(f"Reading data from {args.input_file}...")
    with open(args.input_file, "r", encoding='utf-8') as f:
        train_texts = json.load(f)
    
    train_datas = []
    
    # 处理数据
    print("Processing embeddings...")
    for data in tqdm.tqdm(train_texts, total=len(train_texts)):
        q = data['query']
        pos = data['positive']
        neg = data['negative']
        
        # 获取嵌入
        if isinstance(neg, str):
            outputs = llm.embed([q, pos, neg], use_tqdm=False)
        else:
            outputs = llm.embed([q, pos] + neg, use_tqdm=False)
        embeddings = [output.outputs.embedding for output in outputs]
        
        # 转换为tensor并计算相似度
        query_embedding = torch.tensor(embeddings[0], dtype=torch.float32).unsqueeze(0)
        pos_embedding = torch.tensor(embeddings[1], dtype=torch.float32).unsqueeze(0)
        neg_embedding = torch.tensor(embeddings[2:], dtype=torch.float32)
        
        pos_sim = similarity(query_embedding, pos_embedding)
        neg_sim = similarity(query_embedding, neg_embedding)
        sim = torch.cat([pos_sim, neg_sim], dim=0)
        label = sim.tolist()
        
        train_datas.append({
            "query": q, 
            "positive": pos, 
            "negative": neg, 
            "label": label
        })
    
    # 保存结果
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(train_datas, f, ensure_ascii=False, indent=4)
    
    print(f"Processing completed. Processed {len(train_datas)} samples. Output saved to {args.output_file}")

if __name__ == "__main__":
    main()
            

            
            
            
