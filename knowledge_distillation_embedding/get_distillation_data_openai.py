import openai
import torch
import torch.nn.functional as F
import json
import tqdm
import argparse
import os

def similarity(emb1: torch.Tensor, emb2: torch.Tensor):
    return F.cosine_similarity(emb1, emb2)

def get_embedding(client, text, model="Qwen3-Embedding-4B"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data

def parse_args():
    parser = argparse.ArgumentParser(description="Process embedding data for knowledge distillation")
    
    parser.add_argument("--base_url", type=str, default="http://0.0.0.0:8077/v1", 
                        help="OpenAI API base URL")
    parser.add_argument("--api_key", type=str, default="123", 
                        help="API key")
    parser.add_argument("--model", type=str, default="Qwen3-Embedding-4B", 
                        help="Embedding model name")
    
    parser.add_argument("--input_file", type=str, 
                        default="processed_data/processed_train_texts.json",
                        help="Input JSON file path")
    parser.add_argument("--output_file", type=str, 
                        default="train_data/train.json",
                        help="Output JSON file path")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    client = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)
    
    with open(args.input_file, "r", encoding='utf-8') as f:
        train_texts = json.load(f)
    
    train_datas = []
    
    for data in tqdm.tqdm(train_texts, total=len(train_texts)):
        q = data['query']
        pos = data['positive']
        neg = data['negative']
        if isinstance(neg, str):
            embeddings = get_embedding(client, [q, pos, neg], args.model)
        else:
            embeddings = get_embedding(client, [q, pos] + neg, args.model)
        
        embeddings = [emb.embedding for emb in embeddings]
        
     
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
    

    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(train_datas, f, ensure_ascii=False, indent=4)
    
    print(f"Processing completed. Output saved to {args.output_file}")

if __name__ == "__main__":
    main()