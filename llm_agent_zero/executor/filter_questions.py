import argparse
import json
import pandas as pd
from prompts import executor_prompt
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default='data/generated_questions.json')
parser.add_argument("--output_file", type=str, default='data/filter_questions.parquet')
args = parser.parse_args()

with open(args.input_file, 'r') as f:
    datas = json.load(f)


data_source = []
prompt = []
ability = []
reward_model = []
extra_info = []
agent_name = []
raw_prompt = []

if datas:
    
    for idx, data in enumerate(datas):
        data_source.append('math')
        
        question = data['question']
        p = executor_prompt + [{"role": "user", "content": question}]
        prompt.append(p)
        raw_prompt.append(p)
        ability.append('math')
        answer = data['answer']
        reward_model.append({
            "style": "rule",
            "ground_truth": answer,
        })
        
        extra_info.append({
            'split': 'train',
            'index': idx,
        })
        
        agent_name.append('python_agent')
        
    
train_df = pd.DataFrame({
        'data_source': data_source,
        'prompt': prompt,
        'ability': ability,
        'reward_model': reward_model,
        'extra_info': extra_info,
        'agent_name': agent_name,
        'raw_prompt': raw_prompt
    })

train_df.to_parquet(args.output_file)