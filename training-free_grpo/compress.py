from openai import OpenAI
import dotenv
dotenv.load_dotenv()
import os
from typing import Optional, List
import json
from prompts import EXPERIENCE_COMPRESSION_PROMPT

client = OpenAI(
        api_key=os.getenv('API_KEY'),
        base_url=os.getenv('BASE_URL')
    )
    
def get_llm_output(prompt: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv('MODEL_NAME'),
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

def load_experiences(experiences_file: str = 'experiences.json') -> str:
      
        if os.path.exists(experiences_file):
            with open(experiences_file, 'r', encoding='utf-8') as ef:
                experiences = json.load(ef)
                return experiences
        return {}
    

if __name__ == "__main__":
    experiences = load_experiences()
    print("Loaded experiences:", experiences)
    update_experiences = get_llm_output(EXPERIENCE_COMPRESSION_PROMPT.format(experiences=experiences)) 
    update_experiences = json.loads(update_experiences.split('```json')[-1].split('```')[0].strip())
    print("Update experiences:", update_experiences)
    max_id = max([int(k) for k in experiences.keys()], default=0)  
    for exp in update_experiences: 
        if exp['option'] == 'merge':
            to_merge = exp.get('merged_from', [])
            for exp_id in to_merge:
                if str(exp_id) in experiences:
                    del experiences[str(exp_id)]
                    
            experiences[str(max_id+1)] = exp['experience']
            max_id += 1
    
    # 重新编号
    compresse_experiences = {str(idx+1): exp for idx, exp in enumerate(experiences.values())}
    print("Compressed experiences:", compresse_experiences)
    with open('compresse_experiences.json', 'w', encoding='utf-8') as ef:
        json.dump(compresse_experiences, ef, ensure_ascii=False, indent=4)