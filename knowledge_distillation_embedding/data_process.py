from datasets import load_dataset
import random
import json
import os
def process_data(data_path, output_path, split='train', negative_num=10):
    datas = []
    dataset = load_dataset(data_path)
 
    data = dataset[split]
    for i in data:
        query = i['query']
        positive = i['positive']
        negative = i['negative']
        for pos in positive:
            
            if len(negative) >= negative_num:
                neg = random.sample(negative, negative_num)
            else:
                neg = random.choices(negative, k=negative_num)
            
            datas.append({'query': query, 'positive': pos, 'negative': neg})
    
    with open(os.path.join(output_path, f'processed_{split}_texts_negative_num_{negative_num}.json'), 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    
    process_data('origin_data', 'processed_data', negative_num=1)
               
            
            
    
