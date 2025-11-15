import torch
from torch.utils.data import IterableDataset, Dataset
import json

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):
        line = self.data[index]
        prompt = line['prompt']
        answer = line['answer']
   
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        
        input_ids = prompt_ids + answer_ids
        labels = answer_ids
        
        return {'input_ids': input_ids, 'labels': labels}
        
        
class MyDataCollator:
    def __init__(self):

        pass
        
    def __call__(self, features):
        
        input_ids = [feature['input_ids'] for feature in features]
        labels = [feature['labels'] for feature in features]
    
     
        return {'input_ids': input_ids, 'labels': labels}