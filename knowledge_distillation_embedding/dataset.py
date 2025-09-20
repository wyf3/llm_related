import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from torch.utils.data import IterableDataset, Dataset
import json


class KGDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):

        sample = self.data[index]
        if isinstance(sample['negative'], str):
            input_texts = [sample['query']] + [sample['positive']] + [sample['negative']]
        else:
            input_texts = [sample['query']] + [sample['positive']] + sample['negative']
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            input_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )   
        # batch_dict['input_ids'].shape:[sample_num, seq_len]，sample_num=num_query+num_negative+num_positive

        batch_dict['labels'] = torch.tensor(sample['label'])
        return batch_dict