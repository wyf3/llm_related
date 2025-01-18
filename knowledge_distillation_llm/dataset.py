import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import pandas as pd

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import  PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding_id = tokenizer.pad_token_id
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, index):
        line = self.data[index]
        instruction_text = line['instruction']
        input_text = line['input']
        output_text = line['output']
        query = instruction_text + input_text
        answer = output_text + self.tokenizer.eos_token
        messages = []
        messages.append({'role': 'user', 'content': query})   
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False) 
        
        prompt_input_ids = self.tokenizer.encode(prompt)
        answer_input_ids = self.tokenizer.encode(answer)
        
        input_ids = prompt_input_ids + answer_input_ids
        labels = [-100] * len(prompt_input_ids) + answer_input_ids
        attention_mask = [1] * len(input_ids)
        text_len = len(input_ids)
        
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]
        else:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - text_len)
            labels = labels + [-100] * (self.max_seq_len - text_len)
            attention_mask = attention_mask + [0] * (self.max_seq_len - text_len)
        
        # input_ids = input_ids[:-1]
        # labels = labels[1:]
        return {'input_ids': torch.tensor(input_ids), 'attention_mask':torch.tensor(attention_mask), 'labels': torch.tensor(labels)}