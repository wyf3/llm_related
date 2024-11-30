from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from train import VLMConfig, VLM


def find_assistant_tokens(tokenizer, target):
    result = []
    start_index =0
    end_index = 0
    while start_index <= len(target)-1:
        if target[start_index]!=tokenizer('assistant')['input_ids'][0]:
            start_index+=1
            end_index+=1
        else:
            end_index+=1
            if target[end_index]==tokenizer('<|im_end|>')['input_ids'][0]:
                result.append((start_index+1,end_index+1))
                start_index=end_index+1
    return result

class SFTDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)   
        
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = 'COCO_train2014_' + str(sample['image'])
            conversations = sample['conversations']
            messages = [{"role":"system", "content":'You are a helpful assistant.'}]
            for conversation in conversations:
                if conversation['from'] == 'human':
                    messages.append({"role":"user", "content":conversation['value']})
                else:
                    messages.append({"role":"assistant", "content":conversation['value']})
            text = tokenizer.apply_chat_template(messages, \
                tokenize=False, \
                ).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            # print(text)
            input_ids = tokenizer(text)['input_ids']
            indexs = find_assistant_tokens(tokenizer, input_ids)
            labels = len(input_ids) * [tokenizer.pad_token_id]
            for index in indexs:
                labels[index[0]:index[1]] = input_ids[index[0]:index[1]]
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
            
            image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB')
            
            pixel_values = self.processor(text=None, images=image)['pixel_values']
        except:
            
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":"图片内容是什么\n<image>"}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }   

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}


if __name__ == '__main__':
    config = VLMConfig()
    processor = AutoProcessor.from_pretrained("/home/user/wyf/siglip-base-patch16-224")
    tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct')
    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)
    model = AutoModelForCausalLM.from_pretrained('/home/user/wyf/train_multimodal_from_scratch/save/pretrain')
    
    for name, param in model.named_parameters():
        if 'linear' in name or 'vision_model':
            param.requires_grad = False
        if 'llm_model' in name:
            param.requires_grad = True
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters())}') 
    print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}') 
    images_path = './sft_images'
    data_path = './dataset/llava_instruct_230k.json'
    output_dir = 'save/sft' 
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SFTDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)  
    )
    
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model('save/sft')
    trainer.save_state()
