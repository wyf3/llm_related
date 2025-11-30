from model import Qwen2ForCausalLM
from transformers import Trainer, TrainingArguments, AutoTokenizer, DefaultDataCollator
import torch.nn as nn
import torch

from dataset import SFTDataset
import torch.nn.functional as F




class DSATrainer(Trainer):
    
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs, output_attentions=True)
        all_attentions = outputs.attentions

    
        attention_kl_loss = torch.tensor(0.0, device=outputs.loss.device)
        
        for attention in all_attentions:
            
            topk_indices, raw_attn_weights, indexer_attn_scores = attention
            
            raw_attn_weights = F.softmax(raw_attn_weights, dim=-1)
            
            # head维度求和
            raw_attn_weights = raw_attn_weights.sum(1, keepdim=True)
            
            # L1归一化
            raw_attn_weights = raw_attn_weights / torch.norm(raw_attn_weights, dim=-1, p=1, keepdim=True)
           
            
            # [batch_size, 1, seq_len, seq_len]
            indexer_attn_scores = F.softmax(indexer_attn_scores, dim=-1)
            indexer_attn_scores = torch.clamp(indexer_attn_scores, min=1e-8)
            kl_loss = F.kl_div(indexer_attn_scores.log(), raw_attn_weights.detach())
         
            attention_kl_loss += kl_loss
        
        loss = attention_kl_loss / len(all_attentions)
        return (loss, outputs) if return_outputs else loss
            
        
        


if __name__ == '__main__':
    import os

    model = Qwen2ForCausalLM.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct")
    
    for name, param in model.named_parameters():
     
        if 'indexer' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"可训练参数数量: {trainable_params:,}")
    print(f"总参数数量: {total_params:,}")
    
    tokenizer = AutoTokenizer.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct")
    
    args = TrainingArguments(output_dir='./step1', 
                            max_steps=500, 
                            do_train=True, 
                            per_device_train_batch_size=4,
                            gradient_accumulation_steps=4,
                            logging_steps=1,
                            report_to='tensorboard',
                            save_strategy='steps',
                            save_steps=250,
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=0.001,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True)
    data_collator = DefaultDataCollator()
    dataset = SFTDataset('warmup_data.jsonl', tokenizer=tokenizer, max_seq_len=2048)
    trainer = DSATrainer(model=model,
                        args=args, 
                        train_dataset=dataset, 
                        tokenizer=tokenizer, 
                        data_collator=data_collator)
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./step1_model')
    trainer.save_state()
    
    