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
        ce_loss = outputs.loss

    
        attention_kl_loss = torch.tensor(0.0, device=outputs.loss.device)
        
        for attention in all_attentions:
        
            topk_indices, raw_attn_weights, indexer_attn_scores = attention
            
            raw_attn_weights_topk = torch.gather(raw_attn_weights, -1, topk_indices.expand(-1, raw_attn_weights.shape[1], -1, -1))

            raw_attn_weights_topk = F.softmax(raw_attn_weights_topk, dim=-1)
  
            # head维度求和
            raw_attn_weights_topk = raw_attn_weights_topk.sum(1, keepdim=True)
 
            # L1归一化
            raw_attn_weights_topk = raw_attn_weights_topk / torch.norm(raw_attn_weights_topk, dim=-1, p=1, keepdim=True)

            indexer_attn_scores_topk = torch.gather(indexer_attn_scores, -1, topk_indices)
            # [batch_size, 1, seq_len, seq_len]
            indexer_attn_scores_topk = F.softmax(indexer_attn_scores_topk, dim=-1)
            indexer_attn_scores_topk = torch.clamp(indexer_attn_scores_topk, min=1e-8)
            kl_loss = F.kl_div(indexer_attn_scores_topk.log(), raw_attn_weights_topk.detach())
         
            attention_kl_loss = attention_kl_loss + kl_loss
        
        attention_kl_loss = attention_kl_loss / len(all_attentions)
        
        
        loss = ce_loss + attention_kl_loss
        
        return (loss, outputs) if return_outputs else loss
    
        
        


if __name__ == '__main__':
    import os

    model = Qwen2ForCausalLM.from_pretrained("step1_model")


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"可训练参数数量: {trainable_params:,}")
    print(f"总参数数量: {total_params:,}")
    
    
    tokenizer = AutoTokenizer.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct")
    
    args = TrainingArguments(output_dir='./step2', 
                            max_steps=2000, 
                            do_train=True, 
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=4,
                            logging_steps=1,
                            report_to='tensorboard',
                            save_strategy='steps',
                            save_steps=250,
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=0.000005,
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
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model('./step2_model')
    trainer.save_state()
    
    