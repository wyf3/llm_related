from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from dataset import SFTDataset, OnPolicyDataset
from utils import compute_rkl



class KGTrainer(Trainer):
    
    def __init__(
        self,
        model = None,
        teacher_model = None,
        args = None,
        data_collator = None, 
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None, 
        compute_metrics = None, 
        callbacks = None,
        optimizers = (None, None), 
        preprocess_logits_for_metrics = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model

    
    
    @torch.no_grad()
    def generate_sequences(self, input_ids, attention_mask):
        
        self.model.eval()
        sequences = self.model.generate(input_ids=input_ids, 
                                      attention_mask=attention_mask,
                                      max_length=1024,
                                      do_sample=True,
                                      temperature=1.0,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id
                                      )
        
        
        
        self.model.train()
        return sequences
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        prompt_ids = inputs["input_ids"].to(self.model.device)
        prompt_mask = inputs["attention_mask"].to(self.model.device)
        sequences = self.generate_sequences(prompt_ids, prompt_mask)
        attention_mask = (sequences != self.tokenizer.pad_token_id).long()
        logits = model(sequences, attention_mask=attention_mask).logits[:, prompt_ids.shape[-1]:]
        
        loss = None
        with torch.no_grad():
            teacher_outputs = self.teacher_model(sequences, attention_mask=attention_mask)
        
        teacher_logits = teacher_outputs.logits[:, prompt_ids.shape[-1]:]
        
        
        if logits.shape[-1] != teacher_logits.shape[-1]:
           
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
        
        completion_ids = sequences[:, prompt_ids.shape[-1]:]
        kl = compute_rkl(logits, teacher_logits, completion_ids, padding_id=self.tokenizer.pad_token_id, reduction="mean")
        
        loss = kl.mean()
        
        return loss
        

if __name__ == '__main__':
    
    model = AutoModelForCausalLM.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    
    lora_config = LoraConfig(
    r=8,  
    lora_alpha=256,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, 
    task_type=TaskType.CAUSAL_LM)
    # 使用lora方法训练
    model = get_peft_model(model, lora_config)
    model.cuda()
    print(model.print_trainable_parameters())
    

    teacher_model = AutoModelForCausalLM.from_pretrained("/home/user/Downloads/Qwen2.5-7B-Instruct", trust_remote_code=True)
    
    model.cuda()
    teacher_model.cuda()
    teacher_model.eval()
    
    
    train_dataset = OnPolicyDataset('data.json', tokenizer)
    
    
    
    args = TrainingArguments(output_dir='./outputs', 
                            num_train_epochs=1, 
                            do_train=True, 
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=4,
                            logging_steps=1,
                            report_to='tensorboard',
                            save_strategy='epoch',
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=0.00001,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True)
    data_collator = DefaultDataCollator()
    
    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model, 
                        args=args, 
                        train_dataset=train_dataset, 
                        tokenizer=tokenizer, 
                        data_collator=data_collator)
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves')
    trainer.save_state()
    
    
      
    