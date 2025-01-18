from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from dataset import SFTDataset
from utils import compute_fkl, compute_rkl, compute_skewed_fkl, compute_skewed_rkl


class KGTrainer(Trainer):
    
    def __init__(
        self,
        model = None,
        teacher_model = None,
        if_use_entropy = False,
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
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        loss = outputs.loss
        logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # 如果教师模型和学生模型输出形状不匹配，对学生模型进行padding或对教师模型进行截断
        if logits.shape[-1] != teacher_logits.shape[-1]:
            # gap = teacher_logits.shape[-1] - logits.shape[-1]
            # if gap > 0:
            #     pad_logits = torch.zeros((logits.shape[0], logits.shape[1], gap)).to(logits.device)
            #     logits = torch.cat([logits, pad_logits], dim=-1)
            
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
        
        labels = inputs['labels']
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=2.0)
        
        if self.if_use_entropy:
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl
        
        return (loss_total, outputs) if return_outputs else loss_total
        

if __name__ == '__main__':
    
    # 学生模型
    model = AutoModelForCausalLM.from_pretrained("Qwen2.5-0.5B-Instruct")
    
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
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-Instruct")
    
    # 教师模型，在给定数据上通过lora微调
    teacher_model = AutoModelForCausalLM.from_pretrained("Qwen2.5-7B-Instruct")
    # 是否加载lora模型
    lora_path = 'qwen2.5_7b/lora/sft'
    teacher_model = PeftModel.from_pretrained(teacher_model, lora_path)
    teacher_model.cuda()
    teacher_model.eval()
    
    args = TrainingArguments(output_dir='./results', 
                            num_train_epochs=10, 
                            do_train=True, 
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=16,
                            logging_steps=10,
                            report_to='tensorboard',
                            save_strategy='epoch',
                            save_total_limit=10,
                            bf16=True,
                            learning_rate=0.0005,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True)
    data_collator = DefaultDataCollator()
    dataset = SFTDataset('data.json', tokenizer=tokenizer, max_seq_len=512)
    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model, 
                        if_use_entropy = True,
                        args=args, 
                        train_dataset=dataset, 
                        tokenizer=tokenizer, 
                        data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves')
    trainer.save_state()
    
      
    