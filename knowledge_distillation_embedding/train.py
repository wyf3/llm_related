from transformers import AutoModel, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from dataset import KGDataset
from torch import Tensor

def similarity(emb1: torch.Tensor, emb2: torch.Tensor, dim =1):
    return F.cosine_similarity(emb1, emb2, dim=dim)

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class KGTrainingArguments(TrainingArguments):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temperature = 1



class KGTrainer(Trainer):
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        labels = inputs.pop("labels")
        batch_size = inputs["input_ids"].shape[0]
        seq_len = inputs["input_ids"].shape[-1]
        
        # inputs["input_ids"].shape: [batch_size, sample_num, seq_len],sample_num=num_query+num_negative+num_positive
        inputs = {key: inputs[key].reshape(-1, seq_len) for key in inputs}
      
        outputs = model(**inputs)
        embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = embeddings.reshape(batch_size, -1, embeddings.shape[-1])
  
        query_embeddings = embeddings[:, :1]
  
        pos_neg_embeddings = embeddings[:, 1:]
   
        student_scores = similarity(query_embeddings, pos_neg_embeddings, dim=2)
        student_scores = student_scores / args.temperature
        student_log_probs = torch.log_softmax(student_scores, dim=1)

        teacher_scores = labels / args.temperature
        teacher_probs = torch.softmax(teacher_scores, dim=1)
        loss = loss_fct(student_log_probs, teacher_probs)
        
        loss = loss * (args.temperature**2)
        return (loss, outputs) if return_outputs else loss
        

if __name__ == '__main__':
    
    # 学生模型
    model = AutoModel.from_pretrained("Qwen3-Embedding-0.6B")
    
    lora_config = LoraConfig(
    r=8,  
    lora_alpha=256,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, 
    task_type=TaskType.SEQ_CLS)
    # 使用lora方法训练
    model = get_peft_model(model, lora_config)
    model.cuda()
    print(model.print_trainable_parameters())
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen3-Embedding-0.6B", padding_side='left')
    
    
    args = KGTrainingArguments(output_dir='./results_lora_8b_negative_1', 
                            num_train_epochs=2, 
                            do_train=True, 
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=4,
                            logging_steps=10,
                            report_to='tensorboard',
                            save_strategy='epoch',
                            save_total_limit=10,
                            bf16=True,
                            learning_rate=5e-5,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True)
    data_collator = DefaultDataCollator()
    dataset = KGDataset('train_data/train_negative_num_1_8b.json', tokenizer=tokenizer, max_seq_len=512)
    trainer = KGTrainer(model=model,
                        args=args, 
                        train_dataset=dataset, 
                        tokenizer=tokenizer, 
                        data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves_lora_8b_negative_1')
    trainer.save_state()
    
      
    