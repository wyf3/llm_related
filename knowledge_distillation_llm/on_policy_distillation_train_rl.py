from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, DefaultDataCollator
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
from tqdm import tqdm
import torch

from utils import *
from dataset import *


@dataclass
class TrainingArguments:
    num_train_epochs = 1
    batch_size = 16
    mini_batch_size = 8
    micro_batch_size = 2
    learning_rate = 1e-7
    weight_decay = 0.01
    logger_steps = 1
    save_steps = 500
    output_dir = "./outputs_rl"
    max_grad_norm = 1.0
    warmup_steps = 0
    max_length = 1024
    max_prompt_length = 512
    temperature = 1.0
    max_steps = None
    cliprange = 0.2
    

class OnPolicyDistillationTrainer:
    
    def __init__(
        self,
        model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        args: TrainingArguments,
        tokenizer: PreTrainedTokenizer = None,
        data_collator = None, 
        train_dataset = None,
        eval_dataset = None,
        optimizers = (None, None)
    ):
        
        self.model = model
        self.teacher_model = teacher_model
        self.args = args
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        
        
        if self.data_collator is None:
            self.data_collator = DefaultDataCollator()
            
            
        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                collate_fn=self.data_collator,
                shuffle=True,
                drop_last=True,
                num_workers=8
            )
            
        self.optimizer, self.lr_scheduler = optimizers
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        
        self.max_steps = args.max_steps
        
        if self.max_steps is None:
            self.max_steps = len(self.train_dataloader) * args.num_train_epochs * args.batch_size // args.mini_batch_size
        
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=self.max_steps
            )
            
        self.writer = SummaryWriter(log_dir=args.output_dir)
        
        
            

    @torch.no_grad()
    def generate_sequences(self, input_ids, attention_mask):
        
        self.model.eval()
        sequences = self.model.generate(input_ids=input_ids, 
                                      attention_mask=attention_mask,
                                      max_length=self.args.max_length,
                                      do_sample=True,
                                      temperature=self.args.temperature,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id
                                      )
        
     
        
        self.model.train()
        return sequences

    
    def selective_log_softmax(self, logits, index):
    
        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
            
            logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected_logits - logsumexp_values 
        else:
            per_token_logps = []
            for row_logits, row_labels in zip(logits, index): 
                row_logps = F.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        return per_token_logps

    def train(self):
        
        global_steps = 0
        pbar = tqdm(total=self.max_steps, desc="Training")
        
            
        for epoch in range(self.args.num_train_epochs):
            # 取一个batch_size生成数据
            for _, inputs in enumerate(self.train_dataloader):
                prompt_ids = inputs["input_ids"].to(self.model.device)
                prompt_mask = inputs["attention_mask"].to(self.model.device)
                sequences = self.generate_sequences(prompt_ids, prompt_mask)
                
                
                with torch.no_grad():
                    attention_mask = (sequences != self.tokenizer.pad_token_id).long()
                    logits = self.model(sequences, attention_mask=attention_mask).logits
               
                
                # 取一个mini_batch_size生成经验(kl散度)
                for mini_idx in range(0, self.args.batch_size, self.args.mini_batch_size):
                    mini_input_ids = sequences[mini_idx:mini_idx+self.args.mini_batch_size]
                    mini_attention_mask = attention_mask[mini_idx:mini_idx+self.args.mini_batch_size]
                    with torch.no_grad():
                        mini_teacher_outputs = self.teacher_model(mini_input_ids, attention_mask=mini_attention_mask)
                        mini_student_logits = logits[mini_idx:mini_idx+self.args.mini_batch_size, prompt_ids.shape[-1]:]
                        mini_teacher_logits = mini_teacher_outputs.logits[:, prompt_ids.shape[-1]:]
                        
                        mini_completion_ids = mini_input_ids[:, prompt_ids.shape[-1]:]
                    
                        mini_teacher_logits_clamp = mini_teacher_logits[:, :, :mini_student_logits.shape[-1]]
                    
                        mini_student_probs = self.selective_log_softmax(mini_student_logits, mini_completion_ids)
                    
                        kl = compute_rkl(mini_student_logits, mini_teacher_logits_clamp, mini_completion_ids, self.tokenizer.pad_token_id, reduction="")
                        
                        mini_completion_mask = mini_attention_mask[:, prompt_ids.shape[-1]:]
                     
                        reward = -kl
                        reward_mean = (reward * mini_completion_mask).sum(dim=1, keepdim=True) / mini_completion_mask.sum(dim=1, keepdim=True)
                        # 减去均值，优势有正有负更符合直觉，且正优势对应kl散度小的部分，负优势对应kl散度大的部分
                        adv = reward - reward_mean
                        adv = adv * mini_completion_mask.float()
                        
                   
                    del mini_student_logits,mini_teacher_logits, mini_teacher_logits_clamp, mini_teacher_outputs
                    torch.cuda.empty_cache()
                  
                    self.optimizer.zero_grad()
                    
                    for micro_idx in range(0, self.args.mini_batch_size, self.args.micro_batch_size):
                        micro_input_ids = mini_input_ids[micro_idx:micro_idx+self.args.micro_batch_size]
                        micro_attention_mask = mini_attention_mask[micro_idx:micro_idx+self.args.micro_batch_size]
                        old_micro_student_probs = mini_student_probs[micro_idx:micro_idx+self.args.micro_batch_size]
                        micro_adv = adv[micro_idx:micro_idx+self.args.micro_batch_size]
                        
                        micro_completion_ids = mini_completion_ids[micro_idx:micro_idx+self.args.micro_batch_size]
                        
                        micro_outputs = self.model(micro_input_ids, attention_mask=micro_attention_mask)
                        micro_student_logits = micro_outputs.logits[:, prompt_ids.shape[-1]:, :]
                        
                        micro_student_probs = self.selective_log_softmax(micro_student_logits, micro_completion_ids)
                    
                        del micro_outputs, micro_student_logits
                        torch.cuda.empty_cache()
                        
                        
                        micro_student_probs = micro_student_probs.masked_fill(micro_completion_ids == self.tokenizer.pad_token_id, 0.0)
                        old_micro_student_probs = old_micro_student_probs.masked_fill(micro_completion_ids == self.tokenizer.pad_token_id, 0.0)
                        
                        logprobs_diff = micro_student_probs - old_micro_student_probs
              
                        ratio = torch.exp(logprobs_diff)
                        micro_completion_mask = micro_attention_mask[:, prompt_ids.shape[-1]:]
                        
                     
                        pg_losses = -micro_adv * ratio
                        pg_losses2 = -micro_adv * torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        
                    
                        # sequence loss
                        token_loss_per_seq = (pg_loss_max * micro_completion_mask).sum(dim=1) / (micro_completion_mask.sum(dim=1) + 1e-8)
                        loss = token_loss_per_seq.mean()
                        
                        # token loss
                        # loss = (pg_loss_max * micro_completion_mask).sum() / micro_completion_mask.sum()
                        
                    
                        loss.backward()
                        
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                    self.optimizer.step()
                    self.lr_scheduler.step() 
                    
                    
                    global_steps += 1
                    
                        
                    if global_steps % self.args.logger_steps == 0:
                       
                        pbar.set_postfix({
                        'epoch': f"{epoch+1}/{self.args.num_train_epochs}",
                        'global_step': global_steps,
                        'loss': f"{loss.item():.6f}",
                        'lr': f"{self.lr_scheduler.get_last_lr()[0]:.6f}",
                        'adv': f"{adv.mean().item():.6f}",
                        'kl': f"{kl.mean().item():.6f}"
                        })
                        pbar.update(1)
                        
                    if global_steps % self.args.save_steps == 0:
                        
                        self.model.save_pretrained(f"{self.args.output_dir}/model_{global_steps}")
                    
                    self.writer.add_scalar("loss", loss.item(), global_steps)
                    self.writer.add_scalar("lr", self.lr_scheduler.get_last_lr()[0], global_steps)
                    self.writer.add_scalar("adv", adv.mean().item(), global_steps)
                    self.writer.add_scalar("kl", kl.mean().item(), global_steps)
                    
           
          
        self.model.save_pretrained(f"{self.args.output_dir}/model_{global_steps}")
        self.writer.close()
        pbar.close()                    
                            

if __name__ == "__main__":
    
    model = AutoModelForCausalLM.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    args = TrainingArguments()
    teacher_model = AutoModelForCausalLM.from_pretrained("/home/user/Downloads/Qwen2.5-7B-Instruct", trust_remote_code=True)
    
    model.cuda()
    teacher_model.cuda()
    teacher_model.eval()
    
    
    train_dataset = OnPolicyDataset('data.json', tokenizer, args)
    
    trainer = OnPolicyDistillationTrainer(
        model, 
        teacher_model, 
        args, 
        tokenizer=tokenizer,
        train_dataset=train_dataset)
    
    trainer.train()

                      

   
        
        
            
        