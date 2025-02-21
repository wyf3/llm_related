from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Dict, Any
import json
import os
from torch.utils.tensorboard import SummaryWriter


class Config():
    def __init__(self,
                llm_model_path = '/home/user/Downloads/Qwen2.5-0.5B-Instruct',
                predict_tokens_num = 5,
                **kwargs):
        self.llm_model_path = llm_model_path
        self.predict_tokens_num = predict_tokens_num
        super().__init__(**kwargs)

class MTPModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(2 * hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
        

class MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.main_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path).base_model
        # self.main_model.eval()
        # mtp模块
        self.mtp_modules = nn.ModuleList([MTPModule(self.main_model.config.hidden_size) for _ in range(self.config.predict_tokens_num-1)])
        
        # 每个头共享参数
        self.output_head = nn.Linear(self.main_model.config.hidden_size, self.main_model.config.vocab_size)
        
         
    def forward_main(self, input_ids, attention_mask=None, **kwargs):
        
        # with torch.no_grad():
        main_hidden_output = self.main_model(input_ids, attention_mask, **kwargs).last_hidden_state
       
        
        main_head_output = self.output_head(main_hidden_output)
        
        return main_hidden_output, main_head_output
    
    def forward_mtp(self, input_ids, previous_hidden_output, head_index):
        input_embed = self.main_model.get_input_embeddings()(input_ids)
        mtp_input = torch.cat([previous_hidden_output, input_embed], dim=-1)
        mtp_hidden_output = self.mtp_modules[head_index](mtp_input)
        mtp_head_output = self.output_head(mtp_hidden_output)
        
        return mtp_hidden_output, mtp_head_output
    
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        
        outputs = {}
        main_hidden_output, main_head_output = self.forward_main(input_ids, attention_mask, **kwargs)
        previous_hidden_output = main_hidden_output
        outputs['head_main'] = main_head_output
        for head_index in range(0, self.config.predict_tokens_num-1):
            previous_hidden_output, mtp_head_output = self.forward_mtp(input_ids, previous_hidden_output, head_index)
            outputs[f'mtp_head_{head_index}'] = mtp_head_output
            
        return outputs
    
    def generate(self,input_ids,max_length, **kwargs):
        self.eval()
        seq = input_ids.clone()
        b, s = seq.size()
        
        with torch.no_grad():
            
            while seq.size(1) < max_length:
                outputs = self.forward(seq)
                print(seq.shape)
                speculative_tokens = []
                
                # main模型头生成的token
                logits = outputs['head_main']
                logits = logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)
                speculative_tokens.append(next_token)
                
                # 汇总mtp头生成的token
                for i in  range(self.config.predict_tokens_num-1):
                    logits = outputs[f'mtp_head_{i}']
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.argmax(probs, dim=-1)
                    
                    speculative_tokens.append(next_token)
                
                  
                
                speculative_tokens = torch.cat(speculative_tokens, dim=-1) # shape: (len)
                speculative_tokens = speculative_tokens.unsqueeze(0) # shape: (1, len)
                
                # 将新生成的tokens和原始序列拼接
                all_tokens = torch.cat([seq, speculative_tokens], dim=-1)
                
                # 将新序列输入main模型(验证模型)进行验证，保留符合条件的token
                _, all_logits = self.forward_main(all_tokens)
                
                # 取出需要验证的token对应的logits
                validation_logits = all_logits[:, -speculative_tokens.shape[1]:]
                
                # 获取各个token在main模型的输出概率
                accept_probs =  []

                for i in range(speculative_tokens.shape[1]):
                    logits = validation_logits[:, i] # (batch_size, vocab_size)
                    probs = torch.softmax(logits, dim=-1) # (batch_size, vocab_size)
                    token = speculative_tokens[:, i]
                   
                    token_prob = probs.gather(1, token.unsqueeze(0))
                    accept_probs.append(token_prob)
             
                # 拼接各个token的生成概率
                accept_probs = torch.cat(accept_probs, dim=-1)
                
                # 保留概率值大于阈值的token, 接受这部分token,否则舍弃（舍弃某个token时，后面的token都要舍弃）
                # 接受token的掩码
                accept_mask = (accept_probs > 1e-6)
                print(f'接受掩码：{accept_mask}')
                
                if accept_mask.any():  # [1, 1, 0, 1]  ~accept_mask: [0, 0, 1, 0]
                    print(f'拒绝掩码：{~accept_mask}')
                    # 获取被拒绝（舍弃）token对应的索引
                    reject_token_index = (~accept_mask).nonzero(as_tuple=True)[1]
                    print(f'拒绝token的索引：{reject_token_index}')
                    # 如果有需要舍弃的token
                    if reject_token_index.shape[0] > 0:
                        
                        # 找出第一个被舍弃的token的索引，其之前的token是需要保留的，之后的全部舍弃
                        # 接受token的数量即是第一个被舍弃的token的索引
                        accept_num = reject_token_index[0]
                    
                    else:
                        # 如果没有需要舍弃的token，则全部接受
                        accept_num = speculative_tokens.shape[1]
                        
                    
                
                else:
                    accept_num = 0      
                
                
                if accept_num > 0:
                    
                   # 取出通过验证的token
                    accept_tokens = speculative_tokens[:, :accept_num]
                   
                    seq = torch.cat([seq, accept_tokens], dim=1)
                
                else:
                    logits = outputs['head_main']
                    
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.argmax(probs, dim=-1)
                    next_token = next_token.unsqueeze(0)
                
                    
                    seq = torch.cat([seq, next_token], dim=-1)
                    # print(seq)
                    
                
            return seq
            
            
        
        
                    
        
def train(config, model, dataloader, optimizer, writer, device, epochs, print_step, save_step, save_path):
    steps = 0
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            
            main_hidden_output, main_head_output = model.forward_main(input_ids)
            previous_hidden_output = main_hidden_output
            for index in range(0, config.predict_tokens_num-1):
                previous_hidden_output, mtp_head_output = model.forward_mtp(input_ids, previous_hidden_output, index)
                
                mtp_head_output = mtp_head_output[:, :-(1+index+1)] # [batch_size, seq_len, vocab_size]
                mtp_head_output = mtp_head_output.reshape(-1, model.main_model.config.vocab_size) # [batch_size * seq_len, vocab_size]
                
                target = labels[:, 1+index+1:] # [batch_size, seq_len]
                target = target.contiguous().view(-1) # [batch_size * seq_len]
                
                mtp_loss = F.cross_entropy(mtp_head_output, target, ignore_index=-100)
                
                mtp_loss.backward(retain_graph=True)
                
            main_loss = F.cross_entropy(main_head_output[:, :-1].reshape(-1, model.main_model.config.vocab_size), labels[:, 1:].reshape(-1), ignore_index=-100)
            
            main_loss.backward()
            
            optimizer.step()
            
            if (steps+1) % print_step==0:
                writer.add_scalar('main_loss', main_loss.item(), steps)
                writer.add_scalar('mtp_loss', mtp_loss.item(), steps)
                print(f"Epoch {epoch+1}], Step {steps+1}, main_loss: {main_loss.item():.4f}, mtp_loss: {mtp_loss.item():.4f}")
                
            if (steps+1) % save_step==0:
                torch.save(model.state_dict(), f"{save_path}/model_{steps}.pth")
            
            steps += 1  
        
    
class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.data_path = data_path
        
        self.tokenizer = tokenizer
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = f.readlines()

            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index].strip()
        sample = json.loads(sample)
        conversations = sample['conversations']
        user = conversations[0]['content']
        assistant = conversations[1]['content']
        
        q = self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        
        a = assistant + self.tokenizer.eos_token
        q_input_ids = self.tokenizer(q)['input_ids']
        a_input_ids = self.tokenizer(a)['input_ids']
        
        input_ids = q_input_ids + a_input_ids
        
        labels = [-100] * len(q_input_ids) + a_input_ids
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
        
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)}
        

            
        
        
if __name__ == '__main__':
    # 日志记录
    writer = SummaryWriter('./runs')
    config = Config()
    model = MTP(config)
    model.cuda()
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    dataset = MyDataset('/home/user/wyf/deepseek_learn/MTP_train/lora_medical.jsonl', tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=2, collate_fn=MyDataCollator(tokenizer))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    save_path = './mtp'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train(config, model, dataloader, optimizer, writer, device='cuda', epochs=10, print_step=10, save_step=500, save_path='mtp')
    