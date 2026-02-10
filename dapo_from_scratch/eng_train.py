# Chinese comments are translated into proper English comments_::__
from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import *
import os

# Specify which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class GSM8KDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        data = load_dataset(data_path)
        self.data = data['train']
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        # prompt = self.tokenizer.apply_chat_template(sample['prompt'], tokenize=False, add_generation_prompt=True)
        answer = sample['answer_only']
        prompt = sample['question_zh-cn']
        return {'prompt': prompt, 'answer': answer}


@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: int


class GRPOArguments:
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001
    save_steps = 100
    epoch = 3
    num_generations = 4  # Number of samples per group
    max_prompt_length = 256  # Maximum input prompt length
    max_generate_length = 256  # Maximum generated output length
    reward_weights: List[float] = None  # Weights for multiple reward functions
    beta = 0.0  # KL divergence coefficient; if 0, KL is ignored and no reference model is used
    clip_eps_high = 0.28
    clip_eps_low = 0.2
    gradient_accumulation_steps = 2  # Gradient accumulation steps
    num_iterations = 1  # Number of training iterations per sampled batch
    batch_size = 1


class GRPOTrainer:
    def __init__(self,
        model=None,
        reward_funcs: Union[List[str], List[Callable]] = None,
        args=None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        tokenizer=None,
        reward_tokenizers=None):

        self.args = args

        # Load policy model
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)
        
        # Whether to use a reference model (for KL regularization)
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
    
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        
        # Load reward models if reward functions are provided as model paths
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1
                ).to(self.args.device)
        
        self.reward_funcs = reward_funcs
        
        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(reward_funcs)
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError("Length of reward_tokenizers must equal number of reward_funcs.")
        
        # Initialize reward tokenizers for reward models
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                
                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer
        
        self.reward_tokenizers = reward_tokenizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Buffer to cache generated batches for multiple training iterations
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        
        # Number of model update steps
        self.update_steps = 0 

    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        return tokenizer
    
    # Generate samples in groups
    def generate_samples(self, inputs):
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(prompts)
        
        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]
        
        max_length = self.args.max_generate_length + self.args.max_prompt_length
        
        for prompt, answer in zip(prompts, answers):
            # Apply chat template and insert system prompt
            input_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", 'content': SYSTEM_PROMPT},
                    {"role": "user", 'content': prompt}
                ],
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Create a group of identical inputs for sampling
            inputs = self.tokenizer(
                [input_text] * self.args.num_generations,
                padding='max_length',
                max_length=self.args.max_prompt_length,
                truncation=True,
                return_tensors='pt'
            )
            
            prompt_ids = inputs['input_ids']
            
            with torch.no_grad():
                prompt_response_ids = self.model.generate(
                    **inputs.to(self.args.device),
                    max_new_tokens=self.args.max_generate_length,
                    temperature=0.9,
                    top_p=1,
                    top_k=50
                )
                
            # Pad or truncate to fixed max length
            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                pad = torch.full(
                    (prompt_response_ids.size(0), max_length - prompt_response_ids.size(1)),
                    fill_value=self.tokenizer.pad_token_id,
                    device=prompt_response_ids.device
                )
                prompt_response_ids = torch.cat([prompt_response_ids, pad], dim=1)
          
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).long()
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            action_mask = (
                response_ids.ne(self.tokenizer.eos_token_id) &
                response_ids.ne(self.tokenizer.pad_token_id)
            ).long()

            # Store one group of samples
            samples = Samples(
                prompt_response_ids=prompt_response_ids,
                response_ids=response_ids,
                prompt=prompt,
                answer=answer,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                response_length=action_mask.float().sum(dim=-1)
            )
            samples_list.append(samples)

        return samples_list
    
    # Generate experiences (advantages and token log-probabilities)
    def generate_experiences(self, inputs):
        self.model.eval()
        samples_list = self.generate_samples(inputs)
        
        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []
        
        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids
            response_ids = samples.response_ids
            answer = samples.answer
            attention_mask = samples.attention_mask
            action_mask = samples.action_mask
            num_actions = samples.num_actions
            prompt = samples.prompt
            
            with torch.no_grad():
                # Store rewards from each reward function for the group
                rewards_per_func = torch.zeros(
                    len(self.reward_funcs),
                    self.args.num_generations,
                    device=self.args.device
                )
                
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * len(response_texts)
                prompt_response_texts = [
                    p + r for p, r in zip(prompt_texts, response_texts)
                ]
                
                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    if isinstance(reward_func, PreTrainedModel):
                        reward_inputs = reward_tokenizer(
                            prompt_response_texts,
                            return_tensors="pt",
                            padding=True
                        )
                        rewards_per_func[i] = reward_func(
                            **reward_inputs.to(self.args.device)
                        ).logits.squeeze(-1)
                    else:
                        answers = [answer] * len(prompt_texts)
                        output = reward_func(
                            prompts=prompt_texts,
                            responses=response_texts,
                            answers=answers
                        )
                        output = [r if r is not None else torch.nan for r in output]
                        rewards_per_func[i] = torch.tensor(
                            output,
                            dtype=torch.float32,
                            device=self.args.device
                        )
                
                # Apply reward weights
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                
                rewards = rewards_per_func * torch.tensor(
                    self.args.reward_weights,
                    device=self.args.device
                ).unsqueeze(1)
                
                rewards = rewards.sum(dim=0)
                print(f'rewards: {rewards}')
                
                mean_rewards = rewards.mean()
                std_rewards = rewards.std()
                
                # GRPO uses sentence-level advantages, not token-level
                advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)
                
                # If all advantages are zero, skip this group
                if advantages.count_nonzero().item() == 0:
                    continue
                
                batch_advantages.append(advantages)
                
                # Compute log-probabilities from policy model
                old_action_log_probs = self.get_action_log_probs(
                    self.model,
                    prompt_response_ids,
                    attention_mask,
                    num_actions
                )
                batch_old_action_log_probs.append(old_action_log_probs)
                
                # Compute reference model log-probabilities if enabled
                if self.ref_model:
                    ref_action_log_probs = self.get_action_log_probs(
                        self.ref_model,
                        prompt_response_ids,
                        attention_mask,
                        num_actions
                    )
                    batch_ref_action_log_probs.append(ref_action_log_probs)
                
                batch_prompt_response_ids.append(prompt_response_ids)
                batch_attention_mask.append(attention_mask)
                batch_action_mask.append(action_mask)
        
        return {
            "prompt_response_ids": batch_prompt_response_ids,
            "attention_mask": batch_attention_mask,
            "action_mask": batch_action_mask,
            "old_action_log_probs": batch_old_action_log_probs,
            "ref_action_log_probs": batch_ref_action_log_probs if self.ref_model else None,
            "advantages": batch_advantages,
        }
    
    def compute_loss(self, model, inputs):
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        
        action_log_probs = self.get_action_log_probs(
            model, prompt_response_ids, attention_mask, num_actions
        )
        
        if self.args.beta != 0.0:
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = (ref_action_log_probs - action_log_probs) * action_mask
            k3 = log_ratio.exp() - 1 - log_ratio
        
        advantages = inputs['advantages']
        old_action_log_probs = (
            inputs['old_action_log_probs']
            if self.args.num_iterations > 1
            else action_log_probs.detach()
        )
        
        coef_1 = torch.exp(action_log_probs - old_action_log_probs)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.args.clip_eps_low,
            1 + self.args.clip_eps_high
        )
        
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3
        
        # DAPO-style loss aggregation
        per_token_loss = per_token_loss.view(-1, self.args.num_generations, num_actions)
        action_mask = action_mask.view(-1, self.args.num_generations, num_actions)
        loss = per_token_loss.sum(-1).sum(-1) / action_mask.sum(-1).sum(-1)
        
        return loss.mean()

    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        )
        return log_probs_labels.squeeze(-1)[:, -num_actions:]
    
    def train_step(self, model, inputs, optimizer, step):
        model.train()
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar("grpo_loss", loss.item(), self.update_steps)
            print(
                f"step: {self.update_steps}/{self.global_steps}  "
                f"grpo_loss: {loss.item():.8f}"
            )
        
        torch.cuda.empty_cache()

    def train(self):
        self.global_steps = (
            self.args.num_iterations
            * self.args.epoch
            * len(self.train_dataset)
            // (self.args.batch_size * self.args.gradient_accumulation_steps)
        )
        
        for _ in range(self.args.epoch):
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True
            )
            
            buffer = {
                'prompt_response_ids': [],
                'attention_mask': [],
                'action_mask': [],
                'old_action_log_probs': [],
                'ref_action_log_probs': [],
                'advantages': []
            }
            
            idx = 0
            for batch in dataloader:
                inputs = self.generate_experiences(batch)
                for k in buffer:
                    if k in inputs and inputs[k] is not None:
                        buffer[k] += inputs[k]
                
                # If generated samples are insufficient, continue sampling
                if len(buffer['prompt_response_ids']) < self.args.batch_size:
                    continue
                
                inputs = {
                    k: torch.cat(v[:self.args.batch_size], dim=0)
                    if v is not None else None
                    for k, v in buffer.items()
                }
                
                buffer = {
                    k: v[self.args.batch_size:] if v is not None else None
                    for k, v in buffer.items()
                }
                
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs
                
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        if self.update_steps % self.args.save_steps == 0:
                            self.model.save_pretrained(
                                f"{self.args.output_dir}/checkpoint_{self.update_steps}"
                            )
                            self.tokenizer.save_pretrained(
                                f"{self.args.output_dir}/checkpoint_{self.update_steps}"
                            )
                
                idx += 1
                del inputs

    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)           


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    # System prompt instructing the model to answer in a structured format
    SYSTEM_PROMPT = """
Answer the question in the following format:
<think>
Your reasoning process
</think>
<answer>
Your final answer
</answer>
"""
    
    args = GRPOArguments()
    writer = SummaryWriter('./runs')
    
    # Policy model
    tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-3B-Instruct')
    model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-3B-Instruct')
    
    # Dataset
    prompts_dataset = GSM8KDataset(
        '/home/user/wyf/deepseek_learn/gsm8k_chinese',
        tokenizer
    )
  
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, digit_reward, hard_format_reward, mark_reward],
        args=args,
        train_dataset=prompts_dataset,
        tokenizer=tokenizer
    )
    
    trainer.train()
    trainer.save_model()
