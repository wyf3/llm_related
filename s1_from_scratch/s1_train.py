from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import Dataset
import torch

THINK_TAG = '<|im_start|>think'
ANSWER_TAG = '<|im_start|>answer'
END_TAG = '<|im_end|>'

class S1Dataset(Dataset):
    def __init__(self, ds, tokenizer, max_length=4096):
        self.ds = ds
        self.max_length = max_length
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        sample = self.ds[index]
        question = sample['question']
        gemini_thinking_trajectory = sample['gemini_thinking_trajectory']
        gemini_attempt = sample['gemini_attempt']
        
        q = self.tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
        a = THINK_TAG + gemini_thinking_trajectory + ANSWER_TAG + gemini_attempt + END_TAG
        
        q_input_ids = self.tokenizer.encode(q)
        a_input_ids = self.tokenizer.encode(a)
        
        input_ids = q_input_ids + a_input_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(q_input_ids) + a_input_ids
        
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
        else:
            padding_len = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            labels = labels + [-100] * padding_len
            
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "attention_mask": torch.tensor(attention_mask, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}

    def __len__(self):
        return len(self.ds)

if __name__ == "__main__":

    model = AutoModelForCausalLM.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("/home/user/Downloads/Qwen2.5-0.5B-Instruct")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = load_dataset('./s1K-1.1')
    data_collator = DefaultDataCollator()
    
    args = TrainingArguments(
        output_dir="./s1",
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to="tensorboard",
        bf16=True
    )
    
    train_dataset = S1Dataset(ds['train'], tokenizer, max_length=1024)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()
    trainer.save_model()
    trainer.save_state()
