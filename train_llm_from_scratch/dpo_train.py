from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F
from dataset import DPODataset, DPODataCollator
from train import LLM, Config


def logits_to_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs

def mask_logits(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels_masks shape: (batch_size, seq_len)
    new_logits = []
    for logit, label in zip(logits, labels):
        new_logits.append(logit[label != 0].sum().unsqueeze(0))
    
    return new_logits


def dpo_loss(ref_probs, probs, beta):
    def split_probs(probs):
        len_chosen = int(len(probs) // 2)
        chosen_data = probs[:len_chosen]
        reject_data = probs[len_chosen:]
        return torch.cat(chosen_data), torch.cat(reject_data)
    
    ref_chosen_probs, ref_reject_probs = split_probs(ref_probs)
    chosen_probs, reject_probs = split_probs(probs)
    pi_logratios = chosen_probs - reject_probs
    ref_logratios = ref_chosen_probs - ref_reject_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta*logits)
    return loss.mean()
    


class DPOTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids, labels = labels).logits
        ref_probs = logits_to_probs(ref_logits, labels)
        ref_probs = mask_logits(ref_probs, labels)
        logits = model(input_ids=input_ids, labels = labels).logits
        probs = logits_to_probs(logits, labels)
        probs = mask_logits(probs, labels)
        loss = dpo_loss(ref_probs, probs, 0.1)
        return loss

    # def training_step(
    #     self, model, inputs, num_items_in_batch=None
    # ) -> torch.Tensor:
    #     input_ids = inputs['input_ids']
    #     labels = inputs['labels']
    #     with torch.no_grad():
    #         ref_logits = ref_model(input_ids=input_ids, labels = labels).logits
    #     ref_probs = logits_to_probs(ref_logits, labels)
    #     ref_probs = mask_logits(ref_probs, labels)
    #     # 因为参考模型的累计概率不发生变化，为了尽量减少多次计算，计算一次参考模型的累积概率，多训练几次需要优化的模型
    #     for _ in range(1):
            
    #         model.train()
    #         logits = model(input_ids=input_ids, labels = labels).logits
    #         probs = logits_to_probs(logits, labels)
    #         probs = mask_logits(probs, labels)
        
    #         if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
    #             self.optimizer.train()

    #         with self.compute_loss_context_manager():
    #             # loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
    #             loss = dpo_loss(ref_probs, probs, 0.2)

    #         # del inputs
    #         if (
    #             self.args.torch_empty_cache_steps is not None
    #             and self.state.global_step % self.args.torch_empty_cache_steps == 0
    #         ):
                
    #             torch.cuda.empty_cache()

    #         kwargs = {}

    #         if self.args.n_gpu > 1:
    #             loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #         self.accelerator.backward(loss, retain_graph=True, **kwargs)
    #     # Finally we need to normalize the loss for reporting
    #     if num_items_in_batch is None:
    #         return loss.detach() / self.args.gradient_accumulation_steps
    #     return loss.detach()
    
        
if __name__ == "__main__":
    AutoConfig.register("small_model", Config)
    AutoModelForCausalLM.register(Config, LLM)
    model = AutoModelForCausalLM.from_pretrained('/home/user/wyf/train_model_from_scratch/saves/sft')

    print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    ref_model = AutoModelForCausalLM.from_pretrained('/home/user/wyf/train_model_from_scratch/saves/sft').eval().to('cuda')
    
    tokenizer = AutoTokenizer.from_pretrained("/home/user/wyf/train_model_from_scratch/tokenizer", use_fast=True)
    data_collator = DPODataCollator(tokenizer, max_seq_len=512) # 加载的大模型旋转位置编码最大长度为1024，这里不能超过这个值
    args = TrainingArguments(output_dir='./dpo-1-epoch', 
                            num_train_epochs=1,  # 训练太多轮，模型似乎会输出很多重复内容
                            do_train=True, 
                            per_device_train_batch_size=16,
                            gradient_accumulation_steps=4,
                            # max_steps=15000,
                            logging_steps=50,
                            report_to='tensorboard',
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=0.00001,  # 学习率很重要，太大会把模型训飞
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False,
                            save_steps=100)          
    dataset = DPODataset('/home/user/wyf/train_model_from_scratch/dataset/dpo_data_512.json', tokenizer=tokenizer)
    trainer = DPOTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/dpo-1-epoch')
    trainer.save_state()