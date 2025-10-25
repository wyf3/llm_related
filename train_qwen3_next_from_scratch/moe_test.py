from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import time
from pretrain import LLM, Config
tokenizer = AutoTokenizer.from_pretrained('/home/user/wyf/train_qwen3_next_from_scratch/result_pretrain/checkpoint-5000')
AutoConfig.register("moe_model", Config)
AutoModelForCausalLM.register(Config, LLM)
model = AutoModelForCausalLM.from_pretrained('/home/user/wyf/train_qwen3_next_from_scratch/result_pretrain/checkpoint-5000')

input_data = [tokenizer.bos_token_id] + tokenizer.encode('你是一个')
print(input_data)

t1 = time.time()
for token in model.generate({"input_ids":torch.tensor(input_data).unsqueeze(0)}, tokenizer.eos_token_id, 500, stream=False,temperature=0.0, top_k=1):
    print(tokenizer.decode(token[0]))
    time_diff = time.time()-t1
    print(len(token[0])/time_diff)

