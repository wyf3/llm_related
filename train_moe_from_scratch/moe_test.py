from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from moe_train import LLM, Config
t = AutoTokenizer.from_pretrained('./saves/moe')
AutoConfig.register("moe_model", Config)
AutoModelForCausalLM.register(Config, LLM)
model = AutoModelForCausalLM.from_pretrained('./saves/moe')

input_data = [t.bos_token_id] + t.encode('1+1等于')
print(input_data)

for token in model.generate({"input_ids":torch.tensor(input_data).unsqueeze(0), "labels":None}, t.eos_token_id, 20, stream=False,temperature=0.0, top_k=1):
    print(t.decode(token[0]))