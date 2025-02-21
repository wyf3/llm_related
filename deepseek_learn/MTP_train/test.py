from MTP import MTP,Config
from transformers import AutoTokenizer
import torch
config = Config()

tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
model = MTP(config)
model.cpu()
model.load_state_dict(torch.load('/home/user/wyf/deepseek_learn/MTP_train/mtp/model_4499.pth'))

input_ids = tokenizer.apply_chat_template([{"role": "user", "content": "宝宝生下来很可爱，但是觉得嘴唇发紫，而且心跳的也很慢，新生儿心率多少正常？"}], tokenize=True, return_tensors='pt', add_generation_prompt=True)
seq = model.generate(input_ids, max_length=100)
print(tokenizer.decode(seq[0]))