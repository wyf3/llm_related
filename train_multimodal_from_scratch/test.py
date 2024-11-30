from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from train import VLMConfig, VLM

device = "cuda:1"
processor = AutoProcessor.from_pretrained("/home/user/wyf/siglip-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct')
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

model = AutoModelForCausalLM.from_pretrained('/home/user/wyf/train_multimodal_from_scratch/save/sft')
model.to(device)
q_text = tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":'描述图片内容\n<image>'}], \
            tokenize=False, \
            add_generation_prompt=True).replace('<image>', '<|image_pad|>'*49)

input_ids = tokenizer(q_text, return_tensors='pt')['input_ids']
input_ids = input_ids.to(device)
image = Image.open('/home/user/wyf/train_multimodal_from_scratch/test_images/th4.png').convert("RGB")
pixel_values = processor(text=None, images=image).pixel_values
pixel_values = pixel_values.to(device)
model.eval()
import torch
from torch.nn import functional as F
max_new_tokens = 100
temperature = 0.0
eos = tokenizer.eos_token_id
top_k = None
s = input_ids.shape[1]
while input_ids.shape[1] < s + max_new_tokens - 1:  
    inference_res = model(input_ids, None, pixel_values)  
    logits = inference_res.logits 
    logits = logits[:, -1, :] 

    for token in set(input_ids.tolist()[0]):  
        logits[:, token] /= 1.0

    if temperature == 0.0: 
        _, idx_next = torch.topk(logits, k=1, dim=-1)
    else:
        logits = logits / temperature  
        if top_k is not None:  
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf') 

        probs = F.softmax(logits, dim=-1)  
        idx_next = torch.multinomial(probs, num_samples=1, generator=None)  

    if idx_next == eos:  
        break

    input_ids = torch.cat((input_ids, idx_next), dim=1)  
print(tokenizer.decode(input_ids[:, s:][0]))