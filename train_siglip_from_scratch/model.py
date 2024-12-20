from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoTokenizer, AutoProcessor
from transformers import ViTImageProcessor, ViTForImageClassification

import torch.nn as nn
from transformers.utils import ModelOutput
import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class SiglipOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    logits_per_image: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    



class SiglipConfig(PretrainedConfig):
    model_type = "siglip"
    def __init__(
        self,
        vision_model_name_or_path: str = "vit-base-patch16-224",
        text_model_name_or_path: str = "bert-base-chinese",
        **kwargs):
        super().__init__(**kwargs)
        self.vision_model_name_or_path = vision_model_name_or_path
        self.text_model_name_or_path = text_model_name_or_path
        
        
        
class SiglipModel(PreTrainedModel):
    config_class = SiglipConfig
    def __init__(self, config: SiglipConfig):
        super().__init__(config)
        self.vision_model = AutoModel.from_pretrained(config.vision_model_name_or_path)
        self.process = AutoProcessor.from_pretrained(config.vision_model_name_or_path)
        self.text_model = AutoModel.from_pretrained(config.text_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name_or_path)
        self.t = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        
        
    def forward(self, input_ids, attention_mask, pixel_values):
        
        
        text_outputs = self.text_model(input_ids, attention_mask)
        vision_outputs = self.vision_model(pixel_values)
        
        
        vision_features = vision_outputs[1] # pooler_output
        text_features = text_outputs[1] # pooler_output
        
        vision_features = vision_features / vision_features.norm(p=2, dim=-1, keepdim=True) # l2标准化
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True) # l2标准化
        
        logits_per_text = torch.matmul(text_features, vision_features.t()) * self.t.exp() + self.b
        logits_per_image = logits_per_text.t()
        
        b = logits_per_text.shape[0]
        eye = torch.eye(b, device=logits_per_text.device) # 生成单位矩阵
        labels = 2*eye - torch.ones_like(logits_per_text, device=logits_per_text.device) # 对角线全为1，非对角线为-1，即成对的图文标签为1，非成对的为-1
        loglik = F.logsigmoid(labels * logits_per_text)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
        
        
        return SiglipOutput(loss=loss, logits_per_text=logits_per_text, logits_per_image=logits_per_image, text_embeds=text_features, image_embeds=vision_features)
    
    
    
    