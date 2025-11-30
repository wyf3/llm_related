from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from typing import Optional, Union, List, Tuple, Dict, Any
from collections.abc import Callable
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2MLP, Qwen2RMSNorm, Qwen2PreTrainedModel, Qwen2RotaryEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.utils.generic import check_model_inputs
from transformers.generation import GenerationMixin
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[List[tuple[torch.FloatTensor, ...]]] = None
    
@dataclass
class CausalLMOutputWithPast(ModelOutput):
    

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[List[tuple[torch.FloatTensor, ...]]] = None


class Indexer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.num_attention_heads
        self.key_value_heads = config.num_key_value_heads
        self.head_dim: int = config.hidden_size // config.num_attention_heads
        self.index_topk: int = 128

        self.wk = nn.Linear(self.hidden_size, self.head_dim) 
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads)

        self.register_buffer("k_cache", None, persistent=False)
        


    def forward(self, hidden_states: torch.Tensor, query_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask=None):
        
        bsz, seqlen, _ = hidden_states.size()
        key_states = self.wk(hidden_states)
 
        
        weights = self.weights_proj(hidden_states) * self.n_heads ** -0.5 # [bs, seqlen, n_heads]

        # q:[bs, n_heads, seqlen, head_dim]
        # k:[bs, seqlen, head_dim]
        
        if seqlen > 1:
            self.k_cache = key_states
        
        if seqlen == 1:
            key_states = torch.cat([self.k_cache, key_states], dim=1) # [bs, seqlen, head_dim]
            self.k_cache = key_states

        
        key_states = key_states.unsqueeze(1) # [bs, 1, seqlen, head_dim]
        key_states, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
    
        # [bs, n_heads, seqlen, head_dim] * [bs, 1, head_dim, seqlen] --> [bs, n_heads, seqlen, seqlen]
        attn_scores = query_states @ key_states.transpose(2,3)
        attn_scores = F.relu(attn_scores, inplace=False)
        
        # [bs, n_heads, seqlen, 1] * [bs, n_heads, seqlen, seqlen] --> [bs, n_heads, seqlen, seqlen]
        attn_scores = weights.transpose(1,2).unsqueeze(-1) * attn_scores
        
        attn_scores = attn_scores.sum(1, keepdim=True) # [bs, 1, seqlen, seqlen]
       
        if mask is not None:
            attn_scores = attn_scores + mask
            
        topk_indices = attn_scores.topk(min(self.index_topk, key_states.shape[2]), dim=-1)[1]
        return topk_indices, attn_scores


class Qwen2Attention(nn.Module):

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        
        self.indexer = Indexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        
  
        
        bsz, seqlen, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    

        raw_attn_weights = None
        indexer_attn_scores = None
        topk_indices = None
        # train
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.logical_not()
                attention_mask = attention_mask.float().masked_fill(attention_mask, float('-inf'))
          
    
          
            topk_indices, indexer_attn_scores = self.indexer(hidden_states, query_states, cos, sin, mask=attention_mask)
        
            raw_attn_weights = attn_weights + attention_mask
            

            index_mask = torch.full((bsz, 1, seqlen, seqlen), float("-inf"), device=hidden_states.device).scatter(-1, topk_indices, 0)
            index_mask = index_mask + attention_mask
            attn_weights = attn_weights + index_mask
            attn_weights = attn_weights.softmax(dim=-1, dtype=attn_weights.dtype)
            
        # inference
        else:
            # prefill
            if seqlen > 1:
               
                mask = torch.tril(torch.ones((bsz, 1, seqlen, seqlen), device=hidden_states.device, dtype=torch.bool), diagonal=0)
                mask = mask.logical_not()
                mask = mask.float().masked_fill(mask, float('-inf'))
               
                topk_indices, indexer_attn_scores = self.indexer(hidden_states, query_states, cos, sin, mask=mask)

                raw_attn_weights = attn_weights + mask
                
                index_mask = torch.full((bsz, 1, seqlen, seqlen), float("-inf"), device=hidden_states.device).scatter(-1, topk_indices, 0)
                index_mask = index_mask + mask
                attn_weights = attn_weights + index_mask
                
                attn_weights = attn_weights.softmax(dim=-1, dtype=attn_weights.dtype)
            
            # generate
            else:
                topk_indices, indexer_attn_scores = self.indexer(hidden_states, query_states, cos, sin, mask=None)
                index_mask = torch.full((bsz, 1, 1, key_states.shape[-2]), float("-inf"), device=hidden_states.device).scatter(-1, topk_indices, 0)
                attn_weights = attn_weights + index_mask
                attn_weights = attn_weights.softmax(dim=-1, dtype=attn_weights.dtype)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, (topk_indices, raw_attn_weights, indexer_attn_scores)



class Qwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, attentions = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attentions
  
    
class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_attentions = []
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, attentions = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                output_attentions=output_attentions,
                **kwargs,
            )
            all_attentions.append(attentions)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            attentions=all_attentions if output_attentions else None,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    
    
    tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct/')
    model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct/')
 
    messages = [{"role": "user", "content": "你好"}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize = False)
    # inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=48)
    inputs = tokenizer(text, return_tensors="pt")['input_ids']
    print(inputs)
    
    output = model.generate(inputs, do_sample=False)

    print(tokenizer.decode(output[0]))

    # for layer in model.model.layers:
    #     old_self_attn = layer.self_attn
    #     new_self_attn = Qwen2Attention(layer.self_attn.config, layer.self_attn.layer_idx)
    #     new_self_attn.load_state_dict(old_self_attn.state_dict(), strict=False)
    #     layer.self_attn = new_self_attn
    
    model = Qwen2ForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct/')
    output = model.generate(inputs, do_sample=False)

    print(tokenizer.decode(output[0]))
    

    

    
    