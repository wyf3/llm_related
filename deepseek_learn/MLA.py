import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# rms归一化
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.float()
    
    
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
   
    q_embed = (q*cos) + (rotate_half(q)*sin)
    k_embed = (k*cos) + (rotate_half(k)*sin)
    
    return q_embed, k_embed

# 旋转位置编码
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0)
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
        
    def forward(self, q, k):
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)    

class MLA(nn.Module):
    def __init__(self,
                dim,
                n_heads,
                q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                max_seq_len,
                max_batch_size,
                mode):
        super().__init__()
        self.dim = dim # 隐藏层维度
        self.n_heads = n_heads  #总头数
        self.q_lora_rank = q_lora_rank # q低秩压缩到的维度
        self.kv_lora_rank = kv_lora_rank # kv低秩压缩到的维度
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim # qk的总维度，不带旋转位置编码的维度加上带旋转位置编码的维度
        self.v_head_dim = v_head_dim # value的维度，等于不带旋转位置编码的k维度
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        
        
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank) # q的降维矩阵
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim) # q的升维矩阵
        # 4096*128+128*4864 = 524,288 + 622592 = 1146880    4096*4864 = 19,922,944
        
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim) # kv的降维矩阵
        # nn.Linear(self.dim, self.kv_lora_rank)
        # nn.Linear(self.dim, self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)) # kv的升维矩阵
        
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        
        self.rotary_emb = RotaryEmbedding(self.qk_rope_head_dim) # 旋转旋转位置编码
        
        if self.mode == 'naive':
            self.register_buffer('k_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.n_heads, self.qk_head_dim), persistent=False)
            self.register_buffer('v_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.n_heads, self.v_head_dim), persistent=False)
            
        else:
            self.register_buffer('kv_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer('pe_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.qk_rope_head_dim), persistent=False)
            
        
    def forward(self, x, mask=None):
        
        bs, seq_len, _ = x.shape
        
        q = self.wq_a(x)  # [bs, seq_len, q_lora_rank]
        q = self.q_norm(q) # [bs, seq_len, q_lora_rank]
        q = self.wq_b(q) # [bs, seq_len, n_heads * qk_head_dim]
        q = q.view(bs, seq_len, self.n_heads, self.qk_head_dim) # [bs, seq_len, n_heads, qk_head_dim]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # q_nope shape:[bs, seq_len, n_heads, qk_nope_head_dim] q_pe shape:[bs, seq_len, n_heads, qk_rope_head_dim]
        
        kv = self.wkv_a(x) # [bs, seq_len, kv_lora_rank + qk_rope_head_dim]
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1) # kv shape:[bs, seq_len, kv_lora_rank] k_pe shape:[bs, seq_len, qk_rope_head_dim]
        
        k_pe = k_pe.unsqueeze(2) # k_pe shape:[bs, seq_len, 1, qk_rope_head_dim]
        q_pe, k_pe = self.rotary_emb(q_pe, k_pe)
        if self.mode == 'naive':
            
            q = torch.cat([q_nope, q_pe], dim=-1) # * [bs, seq_len, n_heads, qk_head_dim]
            
            kv = self.kv_norm(kv) # [bs, seq_len, kv_lora_rank)]
            kv = self.wkv_b(kv) # [bs, seq_len, n_heads * (qk_nope_head_dim + v_head_dim)]
            kv = kv.view(bs, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            
            k = torch.cat([k_nope, k_pe.expand(-1,-1,self.n_heads,-1)], dim=-1) 
            # k shape:[bs, seq_len, n_heads, qk_head_dim]
            self.k_cache[:bs, :seq_len, :, :] = k
            self.v_cache[:bs, :seq_len, :, :] = v
            # scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bs, :seq_len]) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
            scores = torch.matmul(q.transpose(1, 2), self.k_cache[:bs, :seq_len, :, :].transpose(1, 2).transpose(2, 3) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim))
            scores = scores.transpose(1, 2)
            
        else:
            k_pe = k_pe.squeeze(2)
            wkv_b = self.wkv_b.weight  # [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
            wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank) # [n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank]
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim]) # q_nope shape:[bs, seq_len, n_heads, kv_lora_rank]
            # q*k(T) = x*wq*(c*wkv_b[:, :self.qk_nope_head_dim])(T) = x*wq*wkv_b[:, :self.qk_nope_head_dim](T)*c(T)    c为压缩后的kv
            # wq*wkv_b[:, :self.qk_nope_head_dim](T)作为q的投影矩阵  c可以替代原先的k，这样就可以直接使用压缩后的kv计算注意力了，kv_caceh时也只需存储压缩后的kv
            kv = self.kv_norm(kv)
            self.kv_cache[:bs, :seq_len, :] = kv # kv shape:[bs, seq_len, kv_lora_rank]
            self.pe_cache[:bs, :seq_len, :] = k_pe # k_pe shape:[bs, seq_len, qk_rope_head_dim]
            
            scores_nope = torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bs, :seq_len, :]) # bshc btc -> bshc bct -> bsht
            scores_pe = torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bs, :seq_len, :])  # bshr btr -> bshr bt1r -> bshr bthr -> bsht
            scores = (scores_nope + scores_pe) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim) # [bs, seq_len, n_heads, seq_len]
        
        if mask is not None:
            # mask shape:[bs, seq_len, seq_len]
            scores += mask.unsqueeze(2)
        
        scores = scores.softmax(dim=-1)
       
        if self.mode == 'naive':
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bs, :seq_len]) # bsht,bthd -> bhst, bhtd -> bhsd -> bshd
        else:
            
            # scores * v = scores * c * wkv_b[:, -self.v_head_dim:]
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bs, :seq_len]) # x shape:[bs, seq_len, n_heads, kv_lora_rank]
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:]) # bshc, hdc -> bshc,dch -> bsdh -> bshd
    
        x = x.contiguous ().view(bs, seq_len, -1)
        x = self.wo(x)
        
        return x

if __name__ == '__main__':
   
    x = torch.randn(4, 100, 4096)
    
    dim = 4096
    n_heads = 16
    q_lora_rank = 128
    kv_lora_rank = 64
    qk_nope_head_dim = 256
    qk_rope_head_dim = 48
    v_head_dim = 256
    max_seq_len = 512
    max_batch_size = 16
    mode = 'none'

    mla = MLA(dim=dim, 
            n_heads=n_heads, 
            q_lora_rank=q_lora_rank, 
            kv_lora_rank=kv_lora_rank, 
            qk_nope_head_dim=qk_nope_head_dim, 
            qk_rope_head_dim=qk_rope_head_dim, 
            v_head_dim=v_head_dim, 
            max_seq_len=max_seq_len, 
            max_batch_size=max_batch_size, 
            mode=mode)
    
    
    
    print(mla(x))
    print(mla.kv_cache)