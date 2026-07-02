import math
from dataclasses import dataclass

import torch
import torch.nn as nn



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


@dataclass
class AttentionConfig:
    dim: int = 256
    head_dim: int = 64
    window_size: int = 128
    compress_ratio: int = 4
    history_topk: int = 16


class Compressor(nn.Module):

    def __init__(self, dim: int, head_dim: int, compress_ratio: int = 4):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        # compress_ratio=4时会overlap
        self.overlap = compress_ratio == 4

        coeff = 1 + self.overlap
        # overlap为true时，shape：[dim, 2* head_dim]，前head_dim：作为下一块的overlap，后head_dim：用于当前块压缩
        self.wkv = nn.Linear(dim, coeff * head_dim)
        self.wgate = nn.Linear(dim, coeff * head_dim)
        self.ape = nn.Parameter(torch.zeros(compress_ratio, coeff * head_dim))
        self.norm = RMSNorm(head_dim)

        self.current_kv = []
        self.current_score = []
        self.prev_block_kv = None
        self.prev_block_score = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        cutoff = (seqlen // ratio) * ratio

        # 如果序列过短，不够一个完整的block
        if cutoff == 0:
            return x.new_zeros(bsz, 0, self.head_dim), x

        
        kv = self.wkv(x[:, :cutoff]) # shape：[b, cutoff, coeff * head_dim]
        score = self.wgate(x[:, :cutoff]) # shape：[b, cutoff, coeff * head_dim]
        kv = kv.view(bsz, cutoff // ratio, ratio, -1) # shape：[b, num_blocks, ratio, coeff * head_dim]
        score = score.view(bsz, cutoff // ratio, ratio, -1) # shape：[b, num_blocks, ratio, coeff * head_dim]
        score = score + self.ape.view(1, 1, ratio, -1) # shape：[b, num_blocks, ratio, coeff * head_dim]

        outputs = []
        self.prev_block_kv = None
        self.prev_block_score = None
        for block_idx in range(kv.size(1)):
            kv_block = kv[:, block_idx]
            score_block = score[:, block_idx]

            if not self.overlap:
                weights = torch.softmax(score_block, dim=1)
                pooled = (kv_block * weights).sum(dim=1)
                outputs.append(self.norm(pooled))
            
            else:

                main_kv = kv_block[..., self.head_dim :]
                main_score = score_block[..., self.head_dim :]

                if self.prev_block_kv is None:
                    overlap_kv = torch.zeros_like(main_kv)
                    overlap_score = torch.full_like(main_score, float("-inf"))
                else:
                    overlap_kv = self.prev_block_kv[..., : self.head_dim]
                    overlap_score = self.prev_block_score[..., : self.head_dim]

                mixed_kv = torch.cat([overlap_kv, main_kv], dim=1)
                mixed_score = torch.cat([overlap_score, main_score], dim=1)
                weights = torch.softmax(mixed_score, dim=1)
                pooled = (mixed_kv * weights).sum(dim=1)

                self.prev_block_kv = kv_block[:, -ratio:].detach()
                self.prev_block_score = score_block[:, -ratio:].detach()
                outputs.append(self.norm(pooled))

        compressed = torch.stack(outputs, dim=1)

        # 剩余不够一个block的token直接返回
        remainder = x[:, cutoff:]
        return compressed, remainder

class Indexer(nn.Module):

    def __init__(self, dim: int, head_dim: int, topk: int = 16):
        super().__init__()
        self.topk = topk
        self.wq = nn.Linear(dim, head_dim)

    def forward(self, x: torch.Tensor, compressed_history: torch.Tensor) -> torch.Tensor:
        
        # 如果无历史压缩，返回空索引
        if compressed_history.size(1) == 0:
            return torch.full(
                (x.size(0), x.size(1), 0),
                -1,
                device=x.device,
                dtype=torch.long,
            )

        # 单头
        q = self.wq(x)
        # 注意力计算
        scores = torch.einsum("bsd,bnd->bsn", q, compressed_history)
        k = min(self.topk, compressed_history.size(1))

        # 返回注意力分数最高的topk个块对应的索引
        return scores.topk(k, dim=-1).indices


class CompressedAttention(nn.Module):
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.wq = nn.Linear(config.dim, config.head_dim)
        self.wk = nn.Linear(config.dim, config.head_dim)
        self.wv = nn.Linear(config.dim, config.head_dim)
        self.compressor = Compressor(
            dim=config.dim,
            head_dim=config.head_dim,
            compress_ratio=config.compress_ratio
        )
        self.indexer = None
        if config.compress_ratio == 4:
            self.indexer = Indexer(config.dim, config.head_dim, config.history_topk)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        
        bsz, seqlen, _ = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        compressed_history, _ = self.compressor(x)
        if self.config.compress_ratio == 4:
            selected_history = self.indexer(x, compressed_history)
        elif self.config.compress_ratio == 128:
            n_blocks = compressed_history.size(1)
            if n_blocks == 0:
                selected_history = torch.full((bsz, seqlen, 0), -1, device=x.device, dtype=torch.long)
            else:
                block_ids = torch.arange(n_blocks, device=x.device)
                token_positions = torch.arange(seqlen, device=x.device)

                # 当前token只能看到之前的block
                # token_positions.unsqueeze(-1) = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
                # block_end_positions = [4, 8]
                # [
                # [False, False],  # t=0
                # [False, False],  # t=1
                # [False, False],  # t=2
                # [False, False],  # t=3
                # [ True, False],  # t=4
                # [ True, False],  # t=5
                # [ True, False],  # t=6
                # [ True, False],  # t=7
                # [ True,  True],  # t=8
                # [ True,  True],  # t=9
                # ]

                visible_blocks = token_positions.unsqueeze(-1) >= (
                    (block_ids + 1) * self.config.compress_ratio
                )

                selected_history = torch.full((bsz, seqlen, n_blocks), -1, device=x.device, dtype=torch.long)
                expanded_ids = block_ids.view(1, 1, n_blocks).expand(bsz, seqlen, n_blocks)
                mask = visible_blocks.unsqueeze(0).expand(bsz, seqlen, n_blocks)
                # 如果block对当前token可见，记录block id
                selected_history[mask] = expanded_ids[mask]
        else:
            raise ValueError(
                "compress_ratio must be in {4, 128}."
            )

        outputs = []
        scale = 1.0 / math.sqrt(self.config.head_dim)
        for t in range(seqlen):
            left = max(0, t + 1 - self.config.window_size)
            recent_k = k[:, left : t + 1]
            recent_v = v[:, left : t + 1]

            idx = selected_history[:, t]
            chosen_k = []
            for b in range(bsz):
                valid = idx[b][idx[b] >= 0]
                if len(valid) == 0:
                    chosen_k.append(compressed_history.new_zeros(0, self.config.head_dim))
                else:
                    chosen_k.append(compressed_history[b, valid])

            max_hist = max(tensor.size(0) for tensor in chosen_k)
            hist_k = compressed_history.new_zeros(bsz, max_hist, self.config.head_dim)
            hist_v = compressed_history.new_zeros(bsz, max_hist, self.config.head_dim)
            for b in range(bsz):
                n = chosen_k[b].size(0)
                if n > 0:
                    hist_k[b, :n] = chosen_k[b]
                    hist_v[b, :n] = chosen_k[b]

            memory_k = torch.cat([recent_k, hist_k], dim=1)
            memory_v = torch.cat([recent_v, hist_v], dim=1)

            score = torch.einsum("bd,bnd->bn", q[:, t], memory_k) * scale
            prob = score.softmax(dim=-1)
            out_t = torch.einsum("bn,bnd->bd", prob, memory_v)
            outputs.append(out_t)

        return torch.stack(outputs, dim=1), {
            "compressed_history": compressed_history,
            "selected_history": selected_history,
        }


if __name__ == "__main__":

    torch.manual_seed(0)
    x = torch.randn(2, 256, 32)

    csa_cfg = AttentionConfig(
        dim=32,
        head_dim=8,
        window_size=16,
        compress_ratio=4,
        history_topk=2,
    )
    csa = CompressedAttention(csa_cfg)
    csa_output, csa_aux = csa(x[:, :32])
    print("csa output shape:", tuple(csa_output.shape))
    print("csa compressed history shape:", tuple(csa_aux["compressed_history"].shape))
    print("csa history indices shape:", tuple(csa_aux["selected_history"].shape))

    hca_cfg = AttentionConfig(
        dim=32,
        head_dim=8,
        window_size=16,
        compress_ratio=128
    )
    hca = CompressedAttention(hca_cfg)
    hca_output, hca_aux = hca(x)
    print("heavy output shape:", tuple(hca_output.shape))
    print("heavy compressed history shape:", tuple(hca_aux["compressed_history"].shape))
    print("heavy history indices shape:", tuple(hca_aux["selected_history"].shape))
