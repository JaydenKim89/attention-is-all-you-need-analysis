"""A small encoder-decoder Transformer built from explicit tensor operations.

Follows the paper (arXiv 1706.03762v7) where stated: post-normalization
LayerNorm(x + Dropout(Sublayer(x))) (Sec. 3.1, 5.4), ReLU FFN (Eq. 2), sinusoidal
positional encoding (Sec. 3.5), embeddings shared between encoder, decoder and the
pre-softmax projection and scaled by sqrt(d_model) (Sec. 3.4), dropout on
sub-layer outputs and on embedding + position sums (Sec. 5.4).

Project choices (not specified, or changed, relative to the paper):
  * attention projections have no bias (the paper's equations show none);
  * no dropout on attention probabilities;
  * embedding init N(0, d_model^-0.5), Xavier-uniform linear weights;
  * per-side switches enc_pos / dec_pos to remove positional encoding (ablation).
nn.Transformer and nn.MultiheadAttention are not used.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    dropout: float = 0.1
    max_len: int = 64
    enc_pos: bool = True
    dec_pos: bool = True

    def to_dict(self):
        return asdict(self)


def sinusoidal_encoding(max_len, d_model):
    """PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(pos / 10000^(2i/d))."""
    pos = torch.arange(max_len, dtype=torch.float64)[:, None]
    i2 = torch.arange(0, d_model, 2, dtype=torch.float64)[None, :]
    angle = pos / torch.pow(10000.0, i2 / d_model)
    pe = torch.zeros(max_len, d_model, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.record = False
        self.last_attn = None

    def _split(self, x):  # (B, L, d_model) -> (B, h, L, dk)
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, xq, xkv, allowed):
        """allowed: bool, broadcastable to (B, h, Lq, Lk); True = may attend."""
        q, k, v = self._split(self.w_q(xq)), self._split(self.w_k(xkv)), self._split(self.w_v(xkv))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dk)          # (B, h, Lq, Lk)
        allowed = allowed.expand_as(scores)
        if not bool(allowed.any(dim=-1).all()):
            raise ValueError("a query row has no allowed key (all-masked row)")
        probs = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        if self.record:
            self.last_attn = probs.detach()
        out = (probs @ v).transpose(1, 2).reshape(xq.shape[0], xq.shape[1], -1)  # concat heads
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.lin2(F.relu(self.lin1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff)
        self.norm1, self.norm2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, src_allowed):
        x = self.norm1(x + self.drop(self.self_attn(x, x, src_allowed)))  # post-norm (Sec. 3.1)
        return self.norm2(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(cfg.d_model) for _ in range(3))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, y, memory, tgt_allowed, cross_allowed):
        y = self.norm1(y + self.drop(self.self_attn(y, y, tgt_allowed)))
        y = self.norm2(y + self.drop(self.cross_attn(y, memory, cross_allowed)))
        return self.norm3(y + self.drop(self.ffn(y)))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)  # shared (Sec. 3.4)
        self.register_buffer("pe", sinusoidal_encoding(cfg.max_len, cfg.d_model).float(), persistent=False)
        self.enc_layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_enc_layers))
        self.dec_layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_dec_layers))
        self.drop = nn.Dropout(cfg.dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=self.cfg.d_model ** -0.5)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _embed(self, tokens, use_pos):
        x = self.embed(tokens) * math.sqrt(self.cfg.d_model)
        if use_pos:
            if tokens.shape[1] > self.cfg.max_len:
                raise ValueError("sequence longer than max_len")
            x = x + self.pe[: tokens.shape[1]].to(x.dtype)
        return self.drop(x)

    # Boolean "allowed" masks, shaped to broadcast over (B, h, Lq, Lk).
    @staticmethod
    def src_mask(src):
        """Keys are the non-PAD source positions; every query row sees the same keys."""
        return (src != PAD)[:, None, None, :]

    @staticmethod
    def tgt_mask(tgt_in):
        """Causal (key j <= query i) and non-PAD keys. BOS at j = 0 keeps every row non-empty."""
        Lt = tgt_in.shape[1]
        causal = torch.ones(Lt, Lt, dtype=torch.bool, device=tgt_in.device).tril()
        return causal[None, None] & (tgt_in != PAD)[:, None, None, :]

    def encode(self, src):
        src_allowed = self.src_mask(src)
        x = self._embed(src, self.cfg.enc_pos)
        for layer in self.enc_layers:
            x = layer(x, src_allowed)
        return x, src_allowed

    def decode(self, tgt_in, memory, src_allowed):
        tgt_allowed = self.tgt_mask(tgt_in)
        y = self._embed(tgt_in, self.cfg.dec_pos)
        for layer in self.dec_layers:
            y = layer(y, memory, tgt_allowed, src_allowed)
        return y @ self.embed.weight.T                              # tied output projection

    def forward(self, src, tgt_in):
        """Teacher-forced logits (B, Lt, V) for next-token prediction."""
        memory, src_allowed = self.encode(src)
        return self.decode(tgt_in, memory, src_allowed)

    @torch.no_grad()
    def greedy_decode(self, src, max_new_tokens):
        """Autoregressive greedy decoding from BOS; stops when every row has emitted EOS.

        Returns a list of token lists, each truncated before its first EOS, and a
        list of booleans saying whether EOS was emitted within max_new_tokens.
        """
        memory, src_allowed = self.encode(src)
        ys = torch.full((src.shape[0], 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
        for _ in range(max_new_tokens):
            nxt = self.decode(ys, memory, src_allowed)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)  # keep finished rows padded
            ys = torch.cat([ys, nxt[:, None]], dim=1)
            done |= nxt == EOS
            if bool(done.all()):
                break
        outs, ended = [], []
        for row in ys[:, 1:].tolist():
            if EOS in row:
                outs.append(row[: row.index(EOS)])
                ended.append(True)
            else:
                outs.append([t for t in row if t != PAD])
                ended.append(False)
        return outs, ended


def token_cross_entropy(logits, tgt_out):
    """Mean cross-entropy over non-PAD target positions in the batch (EOS included)."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1), ignore_index=PAD)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
