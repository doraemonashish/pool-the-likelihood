"""A small, fully inspectable decoder-only transformer.

Design choices made for interpretability rather than speed:
* attention is computed explicitly, so every attention pattern can be returned;
* rotary embeddings can use token *indices* (ordinary sequences) or real-valued *times*
  (event sequences), so attention logits depend only on index/time differences;
* an optional learned "null slot" per head lets a head put mass on nothing, which lets it
  compute un-normalised sums (useful for point processes);
* ``edit`` lets you modify the residual stream after any block (activation patching/steering).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 512
    vocab_size: int = 0  # >0 -> token inputs
    n_features: int = 0  # >0 -> continuous feature inputs (added to token embedding if both)
    n_out: int = 3
    max_len: int = 512  # only used for learned absolute positions
    pos: str = "rope"  # "rope" | "learned" | "none"
    rope_base: float = 10000.0
    rope_freqs: Optional[list] = None  # explicit angular frequencies (for continuous time)
    null_attn: bool = False
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


class Rotary(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, freqs: Optional[list] = None):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for rotary embeddings"
        n = head_dim // 2
        if freqs is not None:
            f = torch.tensor(list(freqs), dtype=torch.float32)
            if f.numel() < n:  # pad with zeros -> those dims are not rotated (content-only)
                f = torch.cat([f, torch.zeros(n - f.numel())])
            f = f[:n]
        else:
            f = 1.0 / (base ** (torch.arange(0, n, dtype=torch.float32) / n))
        self.register_buffer("inv_freq", f, persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, hd); pos: (B, T) float
        ang = pos.float()[:, None, :, None] * self.inv_freq[None, None, None, :]
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.rope = Rotary(self.hd, cfg.rope_base, cfg.rope_freqs) if cfg.pos == "rope" else None
        self.null = nn.Parameter(torch.zeros(self.h)) if cfg.null_attn else None
        self.drop = nn.Dropout(cfg.dropout)
        # interpretability hooks (inactive by default): capture per-head outputs, or replace chosen heads' outputs
        self.capture_heads = False
        self.head_out = None                  # (B, H, T, hd) when capture_heads
        self.head_replace: dict = {}          # head -> tensor (hd,) (mean ablation) or None (zero ablation)

    def forward(self, x, pos, key_mask, want_attn: bool, attn_mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)
        k = k.view(B, T, self.h, self.hd).transpose(1, 2)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        scores = (q @ k.transpose(-2, -1)).float() / math.sqrt(self.hd)  # (B,H,T,T)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        mask = causal[None, None]
        if key_mask is not None:  # (B,T) True = real token
            mask = mask & key_mask[:, None, None, :]
        if attn_mask is not None:  # (B,T,T) or per head (B,H,T,T); True = query i may read key j (and causality)
            mask = mask & (attn_mask if attn_mask.dim() == 4 else attn_mask[:, None, :, :])
        scores = scores.masked_fill(~mask, float("-inf"))
        if self.null is not None:
            null = self.null.float()[None, :, None, None].expand(B, self.h, T, 1)
            scores = torch.cat([scores, null], dim=-1)
        w = torch.softmax(scores, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)  # fully-masked rows (padding queries)
        w_tok = w[..., :T].to(v.dtype)
        y = self.drop(w_tok) @ v
        if self.capture_heads:
            self.head_out = y.detach()
        if self.head_replace:
            y = y.clone()
            for h, val in self.head_replace.items():
                y[:, h] = 0.0 if val is None else val.to(y.dtype)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.proj(y), (w if want_attn else None)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(), nn.Linear(cfg.d_mlp, cfg.d_model))

    def forward(self, x, pos, key_mask, want_attn, attn_mask=None):
        a, w = self.attn(self.ln1(x), pos, key_mask, want_attn, attn_mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, w


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab_size, d) if cfg.vocab_size > 0 else None
        self.feat = nn.Linear(cfg.n_features, d) if cfg.n_features > 0 else None
        assert self.tok is not None or self.feat is not None, "need tokens or features"
        self.pos_emb = nn.Embedding(cfg.max_len, d) if cfg.pos == "learned" else None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.n_out)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def embed(self, tokens=None, feats=None):
        x = 0
        if self.tok is not None and tokens is not None:
            x = x + self.tok(tokens)
        if self.feat is not None and feats is not None:
            x = x + self.feat(feats)
        return x

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        feats: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
        attn_mask=None,
        return_resid: bool = False,
        return_attn: bool = False,
        edit: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ) -> dict:
        x = self.embed(tokens, feats)
        B, T, _ = x.shape
        if pos is None:
            pos = torch.arange(T, device=x.device, dtype=torch.float32)[None].expand(B, T)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=x.device))[None]
        resid, attn = [], []
        if edit is not None:
            x = edit(-1, x)
        if return_resid:
            resid.append(x)
        for i, blk in enumerate(self.blocks):
            m = attn_mask.get(i) if isinstance(attn_mask, dict) else attn_mask
            x, w = blk(x, pos, key_mask, return_attn, m)
            if edit is not None:
                x = edit(i, x)
            if return_resid:
                resid.append(x)
            if return_attn:
                attn.append(w)
        logits = self.head(self.ln_f(x))
        out = {"logits": logits}
        if return_resid:
            out["resid"] = resid  # list of L+1 tensors (B,T,d): embeddings, then after each block
        if return_attn:
            out["attn"] = attn  # list of L tensors (B,H,T,T[+1 if null slot])
        return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
