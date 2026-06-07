"""
Tiny GPT implementation for Part 4 — LLM training proxy.

Implemented from scratch (no HuggingFace model classes, no TransformerDecoderLayer).
Architecture:
  Token Embedding + Positional Embedding
  → N × GPT Block (LayerNorm → CausalSelfAttention → LayerNorm → MLP)
  → LayerNorm → Linear(d_model, vocab_size)

~3M parameters with default config.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)  # each (B, T, C)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Causal scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)

        # Causal mask — upper triangular is -inf
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        att = att.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        att = F.softmax(att, dim=-1)

        if self.dropout > 0 and self.training:
            att = F.dropout(att, p=self.dropout)

        out = att @ v  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class GPTBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture (GPT-2 style)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    Manually implemented GPT for Part 4.

    No dropout by default (config.GPT_DROPOUT = 0.0) — clean loss landscape.
    """

    def __init__(self,
                 vocab_size: int = None,
                 context_length: int = None,
                 d_model: int = None,
                 n_layers: int = None,
                 n_heads: int = None,
                 d_ff: int = None,
                 dropout: float = None):
        super().__init__()
        vocab_size = vocab_size or config.GPT_VOCAB_SIZE
        context_length = context_length or config.GPT_CONTEXT_LENGTH
        d_model = d_model or config.GPT_D_MODEL
        n_layers = n_layers or config.GPT_N_LAYERS
        n_heads = n_heads or config.GPT_N_HEADS
        d_ff = d_ff or config.GPT_D_FF
        dropout = dropout if dropout is not None else config.GPT_DROPOUT

        self.context_length = context_length
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (token embedding ↔ output projection)
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx : (B, T) long tensor of token ids
        Returns logits (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.context_length, \
            f'Sequence length {T} exceeds context_length {self.context_length}'

        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)              # (B, T, d_model)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

    def n_params(self) -> int:
        # Don't double-count tied weights
        return sum(p.numel() for p in self.parameters())

    def compute_loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss for language modeling."""
        logits = self(idx)  # (B, T, V)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, V),
            targets.view(B * T),
            ignore_index=-1,
        )
        return loss

    @torch.no_grad()
    def compute_perplexity(self, loader, device: torch.device) -> float:
        """Compute perplexity over a DataLoader."""
        self.eval()
        total_loss = 0.0
        total_tokens = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = self(x_batch)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                y_batch.view(B * T),
            )
            total_loss += loss.item() * B * T
            total_tokens += B * T
        self.train()
        avg_loss = total_loss / max(total_tokens, 1)
        return math.exp(min(avg_loss, 100.0))
