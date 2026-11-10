"""Shared-head, pre-normalized visual query decoder."""
import torch
from torch import nn


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, ffn, dropout):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory):
        q = self.norms[0](x)
        x = x + self.drop(self.self_attn(q, q, q, need_weights=False)[0])
        value, weights = self.cross_attn(self.norms[1](x), memory, memory, need_weights=True)
        x = x + self.drop(value)
        return x + self.drop(self.ffn(self.norms[2](x))), weights


class QueryDecoder(nn.Module):
    def __init__(self, token_dim, node_text, dim=256, num_layers=4, num_heads=4, ffn_dim=1024, dropout=0.1, init_std=0.02):
        super().__init__()
        self.register_buffer('node_text', node_text.detach().float().clone())
        self.token_proj = nn.Linear(token_dim, dim)
        self.query_proj = nn.Linear(node_text.shape[1], dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([DecoderLayer(dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.out_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight, std=init_std)
                nn.init.zeros_(module.in_proj_bias)

    def forward(self, visual_tokens):
        memory = self.memory_norm(self.token_proj(visual_tokens))
        query = self.query_norm(self.query_proj(self.node_text)).unsqueeze(0).expand(memory.shape[0], -1, -1)
        for layer in self.layers:
            query, weights = layer(query, memory)
        return self.head(self.out_dropout(self.final_norm(query))).squeeze(-1), weights