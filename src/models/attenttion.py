import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, num_attn_heads, hidden_dim, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.num_heads = num_attn_heads
        self.head_size = hidden_dim // num_attn_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=in_proj_bias)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=out_proj_bias)

    def forward(self, features, use_causal_mask=False):
        b, s, d = features.shape

        qkv_combined = self.qkv_proj(features)
        q_mat, k_mat, v_mat = torch.chunk(qkv_combined, 3, dim=-1)

        q_mat = q_mat.view(b, s, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k_mat = k_mat.view(b, s, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        v_mat = v_mat.view(b, s, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        qk = torch.matmul(q_mat, k_mat.transpose(-2, -1))
        sqrt_qk = qk / math.sqrt(self.head_size)

        if use_causal_mask:
            causal_mask = torch.triu(torch.ones_like(sqrt_qk, dtype=torch.bool), diagonal=1)
            sqrt_qk = sqrt_qk.masked_fill(causal_mask, -torch.inf)

        attn_weights = torch.softmax(sqrt_qk, dim=-1)
        attn_values = torch.matmul(attn_weights, v_mat)

        attn_values = attn_values.permute(0, 2, 1, 3).contiguous()
        attn_values = attn_values.view(b, s, d)

        final_output = self.output_proj(attn_values)
        return final_output


class CrossAttention(nn.Module):
    def __init__(self, num_attn_heads, query_dim, context_dim, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.num_heads = num_attn_heads
        self.head_size = query_dim // num_attn_heads

        self.query_map = nn.Linear(query_dim, query_dim, bias=in_proj_bias)
        self.key_map = nn.Linear(context_dim, query_dim, bias=in_proj_bias)
        self.value_map = nn.Linear(context_dim, query_dim, bias=in_proj_bias)

        self.output_map = nn.Linear(query_dim, query_dim, bias=out_proj_bias)

    def forward(self, query_input, context_input):
        b_q, s_q, d_q = query_input.shape
        _, s_kv, _ = context_input.shape

        q_mat = self.query_map(query_input)
        k_mat = self.key_map(context_input)
        v_mat = self.value_map(context_input)

        q_mat = q_mat.view(b_q, s_q, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k_mat = k_mat.view(b_q, s_kv, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        v_mat = v_mat.view(b_q, s_kv, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        qk = torch.matmul(q_mat, k_mat.transpose(-2, -1))
        sqrt_qk = qk / math.sqrt(self.head_size)
        attn_weights = torch.softmax(sqrt_qk, dim=-1)

        attn_values = torch.matmul(attn_weights, v_mat)
        attn_values = attn_values.permute(0, 2, 1, 3).contiguous()
        attn_values = attn_values.view(b_q, s_q, d_q)

        final_output = self.output_map(attn_values)
        return final_output