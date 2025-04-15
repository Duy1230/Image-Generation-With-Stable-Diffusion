import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.proj1 = nn.Linear(n_embd, 4 * n_embd)
        self.proj2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        x = self.proj1(x)
        x = F.silu(x)
        x = self.proj2(x)
        return x



def embed_a_timestep(timestep, embedding_dim=320):
    half_dim = embedding_dim // 2
    freqs = torch.exp(-math.log(10000) *
                      torch.arange(start=0, end=half_dim, dtype=torch.float32) /
                      half_dim)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def embed_timesteps(timesteps, embedding_dim=320):
    half_dim = embedding_dim // 2
    freqs = torch.exp(-math.log(10000) *
                      torch.arange(half_dim, dtype=torch.float32) /
                      half_dim).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)