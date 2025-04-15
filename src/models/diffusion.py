import torch
import torch.nn as nn
from utils.time_embedding import TimeEmbedding
from .unet import UNET
from .unet_parts import UNETOutputLayer


class Diffusion(nn.Module):
    def __init__(self, h_dim=128, n_head=4):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET(h_dim, n_head)
        self.unet_output = UNETOutputLayer(h_dim, 4)

    @torch.autocast(
        device_type='cuda', dtype=torch.float16,
        enabled=True, cache_enabled=True
    )
    def forward(self, latent, context, time):
        time = self.time_embedding(time)
        output = self.unet(latent, context, time)
        output = self.unet_output(output)
        return output
