import torch
import torch.nn as nn
import torch.nn.functional as F
from .attenttion import SelfAttention, CrossAttention


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=1280):
        super().__init__()
        self.gn_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.time_embedding_proj = nn.Linear(time_dim, out_channels)

        self.gn_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_connection = nn.Identity()
        else:
            self.residual_connection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, input_feature, time_emb):
        residual = input_feature

        h = self.gn_feature(input_feature)
        h = F.silu(h)
        h = self.conv_feature(h)

        time_emb_processed = F.silu(time_emb)
        time_emb_projected = self.time_embedding_proj(time_emb_processed)
        time_emb_projected = time_emb_projected.unsqueeze(-1).unsqueeze(-1)

        merged_feature = h + time_emb_projected
        merged_feature = self.gn_merged(merged_feature)
        merged_feature = F.silu(merged_feature)
        merged_feature = self.conv_merged(merged_feature)

        output = merged_feature + self.residual_connection(residual)
        return output


class UNET_AttentionBlock(nn.Module):
    def __init__(self, num_heads, head_dim, context_dim=512):
        super().__init__()
        embed_dim = num_heads * head_dim

        self.gn_in = nn.GroupNorm(32, embed_dim, eps=1e-6)
        self.proj_in = nn.Conv2d(embed_dim, embed_dim,
                                 kernel_size=1, padding=0)

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn_1 = SelfAttention(num_heads, embed_dim, in_proj_bias=False)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn_2 = CrossAttention(
            num_heads, embed_dim, context_dim, in_proj_bias=False)
        self.ln_3 = nn.LayerNorm(embed_dim)

        self.ffn_geglu = nn.Linear(embed_dim, 4 * embed_dim * 2)
        self.ffn_out = nn.Linear(4 * embed_dim, embed_dim)
        self.proj_out = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=1, padding=0)

    def forward(self, input_tensor, context_tensor):
        skip_connection = input_tensor

        B, C, H, W = input_tensor.shape
        HW = H * W

        h = self.gn_in(input_tensor)
        h = self.proj_in(h)
        h = h.view(B, C, HW).transpose(-1, -2)

        attn1_skip = h
        h = self.ln_1(h)
        h = self.attn_1(h)
        h = h + attn1_skip

        attn2_skip = h
        h = self.ln_2(h)
        h = self.attn_2(h, context_tensor)
        h = h + attn2_skip

        ffn_skip = h
        h = self.ln_3(h)
        intermediate, gate = self.ffn_geglu(h).chunk(2, dim=-1)
        h = intermediate * F.gelu(gate)
        h = self.ffn_out(h)
        h = h + ffn_skip

        h = h.transpose(-1, -2).view(B, C, H, W)
        output = self.proj_out(h) + skip_connection
        return output


class Upsample(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels,
                              kernel_size=3, padding=1)

    def forward(self, feature_map):
        x = F.interpolate(feature_map, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class SwitchSequential(nn.Sequential):
    def forward(self, x, guidance_context, time_embedding):
        for module_instance in self:
            if isinstance(module_instance, UNET_AttentionBlock):
                x = module_instance(x, guidance_context)
            elif isinstance(module_instance, UNET_ResidualBlock):
                x = module_instance(x, time_embedding)
            else:
                x = module_instance(x)
        return x


class UNETOutputLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.final_groupnorm = nn.GroupNorm(32, input_channels)
        self.final_conv = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, feature_map):
        norm_map = self.final_groupnorm(feature_map)
        activated_map = F.silu(norm_map)
        output_map = self.final_conv(activated_map)

        return output_map
