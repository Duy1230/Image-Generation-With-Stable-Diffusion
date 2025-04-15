import torch
import torch.nn as nn
from .unet_parts import UNET_AttentionBlock, UNET_ResidualBlock, SwitchSequential, Upsample


class UNET(nn.Module):
    def __init__(self, h_dim, n_head):
        super().__init__()

        self.down_blocks = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, h_dim, kernel_size=3, padding=1)),
            SwitchSequential(
                UNET_ResidualBlock(h_dim, h_dim),
                UNET_AttentionBlock(n_head, (h_dim)//n_head)
            ),
            SwitchSequential(
                UNET_ResidualBlock(h_dim, h_dim),
                UNET_AttentionBlock(n_head, (h_dim)//n_head)
            ),
            SwitchSequential(
                nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(
                UNET_ResidualBlock(h_dim, 2*h_dim),
                UNET_AttentionBlock(n_head, (2*h_dim)//n_head)
            ),
            SwitchSequential(
                UNET_ResidualBlock(2*h_dim, 2*h_dim),
                UNET_AttentionBlock(n_head, (2*h_dim)//n_head)
            ),
            SwitchSequential(nn.Conv2d(2*h_dim, 2*h_dim,
                             kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(2*h_dim, 4*h_dim)),
            SwitchSequential(UNET_ResidualBlock(4*h_dim, 4*h_dim)),
        ])

        self.mid_block = SwitchSequential(
            UNET_ResidualBlock(4*h_dim, 4*h_dim),
            UNET_AttentionBlock(n_head, (4*h_dim)//n_head),
            UNET_ResidualBlock(4*h_dim, 4*h_dim),
        )

        self.up_blocks = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(4*h_dim + 4*h_dim, 4*h_dim)),
            SwitchSequential(UNET_ResidualBlock(4*h_dim + 4*h_dim, 4*h_dim)),
            SwitchSequential(
                UNET_ResidualBlock(4*h_dim + 2*h_dim, 4*h_dim),
                Upsample(4*h_dim)
            ),
            SwitchSequential(
                UNET_ResidualBlock(4*h_dim + 2*h_dim, 2*h_dim),
                UNET_AttentionBlock(n_head, (2*h_dim)//n_head)
            ),
            SwitchSequential(
                UNET_ResidualBlock(4*h_dim, 2*h_dim),
                UNET_AttentionBlock(n_head, (2*h_dim)//n_head)
            ),
            SwitchSequential(
                UNET_ResidualBlock(2*h_dim + h_dim, 2*h_dim),
                UNET_AttentionBlock(n_head, (2*h_dim)//n_head),
                Upsample(2*h_dim)
            ),
            SwitchSequential(
                UNET_ResidualBlock(2*h_dim + h_dim, h_dim),
                UNET_AttentionBlock(n_head, h_dim//n_head)
            ),
            SwitchSequential(
                UNET_ResidualBlock(2*h_dim, h_dim),
                UNET_AttentionBlock(n_head, h_dim//n_head)
            ),
            SwitchSequential(
                UNET_ResidualBlock(2*h_dim, h_dim),
                UNET_AttentionBlock(n_head, h_dim//n_head)
            ),
        ])

    def forward(self, latent_input, context_embedding, time_embedding):
        down_block_residuals = []
        current_feature_map = latent_input

        for block in self.down_blocks:
            current_feature_map = block(
                current_feature_map,
                context_embedding,
                time_embedding
            )
            down_block_residuals.append(current_feature_map)

        current_feature_map = self.mid_block(
            current_feature_map,
            context_embedding,
            time_embedding
        )

        for block in self.up_blocks:
            residual = down_block_residuals.pop()
            current_feature_map = torch.cat(
                (current_feature_map, residual), dim=1)
            current_feature_map = block(
                current_feature_map,
                context_embedding,
                time_embedding
            )

        return current_feature_map
