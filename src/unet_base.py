from typing import Any

import torch
import torch.nn as nn


def get_time_embedding(time_steps, temb_dim):
    """
    :param time_steps : 1D tensor of length batch size
    :param temb_dim : Dimension of embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2!"

    factor = 1e4 ** (
        torch.arange(start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2)
    )

    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([ torch.sin(t_emb), torch.cos(t_emb) ], dim=-1)

    return t_emb


class DownBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, t_emb_dim : int, down_sample = True, num_heads = 4, num_layers = 1 ,*args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.down_sample = down_sample

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )

        self.t_emb_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, out_channels) for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [ nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers) ]
        )

        self.residual_input_conv = nn.ModuleList(
            [ 
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1, stride=1, padding=1)
             for i in range(num_layers)
            ]
        )

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.double else nn.Identity()
    
    