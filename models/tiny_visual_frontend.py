#!/usr/bin/env python3
import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1):
        p = k // 2
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )


class DSBlock(nn.Module):
    """Depthwise-separable residual block."""
    def __init__(self, cin, cout, stride=1, expand=2):
        super().__init__()
        hidden = max(cin, int(cin * expand))
        self.block = nn.Sequential(
            ConvBNAct(cin, hidden, 1, 1),
            ConvBNAct(hidden, hidden, 3, stride, groups=hidden),
            nn.Conv2d(hidden, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.skip = (
            nn.Identity() if cin == cout and stride == 1 else
            nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.block(x) + self.skip(x))


class TinyVisualFrontend(nn.Module):
    """
    Drop-in visual frontend for SEANet.

    Input : [T, B, 1, 112, 112]
    Output: [T, B, embedding_dim]

    temporal_kernel=5:
      same temporal receptive field as the original frontend3D kernel.
    temporal_kernel=1:
      strictly single-frame visual encoder (future experiment).

    Only the first Conv3d mixes time. Everything after it is frame-wise.
    """
    def __init__(
        self,
        embedding_dim=512,
        temporal_kernel=5,
        stem_channels=16,
        widths=(24, 32, 64, 96),
        expand=2,
    ):
        super().__init__()
        if temporal_kernel % 2 != 1:
            raise ValueError("temporal_kernel must be odd")

        self.embedding_dim = embedding_dim
        self.temporal_kernel = temporal_kernel

        self.frontend3d = nn.Sequential(
            nn.Conv3d(
                1, stem_channels,
                kernel_size=(temporal_kernel, 5, 5),
                stride=(1, 2, 2),
                padding=(temporal_kernel // 2, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(stem_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )

        blocks = []
        cin = stem_channels
        for cout in widths:
            blocks.append(DSBlock(cin, cout, stride=2, expand=expand))
            blocks.append(DSBlock(cout, cout, stride=1, expand=expand))
            cin = cout

        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(cin, embedding_dim)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [T,B,C,H,W], got {tuple(x.shape)}")

        # [T,B,C,H,W] -> [B,C,T,H,W]
        x = x.permute(1, 2, 0, 3, 4).contiguous()
        x = self.frontend3d(x)                    # [B,C,T,h,w]

        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)                          # [B*T,D]
        x = x.reshape(B, T, self.embedding_dim)
        return x.permute(1, 0, 2).contiguous()   # [T,B,D]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for k, d in [(5, 512), (1, 512), (1, 128)]:
        m = TinyVisualFrontend(embedding_dim=d, temporal_kernel=k)
        x = torch.randn(25, 2, 1, 112, 112)
        y = m(x)
        print(f"k={k} D={d}: {tuple(y.shape)}, params={count_parameters(m):,}")
