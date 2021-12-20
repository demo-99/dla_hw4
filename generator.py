import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, dilation):
        super(ConvBlock, self).__init__()

        self.net = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size,
                stride=1,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size,
                stride=1,
                dilation=1,
                padding=(kernel_size - 1) // 2
            ),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()

        self.net = nn.ModuleList(
            [ConvBlock(hidden_size, kernel_size, d) for d in dilation]
        )

    def forward(self, x):
        for cb in self.net:
            x = cb(x) + x
        return x


class MRF(nn.Module):
    def __init__(self, hidden_size, kernel_sizes, dilations):
        super(MRF, self).__init__()

        self.resblocks = nn.ModuleList(
            [
                ResBlock(hidden_size, k, d)
                for k, d in zip(kernel_sizes, dilations)
            ]
        )

    def forward(self, x):
        xs = None
        for resblock in self.resblocks:
            if xs is None:
                xs = resblock(x)
            else:
                xs += resblock(x)
        return xs / len(self.resblocks)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.num_upsamples = len(config.upsample_rates)

        self.conv_pre = nn.Conv1d(80, config.first_hidden_size, 7, 1, padding=3)
        self.convs = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    config.first_hidden_size // 2 ** i,
                    config.first_hidden_size // 2 ** (i + 1),
                    kernel_size=upblock_kernel_size,
                    stride=upblock_stride,
                    padding=(upblock_kernel_size - upblock_stride) // 2,
                )
                for i, (upblock_kernel_size, upblock_stride) in enumerate(zip(
                config.upsample_kernel_sizes, config.upsample_rates
            ))
            ]
        )

        self.MRFs = nn.ModuleList(
            [
                MRF(
                    config.first_hidden_size // 2 ** (i + 1),
                    config.resblock_kernel_sizes,
                    config.resblock_dilation_sizes,
                )
                for i in range(self.num_upsamples)
            ]
        )

        self.fc = nn.Conv1d(
            config.first_hidden_size // 2 ** self.num_upsamples,
            1, kernel_size=7, stride=1, padding=3
        )

    def forward(self, x):
        x = self.conv_pre(x)
        for conv, mrf in zip(self.convs, self.MRFs):
            x = F.leaky_relu(x)
            x = conv(x)
            x = mrf(x)
        x = F.leaky_relu(x)
        x = self.fc(x)
        x = torch.tanh(x)

        return x
