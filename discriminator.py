import torch
from torch import nn
import torch.nn.functional as F


class PeriodicSubDiscriminator(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(PeriodicSubDiscriminator, self).__init__()
        self.period = period
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1)),
            nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1)),
            nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1)),
            nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1)),
            nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)),
        )
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        fmap = []

        if x.size(-1) % self.period:
            x = F.pad(x, (0, self.period - (x.size(-1) % self.period)), "reflect")
        b, t = x.size()
        x = x.view(b, t // self.period, self.period).unsqueeze(1)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x,)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MPD(torch.nn.Module):
    def __init__(self):
        super(MPD, self).__init__()

        self.sub_discriminators = nn.Sequential(
            PeriodicSubDiscriminator(2),
            PeriodicSubDiscriminator(3),
            PeriodicSubDiscriminator(5),
            PeriodicSubDiscriminator(7),
            PeriodicSubDiscriminator(11),
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.sub_discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ScaleSubDiscriminator(torch.nn.Module):
    def __init__(self, norm=nn.utils.weight_norm):
        super(ScaleSubDiscriminator, self).__init__()
        self.convs = nn.Sequential(
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        )
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        x = x.unsqueeze(1)
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MSD(torch.nn.Module):
    def __init__(self):
        super(MSD, self).__init__()
        self.discriminators = nn.ModuleList([
            ScaleSubDiscriminator(nn.utils.spectral_norm),
            ScaleSubDiscriminator(),
            ScaleSubDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
