import torch
from torch import nn
import torch.nn.functional as F


class PeriodSubDiscriminator(torch.nn.Module):
    def __init__(self, period):
        super(PeriodSubDiscriminator, self).__init__()
        self.period = period
        self.convs = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1))),
            nn.utils.weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1))),
            nn.utils.weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1))),
            nn.utils.weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        )
        self.fc = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        features = []

        if x.size(-1) % self.period:
            x = F.pad(x, (0, self.period - (x.size(-1) % self.period)), "reflect")
        x = x.view(x.size(0), x.size(1) // self.period, self.period).unsqueeze(1)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x)
            features.append(x)
        x = self.fc(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)

        return x, features


class MPD(torch.nn.Module):
    def __init__(self):
        super(MPD, self).__init__()

        self.sub_discriminators = nn.Sequential(
            PeriodSubDiscriminator(2),
            PeriodSubDiscriminator(3),
            PeriodSubDiscriminator(5),
            PeriodSubDiscriminator(7),
            PeriodSubDiscriminator(11),
        )

    def forward(self, real, generated):
        reals = []
        gens = []
        real_features = []
        gen_features = []
        for i, d in enumerate(self.sub_discriminators):
            real_x, real_feature = d(real)
            gen_x, gen_feature = d(generated)
            reals.append(real_x)
            real_features.append(real_feature)
            gens.append(gen_x)
            gen_features.append(gen_feature)

        return reals, gens, real_features, gen_features


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
        features = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)

        return x, features


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

    def forward(self, real, generated):
        reals = []
        gens = []
        real_features = []
        gen_features = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                real = self.meanpools[i - 1](real)
                generated = self.meanpools[i - 1](generated)
            real_x, real_feature = d(real)
            gen_x, gen_feature = d(generated)
            reals.append(real_x)
            real_features.append(real_feature)
            gens.append(gen_x)
            gen_features.append(gen_feature)

        return reals, gens, real_features, gen_features
