import torch
from torch import nn


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += nn.L1Loss()(rl, gl)

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        loss += nn.MSELoss()(dr, torch.ones(dg.size(), device='cuda')) + \
                nn.MSELoss()(dg, torch.zeros(dg.size(), device='cuda'))

    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        loss += nn.MSELoss()(dg, torch.ones(dg.size(), device='cuda'))

    return loss