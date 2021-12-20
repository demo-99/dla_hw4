import torch
from torch import nn


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
