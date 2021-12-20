import torch
from torch import nn


def feature_loss(features_r, features_g):
    loss = 0
    for r, g in zip(features_r, features_g):
        loss += nn.L1Loss()(r, g)

    return 2 * loss


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
