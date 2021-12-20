from itertools import chain

import numpy as np
import PIL
import torch

from dataclasses import dataclass

import torchaudio

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aligner import GraphemeAligner
from dataset import LJSpeechDataset, LJSpeechCollator
from discriminator import MSD, MPD
from featurizer import MelSpectrogram, MelSpectrogramConfig
from generator import Generator
from loss import discriminator_loss, generator_loss
from utils import plot_spectrogram_to_buf, disable_grads, enable_grads
from writer import WanDBWriter


@dataclass
class GeneratorConfig:
    first_hidden_size: int = 256
    upsample_kernel_sizes = (16, 16, 8)
    upsample_rates = (8, 8, 4)
    resblock_kernel_sizes = (3, 5, 7)
    resblock_dilation_sizes = ((1, 2), (2, 6), (3, 12))


LAMBDA = 45
NUM_EPOCHS = 15
BATCH_SIZE = 4
VALIDATION_TRANSCRIPTS = [
    'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
    'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
    'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space',
]


aligner = GraphemeAligner().to('cuda')
dataloader = DataLoader(LJSpeechDataset('.'), batch_size=BATCH_SIZE, collate_fn=LJSpeechCollator())
featurizer = MelSpectrogram(MelSpectrogramConfig())
writer = WanDBWriter()

generator = Generator(GeneratorConfig).cuda()
msd = MSD().cuda()
mpd = MPD().cuda()


try:
    generator.load_state_dict(torch.load('generator_state'))
    msd.load_state_dict(torch.load('msd_state'))
    mpd.load_state_dict(torch.load('mpd_state'))
except:
    pass

loss_fn = nn.MSELoss()
optim_g = torch.optim.AdamW(params=generator.parameters(), lr=2e-4, betas=(.8, .99), eps=1e-9)
optim_d = torch.optim.AdamW(chain(mpd.parameters(), msd.parameters()), lr=2e-4, betas=(.8, .99), eps=1e-9)
scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, 2000, 1e-7)
scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, 2000, 1e-7)

tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

val_batch = (
    featurizer(torchaudio.load('audio_1.wav')[0]),
    featurizer(torchaudio.load('audio_2.wav')[0]),
    featurizer(torchaudio.load('audio_3.wav')[0]),
)

generator_loss_log = []
discriminator_loss_log = []

for e in range(NUM_EPOCHS):
    generator.train()
    msd.train()
    mpd.train()
    epoch_generator_loss_log = []
    epoch_discriminator_loss_log = []
    for i, batch in tqdm(enumerate(dataloader)):
        mels = batch.mels.cuda()
        waveforms = batch.waveform.cuda()
        waveform_preds = generator(mels).squeeze(1)
        waveform_preds = waveform_preds[:, :batch.waveform.size(-1)]
        melspec_preds = featurizer(waveform_preds.cpu()).cuda()

        optim_d.zero_grad()

        mpd_r, mpd_g, _, _ = mpd(waveforms, waveform_preds.detach())
        mpd_loss = discriminator_loss(mpd_r, mpd_g)

        msd_r, msd_g, _, _ = msd(waveforms, waveform_preds.detach())
        msd_loss = discriminator_loss(msd_r, msd_g)

        loss_disc = mpd_loss + msd_loss

        loss_disc.backward()
        optim_d.step()

        optim_g.zero_grad()

        mel_loss = LAMBDA * nn.L1Loss()(melspec_preds, mels)

        disable_grads(msd)
        disable_grads(mpd)

        mpd_r, mpd_g, mpd_feature_loss = mpd(waveforms, waveform_preds)
        msd_r, msd_g, msd_feature_loss = msd(waveforms, waveform_preds)
        gen_loss_mpd = generator_loss(mpd_g)
        gen_loss_msd = generator_loss(msd_g)
        gen_loss = gen_loss_msd + gen_loss_mpd + msd_feature_loss + mpd_feature_loss + mel_loss

        gen_loss.backward()
        optim_g.step()

        enable_grads(msd)
        enable_grads(mpd)

        epoch_generator_loss_log = np.append(epoch_generator_loss_log, gen_loss.item())
        epoch_discriminator_loss_log = np.append(epoch_discriminator_loss_log, loss_disc.item())

        if i % 10 == 9:
            writer.set_step(e * len(dataloader) + i)
            writer.add_scalar('learning rate', scheduler_g.get_last_lr()[0])
            writer.add_scalar('Train generator loss', epoch_generator_loss_log[i - 9:].mean())
            writer.add_scalar('Train discriminator loss', epoch_discriminator_loss_log[i - 9:].mean())

        scheduler_g.step()
        scheduler_d.step()

    with torch.no_grad():
        generator.eval()
        generated_waves = (generator(sample.cuda()).cpu().squeeze(1) for sample in val_batch)

        for audio, t in zip(generated_waves, VALIDATION_TRANSCRIPTS):
            image = PIL.Image.open(plot_spectrogram_to_buf(audio))
            writer.add_audio("Generated audio for '{}'".format(t), audio, MelSpectrogramConfig.sr)

    torch.save(generator.state_dict(), 'generator_state')
    torch.save(msd.state_dict(), 'msd_state')
    torch.save(mpd.state_dict(), 'mpd_state')
