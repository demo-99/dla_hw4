from itertools import chain

import PIL
import numpy as np
import torch

from dataclasses import dataclass

import torchaudio
from torchvision.transforms import ToTensor

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aligner import GraphemeAligner
from dataset import LJSpeechDataset, LJSpeechCollator
from discriminator import MSD, MPD
from featurizer import MelSpectrogram, MelSpectrogramConfig
from generator import Generator
from loss import discriminator_loss, feature_loss, generator_loss
from utils import plot_spectrogram_to_buf, disable_grads, enable_grads
from vocoder import Vocoder
from writer import WanDBWriter


@dataclass
class GeneratorConfig:
    first_hidden_size: int = 512
    upsample_kernel_sizes = (16,16,4,4)
    upsample_rates = (8,8,2,2)
    resblock_kernel_sizes = (3,7,11)
    resblock_dilation_sizes = ((1,3,5), (1,3,5), (1,3,5))


NUM_EPOCHS = 15
BATCH_SIZE = 2
VALIDATION_TRANSCRIPTS = [
    'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
    'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
    'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space',
]


aligner = GraphemeAligner().to('cuda')
dataloader = DataLoader(LJSpeechDataset('.'), batch_size=BATCH_SIZE, collate_fn=LJSpeechCollator())
featurizer = MelSpectrogram(MelSpectrogramConfig())
vocoder = Vocoder().to('cuda:0').eval()
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

val_batch = tokenizer(VALIDATION_TRANSCRIPTS)[0].to('cuda')

generator_loss_log = []
discriminator_loss_log = []

generator.train()
msd.train()
mpd.train()

for e in range(NUM_EPOCHS):
    epoch_generator_loss_log = []
    epoch_discriminator_loss_log = []
    for i, batch in tqdm(enumerate(dataloader)):
        mels = batch.mels.cuda()
        waveform_preds = generator(mels).squeeze(1)
        waveform_preds = waveform_preds[:, :batch.waveform.size(-1)]
        melspec_preds = featurizer(waveform_preds.cpu()).cuda()

        optim_d.zero_grad()

        y_df_hat_r, y_df_hat_g, _, _ = mpd(batch.waveform.cuda(), waveform_preds.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        y_ds_hat_r, y_ds_hat_g, _, _ = msd(batch.waveform.cuda(), waveform_preds.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc = loss_disc_s + loss_disc_f

        loss_disc.backward()
        optim_d.step()

        optim_g.zero_grad()

        disable_grads(msd)
        disable_grads(mpd)

        min_sz = min(melspec_preds.size(-1), mels.size(-1))

        mel_loss = 45 * nn.L1Loss()(melspec_preds[:, :, :min_sz], mels[:, :, :min_sz])

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(batch.waveform.cuda(), waveform_preds)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(batch.waveform.cuda(), waveform_preds)
        fm_loss_f = feature_loss(fmap_f_r, fmap_f_g)
        fm_loss_s = feature_loss(fmap_s_r, fmap_s_g)
        gen_loss_f, gen_losses_f = generator_loss(y_df_hat_g)
        gen_loss_s, gen_losses_s = generator_loss(y_ds_hat_g)
        gen_loss = gen_loss_s + gen_loss_f + fm_loss_s + fm_loss_f + mel_loss

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
        generated_waves = vocoder.inference(generator(val_batch, None)[0]).cpu()

        for audio, t in zip(generated_waves, VALIDATION_TRANSCRIPTS):
            image = PIL.Image.open(plot_spectrogram_to_buf(audio))
            writer.add_image("Waveform for '{}'".format(t), ToTensor()(image))
            writer.add_audio("Audio for '{}'".format(t), audio, MelSpectrogramConfig.sr)

    torch.save(generator.state_dict(), 'generator_state')
    torch.save(msd.state_dict(), 'msd_state')
    torch.save(mpd.state_dict(), 'mpd_state')
