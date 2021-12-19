import io

import matplotlib.pyplot as plt


def plot_spectrogram_to_buf(reconstructed_wav, name=None):
    plt.figure(figsize=(20, 5))
    plt.plot(reconstructed_wav, alpha=.5)
    plt.title(name)
    plt.grid()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def disable_grads(discriminator):
    for p in discriminator.parameters():
        p.requires_grad = False


def enable_grads(discriminator):
    for p in discriminator.parameters():
        p.requires_grad = True
