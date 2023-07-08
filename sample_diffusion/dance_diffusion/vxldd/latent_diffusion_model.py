from torch import nn

from sample_diffusion.dance_diffusion.base.latent_unet import DiffusionUnet1D
from archisound import ArchiSound


class VXLatentAudioDiffusion(nn.Module):
    def __init__(self, autoencoder: ArchiSound, **model_kwargs):
        super().__init__()

        default_model_kwargs = {
            "io_channels": 32,
            "n_attn_layers": 4,
            "channels": [512] * 6 + [1024] * 4,
            "depth": 10,
        }

        self.latent_dim = 32
        self.downsampling_ratio = 512

        self.diffusion = DiffusionUnet1D(**{**default_model_kwargs, **model_kwargs})

        self.autoencoder = autoencoder

    def forward(self, x, t, **extra_args):
        return self.diffusion(x, t, **extra_args)
