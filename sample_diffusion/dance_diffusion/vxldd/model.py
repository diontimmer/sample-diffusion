import torch
from torch import nn
from typing import Callable

from archisound import ArchiSound
from .latent_diffusion_model import VXLatentAudioDiffusion
from sample_diffusion.dance_diffusion.base.model import ModelWrapperBase
from sample_diffusion.dance_diffusion.base.type import ModelType


class VXLDDModelWrapper(ModelWrapperBase):
    def __init__(self):
        super().__init__()

        self.module: VXLatentAudioDiffusion = None
        self.model: Callable = None

    def load(
        self,
        path: str,
        device_accelerator: torch.device,
        optimize_memory_use: bool = False,
        chunk_size: int = None,
        sample_rate: int = None,
    ):
        default_model_config = dict(
            version=[0, 0, 1],
            model_info=dict(
                name="Very Extra Latent Dance Diffusion Model",
                description="v1.0",
                type=ModelType.VXLDD,
                native_chunk_size=524288,
                sample_rate=44100,
            ),
            latent_diffusion_config=dict(
                io_channels=32,
                n_attn_layers=4,
                channels=[512] * 6 + [1024] * 4,
                depth=10,
            ),
            autoencoder_config=dict(
                channels=64,
                c_mults=[2, 4, 8, 16, 32],
                strides=[2, 2, 2, 2, 2],
                latent_dim=32,
            ),
        )

        file = torch.load(path, map_location="cpu")

        model_config = file.get("model_config")
        if not model_config:
            print(f"Model file {path} is invalid. Please run the conversion script.")
            print(f" - Default model config will be used, which may be inaccurate.")
            model_config = default_model_config

        model_info = model_config.get("model_info")

        self.path = path
        self.native_chunk_size = (
            model_info.get("native_chunk_size") if not chunk_size else chunk_size
        )
        self.sample_rate = (
            model_info.get("sample_rate") if not sample_rate else sample_rate
        )

        # autoencoder_config = model_config.get('autoencoder_config')
        latent_diffusion_config = model_config.get("latent_diffusion_config")

        # autoencoder = AudioAutoencoder(**autoencoder_config).requires_grad_(False)
        autoencoder = ArchiSound.from_pretrained("dmae1d-ATC32-v3")

        autoencoder = autoencoder.to(device_accelerator)

        self.module = VXLatentAudioDiffusion(autoencoder, **latent_diffusion_config)

        # for item in self.module.state_dict():
        #    print(item)

        self.module.load_state_dict(file["state_dict"], strict=False)  # ?
        # self.module.autoencoder.load_state_dict( #?
        #     file["state_dict"],
        #     strict=False
        # )

        self.module.eval().requires_grad_(False)

        self.latent_dim = 32  # self.module.autoencoder.latent_dim
        self.downsampling_ratio = 512  # self.module.autoencoder.downsampling_ratio

        self.ae_encoder = self.module.autoencoder.encode

        self.ae_decoder = self.module.autoencoder.decode

        self.diffusion = (
            self.module.diffusion
            if (optimize_memory_use)
            else self.module.diffusion.to(device_accelerator)
        )
