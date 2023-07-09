from dance_diffusion.base.model import ModelWrapperBase

import torch
import numpy as np

from torch import nn
from sample_diffusion.dance_diffusion.base.type import ModelType
from typing import Callable

from sample_diffusion.dance_diffusion.base.t5 import T5Embedder
from sample_diffusion.dance_diffusion.base.adp_modules import UNetCFG1d
from archisound import ArchiSound


class CVXLatentAudioDiffusion(nn.Module):
    def __init__(
        self,
        autoencoder: ArchiSound,
        aec_latent_dim: int,
        aec_downsampling_ratio: int,
        aec_divisor: float,
        **model_kwargs,
    ):
        super().__init__()

        self.latent_dim = aec_latent_dim
        self.downsampling_ratio = aec_downsampling_ratio
        self.aec_divisor = aec_divisor

        embedding_max_len = 64

        self.embedder = T5Embedder(
            model="t5-small", max_length=embedding_max_len
        ).requires_grad_(False)

        self.embedding_features = 512

        self.diffusion = UNetCFG1d(
            in_channels=self.latent_dim,
            context_embedding_features=self.embedding_features,
            context_embedding_max_length=embedding_max_len + 2,  # 2 for timestep embeds
            channels=256,
            resnet_groups=8,
            kernel_multiplier_downsample=2,
            multipliers=[2, 3, 3, 4, 4],
            factors=[1, 2, 4, 4],
            num_blocks=[3, 3, 3, 3],
            attentions=[0, 0, 3, 3, 3],
            attention_heads=12,
            attention_features=64,
            attention_multiplier=4,
            attention_use_rel_pos=True,
            attention_rel_pos_max_distance=2048,
            attention_rel_pos_num_buckets=64,
            use_nearest_upsample=False,
            use_skip_scale=True,
            use_context_time=True,
        )

        self.autoencoder = autoencoder


class CVXLDDModelWrapper(ModelWrapperBase):
    def __init__(self):
        super().__init__()

        self.module: CVXLatentAudioDiffusion = None
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
                name="Conditional Very Extra Latent Dance Diffusion Model",
                description="v1.0",
                type=ModelType.CVXLDD,
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

        self.module = CVXLatentAudioDiffusion(
            autoencoder, 32, 512, 2.5, **latent_diffusion_config
        )
        self.module.load_state_dict(file["state_dict"], strict=False)  # ?
        self.module.eval().requires_grad_(False)

        self.latent_dim = self.module.autoencoder.latent_dim
        self.downsampling_ratio = self.module.autoencoder.downsampling_ratio

        self.ae_encoder = self.module.autoencoder.encode

        self.ae_decoder = self.module.autoencoder.decode

        self.t5_embedder = self.module.embedder

        self.diffusion = (
            self.module.diffusion
            if (optimize_memory_use)
            else self.module.diffusion.to(device_accelerator)
        )
