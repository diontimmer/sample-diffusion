from sample_diffusion.dance_diffusion.base.model import LatentModelWrapperBase

import torch
import numpy as np

from torch import nn
from sample_diffusion.dance_diffusion.base.type import ModelType
from typing import Callable

from sample_diffusion.dance_diffusion.base.adp_modules import (
    UNetCFG1d,
    T5Embedder,
    NumberEmbedder,
)

class TSCondLatentAudioDiffusion(nn.Module):
    def __init__(
        self,
        autoencoder,
        **model_kwargs,
    ):
        super().__init__()
        
        self.autoencoder = autoencoder
        self.latent_dim = self.autoencoder.latent_dim
        self.downsampling_ratio = self.autoencoder.downsampling_ratio

        self.embedding_features = 768
        self.max_seconds = 512

        embedding_max_len = 64

        self.embedder = T5Embedder(
            model="t5-base", max_length=embedding_max_len
        ).requires_grad_(False)

        self.second_start_embedder = nn.Embedding(
            self.max_seconds + 1, self.embedding_features
        )
        self.second_total_embedder = nn.Embedding(
            self.max_seconds + 1, self.embedding_features
        )
        self.timestamp_start_embedder = NumberEmbedder(features=self.embedding_features)

        self.diffusion = UNetCFG1d(
            in_channels=self.latent_dim,
            context_embedding_features=self.embedding_features,
            context_embedding_max_length=embedding_max_len + 3,  # 3 for timing embeds
            channels=256,
            resnet_groups=8,
            kernel_multiplier_downsample=2,
            multipliers=[2, 3, 3, 4, 4],
            factors=[1, 2, 4, 4],
            num_blocks=[3, 3, 3, 3],
            attentions=[0, 0, 3, 3, 3],
            attention_heads=16,
            attention_features=64,
            attention_multiplier=4,
            attention_use_rel_pos=True,
            attention_rel_pos_max_distance=2048,
            attention_rel_pos_num_buckets=256,
            use_nearest_upsample=False,
            use_skip_scale=True,
            use_context_time=True,
        )

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def get_timing_embeddings(self, seconds_starts_totals):
        seconds_starts_totals = torch.tensor(seconds_starts_totals).to("cpu")
        seconds_starts_totals = seconds_starts_totals.clamp(0, self.max_seconds)
        seconds_starts_totals = seconds_starts_totals.transpose(0, 1)

        second_starts = seconds_starts_totals[0]
        second_totals = seconds_starts_totals[1]

        second_starts_embeds = self.second_start_embedder(second_starts).unsqueeze(1)
        second_totals_embeds = self.second_total_embedder(second_totals).unsqueeze(1)

        # Divide second_starts by second_totals to get t_starts, make sure it's cast to a float
        t_starts = second_starts / second_totals.float()
        t_starts_embeds = self.timestamp_start_embedder(t_starts).unsqueeze(1)

        return second_starts_embeds, second_totals_embeds, t_starts_embeds


class TCVXLDDModelWrapper(LatentModelWrapperBase):
    def __init__(self):
        super().__init__()

        self.module: TSCondLatentAudioDiffusion = None
        self.model: Callable = None

    def load(
        self,
        path: str,
        device_accelerator: torch.device,
        optimize_memory_use: bool = False,
        chunk_size: int = None,
        sample_rate: int = None,
        aec_path: str = None,
        aec_config: dict = None,
    ):
        default_model_config = dict(
            version=[0, 0, 1],
            model_info=dict(
                name="TimeStamped Conditional Very Extra Latent Dance Diffusion Model",
                description="v1.0",
                type=ModelType.TCVXLDD,
                native_chunk_size=524288,
                sample_rate=44100,
            ),
            latent_diffusion_config=dict(
                io_channels=32,
                n_attn_layers=9,
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

        autoencoder = self.load_autoencoder(aec_path, aec_config)

        autoencoder = autoencoder.to(device_accelerator)

        self.module = TSCondLatentAudioDiffusion(
            autoencoder, **latent_diffusion_config
        )
        self.module.load_state_dict(file["state_dict"], strict=False)  # ?
        self.module.eval().requires_grad_(False)

        self.latent_dim = self.module.latent_dim
        self.downsampling_ratio = self.module.downsampling_ratio

        self.ae_encoder = self.module.autoencoder.encode

        self.ae_decoder = self.module.autoencoder.decode

        self.t5_embedder = self.module.embedder

        self.second_start_embedder = self.module.second_start_embedder
        self.second_total_embedder = self.module.second_total_embedder
        self.timestamp_start_embedder = self.module.timestamp_start_embedder

        self.diffusion = (
            self.module.diffusion
            if (optimize_memory_use)
            else self.module.diffusion.to(device_accelerator)
        )
