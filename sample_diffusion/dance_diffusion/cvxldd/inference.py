import torch

from diffusion.utils import t_to_alpha_sigma
from k_diffusion.external import VDenoiser

from typing import Callable
from sample_diffusion.diffusion_library.scheduler import SchedulerType
from sample_diffusion.diffusion_library.sampler import SamplerType
from sample_diffusion.dance_diffusion.base.model import ModelWrapperBase
from sample_diffusion.dance_diffusion.base.inference import InferenceBase


class CVXLDDInference(InferenceBase):
    def __init__(
        self,
        device_accelerator: torch.device = None,
        device_offload: torch.device = None,
        optimize_memory_use: bool = False,
        use_autocast: bool = True,
        model: ModelWrapperBase = None,
    ):
        super().__init__(
            device_accelerator, device_offload, optimize_memory_use, use_autocast, model
        )

    def generate(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args: dict = None,
        sampler: SamplerType = None,
        sampler_args: dict = None,
        **kwargs
    ):
        self.generator.manual_seed(seed)

        with self.offload_context(self.model.t5_embedder):
            embedding = self.model.t5_embedder(sampler_args["text_condition"]).to(
                self.device_accelerator
            )
            negative_embedding = (
                None
                if (sampler_args["inverse_text_condition"] == None)
                else self.model.t5_embedder(sampler_args["inverse_text_condition"]).to(
                    self.device_accelerator
                )
            )

        step_list = scheduler.get_step_list(
            steps, self.device_accelerator.type, **scheduler_args
        )  # step_list = step_list[:-1] if sampler in [SamplerType.V_PRK, SamplerType.V_PLMS, SamplerType.V_PIE, SamplerType.V_PLMS2, SamplerType.V_IPLMS] else step_list

        if SamplerType.is_v_sampler(sampler):
            x_T = torch.randn(
                [
                    batch_size,
                    self.model.latent_dim,
                    self.model.native_chunk_size // self.model.downsampling_ratio,
                ],
                generator=self.generator,
                device=self.device_accelerator,
            )
            model = self.model.diffusion
        else:
            x_T = step_list[0] * torch.randn(
                [
                    batch_size,
                    self.model.latent_dim,
                    self.model.native_chunk_size // self.model.downsampling_ratio,
                ],
                generator=self.generator,
                device=self.device_accelerator,
            )
            model = VDenoiser(self.model.diffusion)

        sampler_args["extra_args"] = {
            "embedding": embedding,
            "negative_embedding": negative_embedding,
            "embedding_scale": sampler_args["cfg_scale"],
        }

        with self.offload_context(self.model.diffusion):
            x_0 = sampler.sample(model, x_T, step_list, callback, **sampler_args)

        with self.offload_context(self.model.ae_decoder):
            x_0 = x_0 * self.model.module.aec_divisor
            return self.model.ae_decoder(x_0, num_steps=20)

    def generate_variation(
        self,
        callback: Callable = None,
        batch_size: int = None,
        seed: int = None,
        audio_source: torch.Tensor = None,
        expansion_map: list[int] = None,
        noise_level: float = None,
        steps: int = None,
        scheduler: SchedulerType = None,
        scheduler_args=None,
        sampler: SamplerType = None,
        sampler_args=None,
        **kwargs
    ) -> torch.Tensor:
        self.generator.manual_seed(seed)
        with self.offload_context(self.model.t5_embedder):
            embedding = self.model.t5_embedder(sampler_args["text_condition"]).to(
                self.device_accelerator
            )
            negative_embedding = (
                None
                if (sampler_args["inverse_text_condition"] == None)
                else self.model.t5_embedder(sampler_args["inverse_text_condition"]).to(
                    self.device_accelerator
                )
            )

        audio_source = self.expand(audio_source, expansion_map)

        audio_source = audio_source.to(self.device_accelerator)
        audio_source = self.model.ae_encoder(audio_source)

        if SamplerType.is_v_sampler(sampler):
            step_list = scheduler.get_step_list(
                steps, self.device_accelerator.type, **scheduler_args
            )
            step_list = step_list[step_list < noise_level]
            alpha_T, sigma_T = t_to_alpha_sigma(step_list[0])
            x_T = alpha_T * audio_source + sigma_T * torch.randn(
                [
                    batch_size,
                    self.model.latent_dim,
                    self.model.native_chunk_size // self.model.downsampling_ratio,
                ],
                generator=self.generator,
                device=self.device_accelerator,
            )
            model = self.model.diffusion
        else:
            scheduler_args.update(
                sigma_max=scheduler_args.get("sigma_max", 1.0) * noise_level
            )
            step_list = scheduler.get_step_list(
                steps, self.device_accelerator.type, **scheduler_args
            )
            x_T = audio_source + step_list[0] * torch.randn(
                [
                    batch_size,
                    self.model.latent_dim,
                    self.model.native_chunk_size // self.model.downsampling_ratio,
                ],
                generator=self.generator,
                device=self.device_accelerator,
            )
            model = VDenoiser(self.model.diffusion)

        sampler_args["extra_args"] = {
            "embedding": embedding,
            "negative_embedding": negative_embedding,
            "embedding_scale": sampler_args["cfg_scale"],
        }

        with self.offload_context(self.model.diffusion):
            x_0 = sampler.sample(
                model, x_T, step_list, callback, **sampler_args
            ).float()

        with self.offload_context(self.model.ae_decoder):
            x_0 = x_0 * self.model.module.aec_divisor
            return self.model.ae_decoder(x_0, num_steps=20)
