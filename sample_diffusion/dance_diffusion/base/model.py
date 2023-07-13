import torch
from sample_diffusion.diffusion_library.audio_lora import (
    AudioLoRANetwork,
    AudioLoRAModule,
)


class ModelWrapperBase:
    def __init__(self):
        # self.uuid: str = None
        # self.name: str = None
        self.path: str = None
        self.lora_path: str = None
        self.lora_strength: float = 1.0

        self.device_accelerator: torch.device = None

        self.chunk_size: int = None
        self.sample_rate: int = None

    def load(
        self,
        path: str,
        device_accelerator: torch.device,
        optimize_memory_use: bool = False,
        chunk_size: int = 131072,
        sample_rate: int = 48000,
    ):
        raise NotImplementedError

    def apply_lora(self, lora_path, lora_strength, device):
        UNET1D_TARGET_REPLACE_MODULE = ["SelfAttention1d", "ResConvBlock"]

        lora_info = torch.load(lora_path)
        lora_dim = lora_info["unet_loras.0.lora_down.weight"].shape[0]
        del lora_info

        print(
            f"Applying LoRAW (RANK:{lora_dim}) with strength {lora_strength}: {lora_path}"
        )

        # lora_strength can never be 0
        lora_strength = max(lora_strength, 1e-6)

        lora = AudioLoRANetwork(
            self.module.diffusion_ema,
            target_modules=UNET1D_TARGET_REPLACE_MODULE,
            multiplier=1,
            lora_dim=lora_dim,
            alpha=lora_strength,
            module_class=AudioLoRAModule,
            verbose=False,
        )
        lora.to(device=device)
        lora.apply_to()
        lora.load_weights(lora_path)
