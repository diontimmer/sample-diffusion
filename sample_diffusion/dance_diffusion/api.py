import torch
import enum

from dataclasses import dataclass
from typing import Callable

from .base.type import ModelType
from .base.model import ModelWrapperBase, LatentModelWrapperBase
from .base.inference import InferenceBase

import gc

import psutil


def print_memory():
    mem_info = psutil.virtual_memory()
    total_memory = mem_info.total / (1024**3)  # Convert bytes to gigabytes
    used_memory = (mem_info.total - mem_info.available) / (1024**3)
    print(f"Total memory: {total_memory} GB")
    print(f"Used memory: {used_memory} GB")


class RequestType(str, enum.Enum):
    Generation = "Generation"
    Variation = "Variation"
    Interpolation = "Interpolation"
    Inpainting = "Inpainting"
    Extension = "Extension"


class Request:
    def __init__(
        self,
        request_type: RequestType,
        model_path: str,
        lora_path: str,
        lora_strength: float,
        model_type: ModelType,
        model_chunk_size: int,
        model_sample_rate: int,
        aec_path: str = None,
        aec_config: dict = None,
        **kwargs,
    ):
        self.request_type = request_type
        self.model_path = model_path
        self.lora_path = lora_path
        self.lora_strength = lora_strength
        self.model_type = model_type
        self.model_chunk_size = model_chunk_size
        self.model_sample_rate = model_sample_rate
        self.aec_path = aec_path
        self.aec_config = aec_config
        self.kwargs = kwargs


class Response:
    def __init__(self, result: torch.Tensor):
        self.result = result


class RequestHandler:
    def __init__(
        self,
        device_accelerator: torch.device,
        device_offload: torch.device = None,
        optimize_memory_use: bool = False,
        use_autocast: bool = True,
    ):
        self.device_accelerator = device_accelerator
        self.device_offload = device_offload

        self.model_wrapper: ModelWrapperBase = None
        self.inference: InferenceBase = None

        self.optimize_memory_use = optimize_memory_use
        self.use_autocast = use_autocast

    def process_request(self, request: Request, callback: Callable = None) -> Response:
        # load the model from the request if it's not already loaded
        if self.model_wrapper == None:
            self.load_model(
                request.model_type,
                request.model_path,
                request.model_chunk_size,
                request.model_sample_rate,
                request.aec_path,
                request.aec_config,
            )
        elif request.model_path != self.model_wrapper.path:
            del self.inference, self.model_wrapper
            gc.collect()
            self.load_model(
                request.model_type,
                request.model_path,
                request.model_chunk_size,
                request.model_sample_rate,
                request.aec_path,
                request.aec_config,
            )

        elif (request.lora_path != self.model_wrapper.lora_path) or (
            request.lora_strength != self.model_wrapper.lora_strength
        ):
            del self.inference, self.model_wrapper
            gc.collect()
            self.load_model(
                request.model_type,
                request.model_path,
                request.model_chunk_size,
                request.model_sample_rate,
                request.aec_path,
                request.aec_config,
            )

        if (
            request.lora_path != None
            and self.model_wrapper.lora_path != request.lora_path
        ):
            self.model_wrapper.lora_path = request.lora_path
            self.model_wrapper.lora_strength = request.lora_strength
            self.model_wrapper.apply_lora(
                request.lora_path, request.lora_strength, self.device_accelerator
            )
        else:
            self.model_wrapper.lora_path = None
            self.model_wrapper.lora_strength = 1.0

        handlers_by_request_type = {
            RequestType.Generation: self.handle_generation,
            RequestType.Variation: self.handle_variation,
            RequestType.Interpolation: self.handle_interpolation,
            RequestType.Inpainting: self.handle_inpainting,
            RequestType.Extension: self.handle_extension,
        }

        Handler = handlers_by_request_type.get(request.request_type)

        if Handler:
            tensor_result = Handler(request, callback)
        else:
            raise ValueError("Unexpected RequestType in process_request")

        return Response(tensor_result)

    def load_model(self, model_type, model_path, chunk_size, sample_rate, aec_path, aec_config):
        try:
            # Ensure that model_type is a valid ModelType enum
            print("loading model")
            if isinstance(model_type, str):
                model_type = ModelType[model_type]
            assert isinstance(model_type, ModelType)

            # Convert the model_type enum to its string representation
            model_type_str = model_type.name

            # Dynamically import the required modules

            wrapper_module = __import__(
                f"sample_diffusion.dance_diffusion.{model_type_str.lower()}.model",
                fromlist=[f"{model_type_str}ModelWrapper"],
            )
            inference_module = __import__(
                f"sample_diffusion.dance_diffusion.{model_type_str.lower()}.inference",
                fromlist=[f"{model_type_str}Inference"],
            )

            # Dynamically create instances of the required classes

            Wrapper = getattr(wrapper_module, f"{model_type_str}ModelWrapper")
            Inference = getattr(inference_module, f"{model_type_str}Inference")

            self.model_wrapper = Wrapper()

            if isinstance(self.model_wrapper, LatentModelWrapperBase):
                self.model_wrapper.load(
                    model_path,
                    self.device_accelerator,
                    self.optimize_memory_use,
                    chunk_size,
                    sample_rate,
                    aec_path,
                    aec_config,
                )
            else:
                self.model_wrapper.load(
                    model_path,
                    self.device_accelerator,
                    self.optimize_memory_use,
                    chunk_size,
                    sample_rate,
                )
            self.inference = Inference(
                self.device_accelerator,
                self.device_offload,
                self.optimize_memory_use,
                self.use_autocast,
                self.model_wrapper,
            )

        except Exception as e:
            raise ValueError("Unexpected error in load_model: " + str(e))

    def validate_model_type(self, model_type):
        if model_type not in [e.value for e in ModelType]:
            raise ValueError(f"Unexpected ModelType: {model_type}")

    def handle_generation(self, request: Request, callback: Callable) -> Response:
        kwargs = request.kwargs.copy()
        self.validate_model_type(request.model_type)

        return self.inference.generate(
            callback=callback,
            scheduler=kwargs["scheduler_type"],
            sampler=kwargs["sampler_type"],
            **kwargs,
        )

    def handle_variation(self, request: Request, callback: Callable) -> torch.Tensor:
        kwargs = request.kwargs.copy()
        kwargs.update(
            expansion_map=[kwargs["batch_size"]],
            audio_source=kwargs["audio_source"][None, :, :],
        )
        self.validate_model_type(request.model_type)

        return self.inference.generate_variation(
            callback=callback,
            scheduler=kwargs["scheduler_type"],
            sampler=kwargs["sampler_type"],
            **kwargs,
        )

    def handle_interpolation(
        self, request: Request, callback: Callable
    ) -> torch.Tensor:
        kwargs = request.kwargs.copy()
        kwargs.update(
            batch_size=len(kwargs["interpolation_positions"]),
            audio_source=kwargs["audio_source"][None, :, :],
            audio_target=kwargs["audio_target"][None, :, :],
        )
        self.validate_model_type(request.model_type)

        return self.inference.generate_interpolation(
            callback=callback,
            scheduler=kwargs["scheduler_type"],
            sampler=kwargs["sampler_type"],
            **kwargs,
        )

    def handle_inpainting(self, request: Request, callback: Callable) -> torch.Tensor:
        kwargs = request.kwargs.copy()
        kwargs.update(
            expansion_map=[kwargs["batch_size"]],
            audio_source=kwargs["audio_source"][None, :, :],
        )
        self.validate_model_type(request.model_type)

        return self.inference.generate_inpainting(
            callback=callback,
            scheduler=kwargs["scheduler_type"],
            sampler=kwargs["sampler_type"],
            **kwargs,
        )

    def handle_extension(self, request: Request, callback: Callable) -> torch.Tensor:
        kwargs = request.kwargs.copy()
        kwargs.update(
            expansion_map=[kwargs["batch_size"]],
            audio_source=kwargs["audio_source"][None, :, :],
        )
        self.validate_model_type(request.model_type)

        return self.inference.generate_extension(
            callback=callback,
            scheduler=kwargs["scheduler_type"],
            sampler=kwargs["sampler_type"],
            **kwargs,
        )
