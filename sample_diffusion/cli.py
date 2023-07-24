import json, torch, argparse, os, logging, sys


from .util.util import load_audio, save_audio, crop_audio
from .util.platform import get_torch_device_type
from .dance_diffusion.api import (
    RequestHandler,
    Request,
    Response,
    RequestType,
    ModelType,
)
from .diffusion_library.sampler import SamplerType
from .diffusion_library.scheduler import SchedulerType
from transformers import logging as transformers_logging
import json


def main():
    args = parse_cli_args()
    run(args)


def run(args, request_handler=None):
    if args.get("argsfile") is None and args.get("model") is None:
        raise ValueError("Either argsfile or model must be provided.")

    args["modeltype"] = (
        ModelType[args.get("modeltype")]
        if isinstance(args.get("modeltype"), str)
        else args.get("modeltype")
    )

    args["sampler"] = (
        SamplerType[args.get("sampler")]
        if isinstance(args.get("sampler"), str)
        else args.get("sampler")
    )

    args["schedule"] = (
        SchedulerType[args.get("schedule")]
        if isinstance(args.get("schedule"), str)
        else args.get("schedule")
    )

    args["mode"] = (
        RequestType[args.get("mode")]
        if isinstance(args.get("mode"), str)
        else args.get("mode")
    )

    device_type_accelerator = (
        args.get("device_accelerator")
        if (args.get("device_accelerator") != None)
        else get_torch_device_type()
    )
    device_accelerator = torch.device(device_type_accelerator)
    device_offload = torch.device(args.get("device_offload"))
    if args.get("autoencoder"):
        with open(args.get("autoencoder_config")) as f:
            args["autoencoder_config"] = json.load(f)

    crop = (
        lambda audio: crop_audio(audio, args.get("chunk_size"), args.get("crop_offset"))
        if args.get("crop_offset") is not None
        else audio
    )
    load_input = (
        lambda source: crop(
            load_audio(device_accelerator, source, args.get("sample_rate"))
        )
        if source is not None
        else None
    )

    request_handler = (
        RequestHandler(
            device_accelerator,
            device_offload,
            optimize_memory_use=args.get("optimize_memory_use"),
            use_autocast=args.get("use_autocast"),
        )
        if request_handler is None
        else request_handler
    )

    os.makedirs(args.get("output"), exist_ok=True)

    paths = []

    for i in range(args.get("batch_loops")):
        seed = (
            args.get("seed")
            if (args.get("seed") != -1)
            else torch.randint(
                0, 4294967294, [1], device=device_type_accelerator
            ).item()
        )
        print(
            f"Now generating batch {i+1}/{args.get('batch_loops')} | Using accelerator: {device_type_accelerator} | Seed: {seed}."
        )
        request = Request(
            request_type=args.get("mode"),
            model_path=args.get("model"),
            lora_path=args.get("lora_path"),
            lora_strength=args.get("lora_strength"),
            model_type=args.get("model_type"),
            model_chunk_size=args.get("chunk_size"),
            model_sample_rate=args.get("sample_rate"),
            seed=seed,
            batch_size=args.get("batch_size"),
            audio_source=load_input(args.get("audio_source")),
            audio_target=load_input(args.get("audio_target")),
            mask=torch.load(args.get("mask")) if (args.get("mask") != None) else None,
            noise_level=args.get("noise_level"),
            interpolation_positions=args.get("interpolations")
            if (args.get("interpolations_linear") == None)
            else torch.linspace(
                0, 1, args.get("interpolations_linear"), device=device_accelerator
            ),
            keep_start=args.get("keep_start"),
            steps=args.get("steps"),
            sampler_type=args.get("sampler"),
            sampler_args=args.get("sampler_args"),
            scheduler_type=args.get("schedule"),
            scheduler_args=args.get("schedule_args"),
            inpainting_args=args.get("inpainting_args"),
            aec_path=args.get("autoencoder"),
            aec_config=args.get("autoencoder_config"),
        )

        response = request_handler.process_request(request)
        loop_paths = save_audio(
            (0.5 * response.result).clamp(-1, 1)
            if (args.get("tame") == True)
            else response.result,
            f"{args.get('output')}/{args.get('model_type')}/{args.get('mode')}/",
            args.get("sample_rate"),
            f"{seed}",
            args.get("trim_silence"),
        )
        paths.extend(loop_paths)
    return paths


def str2bool(value):
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--argsfile",
        type=str,
        default=None,
        help="When used, uses args from a provided .json file instead of using the passed cli args.",
    )
    parser.add_argument(
        "--crop_offset",
        type=int,
        default=0,
        help="The starting sample offset to crop input audio to. Use -1 for random cropping.",
    )
    parser.add_argument(
        "--optimize_memory_use",
        type=str2bool,
        default=True,
        help="Try to minimize memory use during execution, might decrease performance.",
    )
    parser.add_argument(
        "--use_autocast", type=str2bool, default=True, help="Use autocast."
    )
    parser.add_argument(
        "--use_autocrop",
        type=str2bool,
        default=True,
        help="Use autocrop(automatically crops audio provided to chunk_size).",
    )
    parser.add_argument(
        "--device_accelerator", type=str, default=None, help="Device of execution."
    )
    parser.add_argument(
        "--device_offload",
        type=str,
        default="cpu",
        help="Device to store models when not in use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the model checkpoint file to be used.",
        default=None,
    )
    parser.add_argument(
        "--autoencoder",
        type=str,
        help="Path to the autoencoder checkpoint file to be used.",
        default=None,
    )
    parser.add_argument(
        "--autoencoder_config",
        type=str,
        help="Path to the autoencoder config json to be used.",
        default=None,
    )
    parser.add_argument(
        "--model_type",
        type=ModelType,
        choices=ModelType,
        default=ModelType.DD,
        help="The model type.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=48000,
        help="The samplerate the model was trained on.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=65536,
        help="The native chunk size of the model.",
    )
    parser.add_argument(
        "--mode",
        type=RequestType,
        choices=RequestType,
        default=RequestType.Generation,
        help="The mode of operation (Generation, Variation, Interpolation, Inpainting, Extension or Upscaling).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="The seed used for reproducable outputs. Leave empty for random seed.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The maximal number of samples to be produced per batch.",
    )
    parser.add_argument(
        "--batch_loops",
        type=int,
        default=1,
        help="The number of batches the generation will output.",
    )
    parser.add_argument(
        "--audio_source", type=str, default=None, help="Path to the audio source."
    )
    parser.add_argument(
        "--audio_target",
        type=str,
        default=None,
        help="Path to the audio target (used for interpolations).",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to the mask tensor (used for inpainting).",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.7,
        help="The noise level used for variations & interpolations.",
    )
    parser.add_argument(
        "--interpolations_linear",
        type=int,
        default=1,
        help="The number of interpolations, even spacing.",
    )
    parser.add_argument(
        "--interpolations",
        nargs="+",
        type=float,
        default=None,
        help="The interpolation positions.",
    )
    parser.add_argument(
        "--keep_start",
        type=str2bool,
        default=True,
        help="Keep beginning of audio provided(only applies to mode Extension).",
    )
    parser.add_argument(
        "--tame", type=str2bool, default=True, help="Decrease output by 3db, then clip."
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="The number of steps for the sampler."
    )
    parser.add_argument(
        "--sampler",
        type=SamplerType,
        choices=SamplerType,
        default=SamplerType.V_IPLMS,
        help="The sampler used for the diffusion model.",
    )
    parser.add_argument(
        "--sampler_args",
        type=json.loads,
        default={
            "aec_divisor": 2.5,
            "cfg_scale": 3,
            "text_condition": "",
            "inverse_text_condition": "",
        },
        help="Additional arguments of the DD sampler.",
    )
    parser.add_argument(
        "--schedule",
        type=SchedulerType,
        choices=SchedulerType,
        default=SchedulerType.V_CRASH,
        help="The schedule used for the diffusion model.",
    )
    parser.add_argument(
        "--schedule_args",
        type=json.loads,
        default={},
        help="Additional arguments of the DD schedule.",
    )
    parser.add_argument(
        "--inpaint_args", type=json.loads, default={}, help="Arguments for inpainting."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_diffusion_output",
        help="The folder to save the output to.",
    )
    parser.add_argument(
        "--trim_silence",
        type=str2bool,
        default=True,
        help="Trim silence from the beginning and end of the output.",
    )

    args = parser.parse_args()

    if args.argsfile is not None:
        if os.path.exists(args.argsfile):
            with open(args.argsfile, "r") as f:
                print(f"Using cli args from file: {args.argsfile}")
                args = json.load(f)

                # parse enum objects from strings & apply defaults
                args["sampler"] = SamplerType(args.get("sampler", SamplerType.V_IPLMS))
                args["schedule"] = SchedulerType(
                    args.get("schedule", SchedulerType.V_CRASH)
                )

                return args
        else:
            print(f"Could not locate argsfile: {args.argsfile}")

    return vars(args)


if __name__ == "__main__":
    transformers_logging.set_verbosity_error()
    main()
