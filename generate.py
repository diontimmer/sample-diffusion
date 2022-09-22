import os, argparse, math, gc
import torchaudio
import torch
import json
from torch import nn

from einops import rearrange
from diffusion import sampling
from audio_diffusion.models import DiffusionAttnUnet1D
from audio_diffusion.utils import Stereo, PadCrop

from sample_diffusion.model import load_model
from sample_diffusion.inference import generate_audio

def main():
    args = parse_cli_args()

    model, device = load_model(args)
    audio_out = generate_audio(args, args, device, model)

    save_audio(args, audio_out)

def save_audio(args, audio_out):
    output_path = get_output_path(args)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for ix, sample in enumerate(audio_out):
        output_file = os.path.join(output_path, f'sample #{ix + 1}.wav')
        open(output_file, 'a').close()
        output = sample.cpu()
        torchaudio.save(output_file, output, args.sr)
        
    print(f'Your samples are waiting for you here: {output_path}')
    print(f'Seed: {args.seed}, Steps: {args.n_steps}, Noise: {args.noise_level}')

def load_number_file(path):
    with open(path, 'r') as f:
        return int(f.read())

def save_number_file(path, num):
    with open(path, 'w') as f:
        f.write(str(num))

def write_to_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

def get_output_path(args):
    seed = args.seed
    steps = args.n_steps
    noise = args.noise_level

    val = load_number_file("value.txt")
    
    if args.input:
        parent_folder = os.path.join(args.out_path, f'variations{val}')
    else:
        parent_folder = os.path.join(args.out_path, f'generations{val}')

    write_to_json(args, os.path.join(parent_folder, 'meta.json'))

    return parent_folder
    
def parse_cli_args():
    parser = argparse.ArgumentParser()

    # args for model
    parser.add_argument("--ckpt", type = str, default = "models/model.ckpt", help = "path to the model to be used")
    parser.add_argument("--spc", type = int, default = 65536, help = "the samples per chunk of the model")
    parser.add_argument("--sr", type = int, default = 48000, help = "the samplerate of the model")

    # args for generation
    parser.add_argument("--out_path", type = str, default = "audio_out", help = "path to the folder for the samples to be saved in")
    parser.add_argument("--sample_length_multiplier", type = int, default = 1, help = "sample length multiplier for audio2audio")
    parser.add_argument("--input_sr", type = int, default = 44100, help = "samplerate of the input audio specified in --input")
    parser.add_argument("--noise_level", type = float, default = 0.7, help = "noise level for audio2audio")
    parser.add_argument("--n_steps", type = int, default = 25, help = "number of sampling steps")
    parser.add_argument("--n_samples", type = int, default = 1, help = "how many samples to produce / batch size")
    parser.add_argument("--seed", type = int, default = -1, help = "the seed (for reproducible sampling)")
    parser.add_argument("--input", type = str, default = '', help = "path to the audio to be used for audio2audio")

    return parser.parse_args()

if __name__ == "__main__":
    main()