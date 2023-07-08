# sample-diffusion

A Python library and CLI for generating audio samples using Harmonai [Dance Diffusion](https://github.com/Harmonai-org/sample-generator) models.

🚧 This project is early in development. Expect breaking changes! 🚧

This is my personal fork, its pip installable and has some extra modules.

## Features

- A CLI for generating audio samples from the command line using Dance Diffusion models. (`python -m sample_diffusion.cli`)
- A script for reducing the file size of Dance Diffusion models by removing data that is only needed for training and not inference. (`python -m sample_diffusion.scripts.trim_model`)

## Installation

### Requirements

- [git](https://git-scm.com/downloads) (to clone the repo)

### Install

```sh
pip install git+https://github.com/diontimmer/sample-diffusion-lib
```

## Using the CLI

### Generating samples

Pass a custom model path and the output folder as arguments (will create folder if does not exist):

```sh
python -m sample_diffusion.cli --model "models/DD/some-model.ckpt" --output "audio_output"
```

### `cli` Command Line Arguments

| argument                  | type             | default                | desc                                                                                   |
|---------------------------|------------------|------------------------|----------------------------------------------------------------------------------------|
| `--argsfile`              | str              | None                   | Path to JSON file containing cli args. If used, other passed cli args are ignored.     |
| `--use_autocast`          | bool             | True                   | Use autocast.                                                                          |
| `--crop_offset`           | int              | 0                      | The starting sample offset to crop input audio to. Use -1 for random cropping.         |
| `--device_accelerator`    | str              | None                   | Device of execution.                                                                   |
| `--device_offload`        | str              | `cpu`                  | Device to store models when not in use.                                                |
| `--model`                 | str              | `models/dd/model.ckpt` | Path to the model checkpoint file to be used (default: models/dd/model.ckpt).          |
| `--sample_rate`           | int              | 48000                  | The samplerate the model was trained on.                                               |
| `--chunk_size`            | int              | 65536                  | The native chunk size of the model.                                                    |
| `--mode`                  | RequestType      | `Generation`           | The mode of operation (Generation, Variation, Interpolation, Inpainting or Extension). |
| `--seed`                  | int              | -1 (Random)            | The seed used for reproducable outputs. Leave empty for random seed.                   |
| `--batch_size`            | int              | 1                      | The maximal number of samples to be produced per batch.                                |
| `--audio_source`          | str              | None                   | Path to the audio source.                                                              |
| `--audio_target`          | str              | None                   | Path to the audio target (used for interpolations).                                    |
| `--mask`                  | str              | None                   | Path to the mask tensor (used for inpainting).                                         |
| `--noise_level`           | float            | 0.7                    | The noise level used for variations & interpolations.                                  |
| `--interpolations_linear` | int              | 1                      | The number of interpolations, even spacing.                                            |
| `--interpolations`        | float or float[] | None                   | The interpolation positions.                                                           |
| `--keep_start`            | bool             | True                   | Keep beginning of audio provided(only applies to mode Extension).                      |
| `--tame`                  | bool             | True                   | Decrease output by 3db, then clip.                                                     |
| `--steps`                 | int              | 50                     | The number of steps for the sampler.                                                   |
| `--sampler`               | SamplerType      | `IPLMS`                | The sampler used for the diffusion model.                                              |
| `--sampler_args`          | Json String      | `{}`                   | Additional arguments of the DD sampler.                                                |
| `--schedule`              | SchedulerType    | `CrashSchedule`        | The schedule used for the diffusion model.                                             |
| `--schedule_args`         | Json String      | `{}`                   | Additional arguments of the DD schedule.                                               |
| `--inpainting_args`       | Json String      | `{}`                   | Additional arguments for inpainting (currently unsupported)                            |

### Using args.json
Instead of specifying all the necessary arguments each time we encourage you to try using the args.json file provided with this library:
```sh
python -m sample_diffusion.cli --argsfile 'args.json'
```
To change any settings you can edit the args.json file.

## Using the model trimming script

`python -m sample_diffusion.scripts.trim_model` can be used to reduce the file size of Dance Diffusion models by removing data that is only needed for training and not inference. For our first models, this reduced the model size by about 75% (from 3.46 GB to 0.87 GB).

To use it, simply pass the path to the model you want to trim as an argument:

```sh
python -m sample_diffusion.scripts.trim_model models/model.ckpt
```

This will create a new model file at `models/model_trim.ckpt`.
