# sample-diffusion

A Python library and CLI for generating audio samples using Harmonai [Dance Diffusion](https://github.com/Harmonai-org/sample-generator) models.

🚧 This project is early in development. Expect breaking changes! 🚧

## Features

- A CLI for generating audio samples from the command line using Dance Diffusion models. (`cli.py`)
- A script for reducing the file size of Dance Diffusion models by removing data that is only needed for training and not inference. (`scripts/trim_model.py`)

## Installation

### Requirements

- [git](https://git-scm.com/downloads) (to clone the repo)
- [conda](https://docs.conda.io/en/latest/) (to set up the python environment)

`conda` can be installed through [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). To run on an Apple Silicon device, you will need to use a conda installation that includes Apple Silicon support, such as [Miniforge](https://github.com/conda-forge/miniforge).

### Cloning the repo

Clone the repo and `cd` into it:

```sh
git clone https://github.com/sudosilico/sample-diffusion
cd sample-diffusion
```

### Setting up the conda environment

Create the `conda` environment:

```sh
# If you're not running on an Apple Silicon machine:
conda env create -f environment.yml

# For Apple Silicon machines:
conda env create -f environment-mac.yml
```

This may take a few minutes as it will install all the necessary Python dependencies so that they will be available to the CLI script.

> Note: You must activate the `dd` conda environment after creating it. You can do this by running `conda activate dd` in the terminal. You will need to do this every time you open a new terminal window. Learn more about [conda environments.](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html)

```sh
conda activate dd
```

## Using the `cli.py` CLI

### Generating samples

Pass a custom model path and the output folder as arguments (will create folder if does not exist):

```sh
python -m sample_diffusion.cli --model "models/DD/some-model.ckpt" --output "audio_output"
```

### `cli.py` Command Line Arguments

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
python cli.py --argsfile 'args.json'
```
To change any settings you can edit the args.json file.

## Using the model trimming script

`scripts/trim_model.py` can be used to reduce the file size of Dance Diffusion models by removing data that is only needed for training and not inference. For our first models, this reduced the model size by about 75% (from 3.46 GB to 0.87 GB).

To use it, simply pass the path to the model you want to trim as an argument:

```sh
python scripts/trim_model.py models/model.ckpt
```

This will create a new model file at `models/model_trim.ckpt`.
