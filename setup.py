"""Install packages as defined in this file into the Python environment."""
from setuptools import setup, find_packages

# The version of this tool is based on the following steps:
# https://packaging.python.org/guides/single-sourcing-package-version/
VERSION = {}

setup(
    name="Sample Diffusion",
    author="sudosilico, mrhyazinth",
    author_email="",
    url="",
    description="A Python library and CLI for generating audio samples using Harmonai Dance Diffusion models.",
    version=VERSION.get("__version__", "0.0.1"),
    packages=["sample_diffusion"],
    install_requires=[
        "torch",
        "torchaudio",
        "tqdm",
        "v-diffusion-pytorch",
        "k-diffusion",
        "pysoundfile",
        "black",
        "diffusers",
        "transformers",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Topic :: Utilities",
        "Programming Language :: Python",
    ],
)
