import enum
import os
from pathlib import Path


def get_modules():
    return [
        f.name.upper()
        for f in Path(os.path.abspath(__file__)).parent.parent.glob("*")
        if f.is_dir() and f.name not in ["base", "__pycache__"]
    ]


ModelType = enum.Enum("ModelType", {module: module for module in get_modules()})
