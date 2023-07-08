import argparse
import torch
from torch.nn.parameter import Parameter
import os


def main(args):
    file = torch.load(args.input_file, map_location="cpu")

    new_state_dict = {}

    for name, param in file["state_dict"].items():
        if name.startswith("diffusion_ema.ema_model."):
            new_name = name.replace("diffusion_ema.ema_model.", "")
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            new_state_dict[new_name] = param

    model = dict(state_dict=new_state_dict)

    output_file = os.path.splitext(args.output_file)[0] + "_inference.ckpt"

    torch.save(model, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output file paths")
    parser.add_argument(
        "--input_file", type=str, required=True, help="path to input file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="path to output file"
    )
    args = parser.parse_args()

    main(args)
