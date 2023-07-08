from archisound import ArchiSound
import torchaudio
import torch
import random
import os


def get_std(autoencoder_encode, data_folder):
    # pick random file from dataset
    dataset_files = os.listdir(data_folder)
    path = os.path.join(
        data_folder,
        random.choice(dataset_files),
    )

    audio, _ = torchaudio.load(path)

    # Check if audio is long enough
    if audio.shape[1] < 2097152:
        raise ValueError(f"File {path} is too short for a chunk of 2097152 samples")

    # get random chunk from audio of 2097152 samples {44s}
    start = random.randint(0, audio.shape[1] - 2097152)
    audio = audio[:, start : start + 2097152]

    reals = audio.unsqueeze(0)  # batch 1

    reals = reals.to("cuda")

    fakes = autoencoder_encode(reals)

    print(f"reals: {reals.shape}")
    print(f"fakes: {fakes.shape}")

    std = torch.std(fakes)

    return std


if __name__ == "__main__":
    # load autoencoder and set autoencoder_encode to your encode function
    autoencoder = ArchiSound.from_pretrained("dmae1d-ATC32-v3")
    autoencoder = autoencoder.to("cuda")
    autoencoder_encode = autoencoder.encode
    data_folder = "music"

    stds = []
    for i in range(10):
        stds.append(get_std(autoencoder_encode, data_folder))
    print(f"average std: {sum(stds) / len(stds)}")
