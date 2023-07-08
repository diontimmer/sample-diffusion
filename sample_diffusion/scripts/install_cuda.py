import subprocess
import sys


def install_packages(command, uninstall_command):
    # Uninstall previous versions of torch, torchvision and torchaudio
    subprocess.Popen(uninstall_command.split(), stdout=subprocess.PIPE).wait()

    process = subprocess.Popen(
        command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    # Real-time print the output of the installation
    for line in iter(process.stdout.readline, b""):
        print(line.decode().strip())

    process.stdout.close()
    return process.wait()


def install_cuda():
    try:
        # get interpreter exceutable path
        python_path = sys.executable
        uninstall_command = (
            f"{python_path} -m pip uninstall torch torchvision torchaudio -y"
        )
        install_command = f"{python_path} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U"

        print("Trying to install with pip...")
        exit_code = install_packages(install_command, uninstall_command)

        # If pip fails (returns non-zero exit code), fall back to pip3
        if exit_code != 0:
            print("pip failed, trying with pip3...")
            install_command = install_command.replace("pip", "pip3")
            uninstall_command = uninstall_command.replace("pip", "pip3")
            install_packages(install_command, uninstall_command)

    except Exception as e:
        print("Error:", str(e))


if __name__ == "__main__":
    install_cuda()
