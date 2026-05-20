import subprocess

import fire
from datasets import load_dataset


def download(
    path: str = "Djacon/ru-izard-emotions",
    target_dir: str = "./data",
    remote_name: str = "ruizard-emotions-data",
):
    """Download dataset from Hugging Face and save to target_dir"""
    dataset = load_dataset(path)
    target_path = f"{target_dir}/ru-izard-emotions"
    dataset.save_to_disk(target_path)
    print(f"Dataset saved to {target_path}")

    subprocess.run(["dvc", "add", target_path], check=True)
    subprocess.run(["dvc", "push", "-r", remote_name], check=True)
    print("Dataset added to DVC and pushed to remote storage")


if __name__ == "__main__":
    fire.Fire(download)
