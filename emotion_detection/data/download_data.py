import subprocess

from datasets import load_dataset


def download_data(cfg):
    """Download dataset from Hugging Face and save to target_dir"""
    dataset = load_dataset(cfg.data_url)
    target_path = cfg.data_dir
    dataset.save_to_disk(target_path)
    print(f"Dataset saved to {target_path}")

    subprocess.run(["dvc", "add", target_path], check=True)
    subprocess.run(["dvc", "push", "-r", cfg.remote_name], check=True)
    print("Dataset added to DVC and pushed to remote storage")


if __name__ == "__main__":
    download_data()
