from datasets import load_dataset
import fire


def download(path: str = "Djacon/ru-izard-emotions", target_dir: str = "./data"):
    """Download dataset from Hugging Face and save to target_dir"""
    dataset = load_dataset(path)
    target_path = f"{target_dir}/ru-izard-emotions"
    dataset.save_to_disk(target_path)
    print(f"Dataset saved to {target_path}")


if __name__ == "__main__":
    fire.Fire(download)
