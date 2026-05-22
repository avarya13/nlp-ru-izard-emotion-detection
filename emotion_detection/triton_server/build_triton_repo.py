import shutil
from pathlib import Path

from omegaconf import DictConfig


def get_latest_timestamp_dir(model_dir: Path) -> Path | None:
    timestamp_dirs = [
        d
        for d in model_dir.iterdir()
        if d.is_dir() and len(d.name) == 13 and d.name[8] == "-"
    ]
    return max(timestamp_dirs, key=lambda d: d.name) if timestamp_dirs else None


def prepare_model(
    model_name: str, onnx_path: Path, triton_repo_dir: Path, num_labels: int
):
    print(f"Preparing Triton model: {model_name}")
    triton_model_dir = triton_repo_dir / model_name
    version_dir = triton_model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    destination_onnx = version_dir / "model.onnx"
    shutil.copy2(onnx_path, destination_onnx)

    config_template = """
        name: "{model_name}"
        platform: "onnxruntime_onnx"
        max_batch_size: 8
        input [
        {{
            name: "input_ids"
            data_type: TYPE_INT64
            dims: [ -1 ]
        }},
        {{
            name: "attention_mask"
            data_type: TYPE_INT64
            dims: [ -1 ]
        }}
        ]
        output [
        {{
            name: "logits"
            data_type: TYPE_FP32
            dims: [ {num_labels} ]
        }}
        ]
        dynamic_batching {{ }}
    """
    config_path = triton_model_dir / "config.pbtxt"
    config_path.write_text(
        config_template.format(model_name=model_name, num_labels=num_labels)
    )
    print(f"ONNX copied to: {destination_onnx}")
    print(f"Config created: {config_path}")


def build_repo(cfg: DictConfig):
    onnx_base = Path(cfg.paths.onnx_dir)
    triton_repo_dir = Path(cfg.paths.triton_model_repository)
    num_labels = cfg.data.num_labels

    triton_repo_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = [d for d in onnx_base.iterdir() if d.is_dir()]
    if not model_dirs:
        print("No ONNX models found")
        return

    for model_dir in model_dirs:
        latest_dir = get_latest_timestamp_dir(model_dir)
        if latest_dir is None:
            print(f"Skipping {model_dir.name}: no timestamp dirs")
            continue
        onnx_path = latest_dir / "model.onnx"
        if not onnx_path.exists():
            print(f"Skipping {model_dir.name}: model.onnx not found")
            continue
        prepare_model(model_dir.name, onnx_path, triton_repo_dir, num_labels)

    print("\nTriton model repository prepared successfully")


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
    def main(cfg: DictConfig):
        build_repo(cfg)
