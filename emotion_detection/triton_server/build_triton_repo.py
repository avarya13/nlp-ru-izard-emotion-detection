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
    model_name: str,
    engine_path: Path,
    triton_repo_dir: Path,
    num_labels: int,
):
    print(f"Preparing TensorRT Triton model: {model_name}")

    triton_model_dir = triton_repo_dir / model_name
    version_dir = triton_model_dir / "1"

    version_dir.mkdir(parents=True, exist_ok=True)

    destination_path = version_dir / "model.plan"

    shutil.copy2(engine_path, destination_path)

    config_template = """
name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: 0

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ 1, -1 ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ 1, -1 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ 1, {num_labels} ]
  }}
]
"""

    config_path = triton_model_dir / "config.pbtxt"

    config_path.write_text(
        config_template.format(
            model_name=model_name,
            num_labels=num_labels,
        )
    )

    print(f"TensorRT engine copied to: {destination_path}")
    print(f"Config created: {config_path}")


def build_repo(cfg: DictConfig):
    trt_base = Path(cfg.paths.tensorrt_dir)

    triton_repo_dir = Path(cfg.paths.triton_model_repository)

    num_labels = cfg.data.num_labels

    triton_repo_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = [d for d in trt_base.iterdir() if d.is_dir()]

    if not model_dirs:
        print("No TensorRT models found")
        return

    for model_dir in model_dirs:
        latest_dir = get_latest_timestamp_dir(model_dir)

        if latest_dir is None:
            print(f"Skipping {model_dir.name}: no timestamp dirs")
            continue

        engine_path = latest_dir / "model.engine"

        if not engine_path.exists():
            print(f"Skipping {model_dir.name}: model.engine not found")
            continue

        prepare_model(
            model_name=model_dir.name,
            engine_path=engine_path,
            triton_repo_dir=triton_repo_dir,
            num_labels=num_labels,
        )

    print("\nTensorRT Triton repository prepared successfully")
