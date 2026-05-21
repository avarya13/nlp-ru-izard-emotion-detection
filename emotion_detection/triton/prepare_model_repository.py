import shutil
from pathlib import Path

NUM_LABELS = 10

ONNX_MODELS_DIR = Path("onnx_models")
TRITON_REPOSITORY_DIR = Path("triton/model_repository")


CONFIG_TEMPLATE = """
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


def get_latest_timestamp_dir(model_dir: Path) -> Path | None:
    timestamp_dirs = [
        d
        for d in model_dir.iterdir()
        if d.is_dir() and len(d.name) == 13 and d.name[8] == "-"
    ]

    if not timestamp_dirs:
        return None

    return max(timestamp_dirs, key=lambda d: d.name)


def prepare_model(model_name: str, onnx_path: Path):
    print(f"Preparing Triton model: {model_name}")

    triton_model_dir = TRITON_REPOSITORY_DIR / model_name
    version_dir = triton_model_dir / "1"

    version_dir.mkdir(parents=True, exist_ok=True)

    destination_onnx = version_dir / "model.onnx"
    # destination_onnx_data = version_dir / "model.onnx.data"

    shutil.copy2(onnx_path, destination_onnx)
    # shutil.copy2(onnx_path, destination_onnx_data)

    config_text = CONFIG_TEMPLATE.format(
        model_name=model_name,
        num_labels=NUM_LABELS,
    )

    config_path = triton_model_dir / "config.pbtxt"

    config_path.write_text(config_text)

    print(f"ONNX copied to: {destination_onnx}")
    print(f"Config created: {config_path}")


def main():
    TRITON_REPOSITORY_DIR.mkdir(parents=True, exist_ok=True)

    model_dirs = [d for d in ONNX_MODELS_DIR.iterdir() if d.is_dir()]

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

        prepare_model(
            model_name=model_dir.name,
            onnx_path=onnx_path,
        )

    print("\nTriton model repository prepared successfully")


if __name__ == "__main__":
    main()
