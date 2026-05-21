from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    base_model_dir = Path(cfg.paths.save_dir) / cfg.model.model_name.replace("/", "_")

    timestamp = cfg.paths.timestamp

    if timestamp:
        model_dir = base_model_dir / timestamp
        print(f"Using specified model directory: {model_dir}")

    else:
        if base_model_dir.exists():
            subdirs = [
                d
                for d in base_model_dir.iterdir()
                if d.is_dir() and len(d.name) == 13 and d.name[8] == "-"
            ]

            if not subdirs:
                raise ValueError(
                    f"No timestamped model directories found in {base_model_dir}"
                )

            latest = max(subdirs, key=lambda d: d.name)
            model_dir = latest

            print(f"Using latest model directory: {model_dir}")

        else:
            raise ValueError(f"Model directory does not exist: {base_model_dir}")

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.eval()

    output_dir = Path(cfg.paths.onnx_dir) / cfg.model.model_name.replace("/", "_")

    export_timestamp = datetime.now().strftime("%Y%m%d-%H%M")

    output_dir = output_dir / export_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "model.onnx"

    dummy_input_ids = torch.randint(
        low=0,
        high=1000,
        size=(1, cfg.data.max_length),
        dtype=torch.long,
    )

    dummy_attention_mask = torch.ones(
        (1, cfg.data.max_length),
        dtype=torch.long,
    )

    print("Exporting model to ONNX...")

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {
                0: "batch_size",
                1: "sequence_length",
            },
            "attention_mask": {
                0: "batch_size",
                1: "sequence_length",
            },
            "logits": {
                0: "batch_size",
            },
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    main()
