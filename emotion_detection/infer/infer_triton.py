import json
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from .triton_client import TritonClient


class TritonNLPInfer:
    def __init__(
        self,
        triton_model_name: str,
        hf_model_name: str,
        triton_url: str,
    ):
        self.client = TritonClient(triton_url)

        self.model_name = triton_model_name

        self.metadata = self.client.get_model_metadata(triton_model_name)

        self.input_names = [i["name"] for i in self.metadata["inputs"]]

        self.output_names = [o["name"] for o in self.metadata["outputs"]]

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    def predict(
        self,
        text: str,
        max_length: int,
    ):
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        inputs = {
            name: enc[name].astype(np.int64) for name in self.input_names if name in enc
        }

        try:
            result = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=self.output_names,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to run inference via Triton: {e}") from e

        return result


def apply_activation(
    logits: np.ndarray,
    activation: str,
) -> np.ndarray:
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-logits))

    elif activation == "softmax":
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    else:
        raise ValueError(f"Unsupported activation: {activation}")


def run_triton_infer(cfg: DictConfig):
    if not hasattr(cfg, "text") or cfg.text is None:
        raise ValueError("Please provide text via +text='Your text here'")

    text = cfg.text

    script_dir = Path(__file__).parent

    labels_path = script_dir / "../../configs/data/labels.yaml"

    cfg_labels = OmegaConf.load(labels_path.resolve())

    labels = cfg_labels.labels

    infer = TritonNLPInfer(
        triton_model_name=cfg.model.triton_name,
        hf_model_name=cfg.model.model_name,
        triton_url=cfg.triton.url,
    )

    result = infer.predict(
        text=text,
        max_length=cfg.data.max_length,
    )

    output_name = infer.output_names[0]

    logits = result[output_name][0]

    probs = apply_activation(
        logits=logits,
        activation=cfg.model.activation,
    )

    emotions = {labels[i]: float(probs[i]) for i in range(len(labels))}

    emotions = dict(sorted(emotions.items(), key=lambda x: -x[1]))

    print("\nPredicted emotions:\n")

    for label, prob in emotions.items():
        print(f"{label.title().ljust(12)}: " f"{prob:.4f}")

    save_json = cfg.paths.get(
        "save_json",
        False,
    )

    if save_json:
        output_dir = Path(
            cfg.paths.get(
                "triton_infer_dir",
                "triton_infer_results",
            )
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        output_path = output_dir / f"triton_prediction_{timestamp}.json"

        payload = {
            "text": text,
            "model": cfg.model.model_name,
            "activation": cfg.model.activation,
            "predictions": {label: f"{prob:.4f}" for label, prob in emotions.items()},
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\nSaved predictions to: " f"{output_path}")


if __name__ == "__main__":
    run_triton_infer()
