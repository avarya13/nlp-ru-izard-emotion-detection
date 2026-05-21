from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scripts.triton_client import TritonClient
from transformers import AutoTokenizer


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

        result = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=self.output_names,
        )

        return result


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig):
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

    logits = result["logits"][0]

    probs = 1 / (1 + np.exp(-logits))

    emotions = {labels[i]: probs[i].item() for i in range(len(labels))}

    for label, prob in sorted(emotions.items(), key=lambda x: -x[1]):
        print(f"{label.title().ljust(12)}: {prob:.4f}")


if __name__ == "__main__":
    main()
