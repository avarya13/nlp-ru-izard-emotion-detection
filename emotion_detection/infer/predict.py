from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    return tokenizer, model


def predict_emotion(text: str, tokenizer, model, labels):
    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze(0)
    return {labels[i]: probs[i].item() for i in range(len(labels))}


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    if not hasattr(cfg, "text") or cfg.text is None:
        raise ValueError("Please provide text via +text='Your text here'")
    text = cfg.text

    script_dir = Path(__file__).parent
    labels_path = script_dir / "../../configs/data/labels.yaml"
    cfg_labels = OmegaConf.load(labels_path.resolve())
    labels = cfg_labels.labels

    model_path = Path(cfg.paths.save_dir) / cfg.model.model_name.replace("/", "-")
    tokenizer, model = load_model(model_path)

    emotions = predict_emotion(text, tokenizer, model, labels)

    for label, prob in sorted(emotions.items(), key=lambda x: -x[1]):
        print(f"{label.title().ljust(12)}: {prob:.4f}")


if __name__ == "__main__":
    main()
