from pathlib import Path

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


def run_infer(cfg: DictConfig):
    if not hasattr(cfg, "text") or cfg.text is None:
        raise ValueError("Please provide text via +text='Your text here'")
    text = cfg.text

    script_dir = Path(__file__).parent
    labels_path = script_dir / "../../configs/data/labels.yaml"
    cfg_labels = OmegaConf.load(labels_path.resolve())
    labels = cfg_labels.labels

    base_model_dir = Path(cfg.paths.save_dir) / cfg.model.model_name.replace("/", "-")
    if hasattr(cfg.paths, "timestamp") and cfg.paths.timestamp:
        model_dir = base_model_dir / cfg.paths.timestamp
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
        else:
            raise ValueError(f"Model directory does not exist: {base_model_dir}")

    tokenizer, model = load_model(model_dir)
    emotions = predict_emotion(text, tokenizer, model, labels)

    for label, prob in sorted(emotions.items(), key=lambda x: -x[1]):
        print(f"{label.title().ljust(12)}: {prob:.4f}")


if __name__ == "__main__":
    run_infer()
