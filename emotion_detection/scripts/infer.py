from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from src.data.emotion_datamodule import EmotionDataModule
from src.utils.dvc_pull import dvc_pull
from src.utils.metrics import compute_f1_macro, compute_f1_micro
from torchmetrics import F1Score, Precision, Recall
from transformers import AutoModelForSequenceClassification


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    dvc_pull()
    dm = EmotionDataModule(
        data_dir=cfg.data.data_dir,
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    dm.setup("test")

    base_path = f"{cfg.paths.save_dir}/{cfg.model.model_name.replace('/', '_')}"
    timestamp = cfg.paths.timestamp
    if timestamp:
        model_path = f"{base_path}/{timestamp}"
        print(f"Loading model from specified timestamp: {model_path}")
    else:
        base = Path(base_path)
        if base.exists():
            subdirs = [
                d
                for d in base.iterdir()
                if d.is_dir() and len(d.name) == 13 and d.name[8] == "-"
            ]
            if subdirs:
                latest = max(subdirs, key=lambda d: d.name)
                model_path = str(latest)
                print(f"Using latest timestamped model: {model_path}")
            else:
                model_path = base_path
                print(f"No timestamped subdir found, using base path: {model_path}")
        else:
            model_path = base_path
            print(f"Base path does not exist: {base_path}")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    test_loader = dm.test_dataloader()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            # probs = torch.sigmoid(outputs.logits)
            probs = torch.softmax(outputs.logits, -1)
            all_probs.append(probs.cpu())
            all_labels.append(batch["labels"].cpu())

    y_probs = torch.cat(all_probs)
    y_true = torch.cat(all_labels).int()

    num_labels = cfg.model.num_labels
    f1_macro = F1Score(task="multilabel", num_labels=num_labels, average="macro")
    f1_micro = F1Score(task="multilabel", num_labels=num_labels, average="micro")
    precision_macro = Precision(
        task="multilabel", num_labels=num_labels, average="macro"
    )
    recall_macro = Recall(task="multilabel", num_labels=num_labels, average="macro")

    print(f"probs: {y_probs}")

    f1_macro_score = f1_macro(y_probs, y_true)
    f1_micro_score = f1_micro(y_probs, y_true)
    precision_score = precision_macro(y_probs, y_true)
    recall_score = recall_macro(y_probs, y_true)

    print(f"Macro F1-score: {f1_macro_score:.4f}")
    print(f"Micro F1-score: {f1_micro_score:.4f}")
    print(f"Macro Precision: {precision_score:.4f}")
    print(f"Macro Recall: {recall_score:.4f}")

    y_true_np = y_true.cpu().numpy()
    y_probs_np = y_probs.cpu().numpy()

    test_f1_macro = compute_f1_macro(y_true_np, y_probs_np)
    test_f1_micro = compute_f1_micro(y_true_np, y_probs_np)

    print(f"Custom F1-micro: {test_f1_micro}")
    print(f"Custom F1-macro: {test_f1_macro}")


if __name__ == "__main__":
    main()
