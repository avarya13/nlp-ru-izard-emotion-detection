import json
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import DictConfig
from torchmetrics import F1Score, Precision, Recall
from torchmetrics.classification import MultilabelAUROC, MultilabelRankingLoss
from transformers import AutoModelForSequenceClassification
from utils.dvc_pull import dvc_pull

from data.emotion_datamodule import EmotionDataModule

from .metrics import compute_f1_macro, compute_f1_micro


def run_eval(cfg: DictConfig):
    dvc_pull(remote=cfg.data.remote_name, target=cfg.data.dvc_target)

    dm = EmotionDataModule(
        data_dir=cfg.data.data_dir,
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    dm.setup("test")

    base_path = Path(cfg.paths.save_dir) / cfg.model.model_name.replace("/", "-")

    timestamp = cfg.paths.timestamp

    if timestamp:
        model_path = base_path / timestamp
        print(f"Loading model from specified timestamp: {model_path}")

    else:
        if base_path.exists():
            subdirs = [
                d
                for d in base_path.iterdir()
                if d.is_dir() and len(d.name) == 13 and d.name[8] == "-"
            ]

            if subdirs:
                latest = max(subdirs, key=lambda d: d.name)
                model_path = latest
                print(f"Using latest timestamped model: {model_path}")

            else:
                model_path = base_path
                print(f"No timestamped subdir found, using base path: {model_path}")

        else:
            model_path = base_path
            print(f"Base path does not exist: {model_path}")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.eval()

    test_loader = dm.test_dataloader()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits = outputs.logits

            if cfg.model.activation == "softmax":
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(batch["labels"].cpu())

    y_probs = torch.cat(all_probs)
    y_true = torch.cat(all_labels).int()

    num_labels = cfg.model.num_labels

    f1_macro = F1Score(
        task="multilabel",
        num_labels=num_labels,
        average="macro",
    )

    f1_micro = F1Score(
        task="multilabel",
        num_labels=num_labels,
        average="micro",
    )

    precision_macro = Precision(
        task="multilabel",
        num_labels=num_labels,
        average="macro",
    )

    recall_macro = Recall(
        task="multilabel",
        num_labels=num_labels,
        average="macro",
    )
    auc = MultilabelAUROC(num_labels=num_labels, average="macro")
    rank_loss = MultilabelRankingLoss()

    f1_macro_score = f1_macro(y_probs, y_true)
    f1_micro_score = f1_micro(y_probs, y_true)

    precision_score = precision_macro(y_probs, y_true)
    recall_score = recall_macro(y_probs, y_true)

    y_true_np = y_true.cpu().numpy()
    y_probs_np = y_probs.cpu().numpy()

    custom_f1_macro = compute_f1_macro(y_true_np, y_probs_np)
    custom_f1_micro = compute_f1_micro(y_true_np, y_probs_np)

    auc_score = auc(y_probs, y_true.int())
    ranking_loss = rank_loss(y_probs, y_true.int())

    print(f"Macro F1-score: {f1_macro_score:.4f}")
    print(f"Micro F1-score: {f1_micro_score:.4f}")
    print(f"Macro Precision: {precision_score:.4f}")
    print(f"Macro Recall: {recall_score:.4f}")

    print(f"Custom F1-macro: {custom_f1_macro:.4f}")
    print(f"Custom F1-micro: {custom_f1_micro:.4f}")

    print(f"ROC-AUC: {auc_score:.4f}")
    print(f"Ranking loss: {ranking_loss:.4f}")

    inference_timestamp = datetime.now().strftime("%Y%m%d-%H%M")

    results_dir = (
        Path(cfg.paths.inference_dir)
        / cfg.model.model_name.replace("/", "-")
        / inference_timestamp
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_path": str(model_path),
        "activation": cfg.model.activation,
        "macro_f1": float(f1_macro_score),
        "micro_f1": float(f1_micro_score),
        "macro_precision": float(precision_score),
        "macro_recall": float(recall_score),
        "custom_macro_f1": float(custom_f1_macro),
        "custom_micro_f1": float(custom_f1_micro),
        "roc_auc": float(auc_score),
        "ranking_loss": float(ranking_loss),
    }

    results_path = results_dir / "metrics.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    run_eval()
