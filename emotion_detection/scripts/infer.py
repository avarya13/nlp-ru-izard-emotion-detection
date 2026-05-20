from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from src.data.emotion_datamodule import EmotionDataModule
from src.utils.dvc_pull import dvc_pull
from src.utils.metrics import compute_f1_macro, compute_f1_micro
from torchmetrics import AUROC, F1Score, Precision, Recall
from torchmetrics.classification import MultilabelRankingLoss
from transformers import AutoModelForSequenceClassification


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    dvc_pull()
    dm = EmotionDataModule(
        data_dir=cfg.data.data_dir,
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        # max_length=cfg.data.max_length,
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

    # model_path = f"{cfg.paths.save_dir}/{cfg.model.model_name.replace('/', '_')}"
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
    auroc = AUROC(task="multilabel", num_labels=num_labels)
    f1_macro = F1Score(task="multilabel", num_labels=num_labels, average="macro")
    f1_micro = F1Score(task="multilabel", num_labels=num_labels, average="micro")
    precision_macro = Precision(
        task="multilabel", num_labels=num_labels, average="macro"
    )
    recall_macro = Recall(task="multilabel", num_labels=num_labels, average="macro")
    ranking_loss = MultilabelRankingLoss(num_labels=num_labels)

    print(f"probs: {y_probs}")

    auc_score = auroc(y_probs, y_true)
    f1_macro_score = f1_macro(y_probs, y_true)
    f1_micro_score = f1_micro(y_probs, y_true)
    precision_score = precision_macro(y_probs, y_true)
    recall_score = recall_macro(y_probs, y_true)
    ranking_loss.update(y_probs, y_true)
    lrl = ranking_loss.compute().item()

    print(f"Mean ROC-AUC: {auc_score:.4f}")
    print(f"Macro F1-score: {f1_macro_score:.4f}")
    print(f"Micro F1-score: {f1_micro_score:.4f}")
    print(f"Macro Precision: {precision_score:.4f}")
    print(f"Macro Recall: {recall_score:.4f}")
    print(f"Label Ranking Loss: {lrl:.4f}")

    # y_pred_bin = (y_probs > 0.5).int().cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    y_probs_np = y_probs.cpu().numpy()

    # labels_np = batch["labels"].int().cpu().detach().numpy()
    # probs_np = probs.cpu().detach().numpy()
    test_f1_macro = compute_f1_macro(y_true_np, y_probs_np)
    test_f1_micro = compute_f1_micro(y_true_np, y_probs_np)

    print(f"Custom F1-micro: {test_f1_micro}")
    print(f"Custom F1-macro: {test_f1_macro}")


if __name__ == "__main__":
    main()
