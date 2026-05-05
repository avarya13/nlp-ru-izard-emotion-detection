import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data.emotion_datamodule import EmotionDataModule
from torchmetrics import AUROC, F1Score, Precision, Recall

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    dm = EmotionDataModule(
        data_dir=cfg.data.data_dir,
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length,
        num_workers=cfg.data.num_workers,
    )
    dm.setup("test")
    
    model_path = f"{cfg.paths.save_dir}/{cfg.model.model_name.replace('/', '_')}"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    test_loader = dm.test_dataloader()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            probs = torch.sigmoid(outputs.logits)
            all_probs.append(probs.cpu())
            all_labels.append(batch["labels"].cpu())

    y_probs = torch.cat(all_probs)
    y_true = torch.cat(all_labels).int()

    num_labels = cfg.model.num_labels  
    auroc = AUROC(task="multilabel", num_labels=num_labels)
    f1_macro = F1Score(task="multilabel", num_labels=num_labels, average="macro")
    f1_micro = F1Score(task="multilabel", num_labels=num_labels, average="micro")
    precision_macro = Precision(task="multilabel", num_labels=num_labels, average="macro")
    recall_macro = Recall(task="multilabel", num_labels=num_labels, average="macro")

    auc_score = auroc(y_probs, y_true)
    f1_macro_score = f1_macro(y_probs, y_true)
    f1_micro_score = f1_micro(y_probs, y_true)
    precision_score = precision_macro(y_probs, y_true)
    recall_score = recall_macro(y_probs, y_true)

    print(f"Mean ROC-AUC: {auc_score:.4f}")
    print(f"Macro F1-score: {f1_macro_score:.4f}")
    print(f"Micro F1-score: {f1_micro_score:.4f}")
    print(f"Macro Precision: {precision_score:.4f}")
    print(f"Macro Recall: {recall_score:.4f}")

if __name__ == "__main__":
    main()