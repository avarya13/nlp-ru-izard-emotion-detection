import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data.emotion_datamodule import EmotionDataModule
from torchmetrics import AUROC, F1Score, Precision, Recall
from torchmetrics.classification import MultilabelRankingLoss
from src.utils.metrics import compute_f1_macro, compute_f1_micro

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
    lrl = MultilabelRankingLoss(num_labels=num_labels)

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
    print(f"Label Ranking Loss: {lrl:.4f}")

    from sklearn.metrics import f1_score
    y_pred_bin = (y_probs > 0.5).int().cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    y_probs_np = y_probs.cpu().numpy()

    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score 

    sk_auc = roc_auc_score(y_true_np, y_probs_np, average='macro')
    sk_f1_macro = f1_score(y_true_np, y_pred_bin, average='macro')
    sk_f1_micro = f1_score(y_true_np, y_pred_bin, average='micro')
    sk_precision = precision_score(y_true_np, y_pred_bin, average='macro')
    sk_recall = recall_score(y_true_np, y_pred_bin, average='macro')

    print("\n--- sklearn metrics ---")
    print(f"Macro ROC-AUC: {sk_auc:.4f}")
    print(f"Macro F1-score: {sk_f1_macro:.4f}")
    print(f"Micro F1-score: {sk_f1_micro:.4f}")
    print(f"Macro Precision: {sk_precision:.4f}")
    print(f"Macro Recall: {sk_recall:.4f}")

    # labels_np = batch["labels"].int().cpu().detach().numpy()
    # probs_np = probs.cpu().detach().numpy()
    test_f1_macro = compute_f1_macro(y_true_np, y_probs_np)
    test_f1_micro = compute_f1_micro(y_true_np, y_probs_np)

    print(f'Custom F1-micro: {test_f1_micro}')
    print(f'Custom F1-macro: {test_f1_macro}')

if __name__ == "__main__":
    main()