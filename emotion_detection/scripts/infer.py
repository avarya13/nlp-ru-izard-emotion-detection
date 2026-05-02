import hydra
import lightning as L
import torch
from src.data.emotion_datamodule import EmotionDataModule
from src.models.multilabel_classifier import MultiLabelClassifier
from src.utils.metrics import compute_auc, compute_f1_macro
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)

    # DataModule
    dm = EmotionDataModule(
        data_dir=cfg.data.data_dir,
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length,
        num_workers=cfg.data.num_workers,
    )
    dm.setup("test")

    # Model
    model = MultiLabelClassifier.load_from_checkpoint(
        f"{cfg.paths.save_dir}/{cfg.model.model_name.replace('/', '_')}.ckpt",
        model_name=cfg.model.model_name,
        num_labels=cfg.data.num_labels,
    )
    model.eval()

    # Collect predictions on test
    test_loader = dm.test_dataloader()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

            probs = torch.sigmoid(outputs.logits)
            all_probs.append(probs.cpu())
            all_labels.append(batch["labels"].cpu())

    y_true = torch.cat(all_labels).numpy()
    y_probs = torch.cat(all_probs).numpy()
    aucs = compute_auc(y_true, y_probs)
    f1_macro = compute_f1_macro(y_true, y_probs)
    print(f"Test ROC-AUC per class: {aucs}")
    print(f"Mean ROC-AUC: {aucs.mean():.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")


if __name__ == "__main__":
    main()
