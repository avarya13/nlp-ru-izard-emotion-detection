import argparse
import lightning as L
import torch
from src.data.emotion_datamodule import EmotionDataModule
from src.models.bert_multilabel import BertMultiLabelClassifier
from src.utils.metrics import compute_auc, compute_f1_macro

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/ru-izard-emotions")
    parser.add_argument("--model_path", default="./models/emotion_model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    # DataModule
    dm = EmotionDataModule(
        data_dir=args.data_dir,
        model_name=args.model_path,  # load custom tokenizer
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    dm.setup("test")

    # Model
    model = BertMultiLabelClassifier.load_from_checkpoint(
        f"{args.model_path}.ckpt",
        model_name=args.model_path,
        num_labels=10,
    )
    model.eval()

    # Collect predictions on test
    test_loader = dm.test_dataloader()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            probs = torch.sigmoid(outputs.logits)
            all_probs.append(probs.cpu())
            all_labels.append(batch["labels"].cpu())
    y_probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()

    aucs = compute_auc(y_true, y_probs)
    f1_macro = compute_f1_macro(y_true, y_probs)
    print(f"Test ROC-AUC per class: {aucs}")
    print(f"Mean ROC-AUC: {aucs.mean():.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")

if __name__ == "__main__":
    main()