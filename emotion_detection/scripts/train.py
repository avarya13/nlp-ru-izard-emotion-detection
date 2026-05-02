import argparse
import lightning as L
from src.data.emotion_datamodule import EmotionDataModule
from src.models.bert_multilabel import BertMultiLabelClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/ru-izard-emotions")
    parser.add_argument("--model_name", default="cointegrated/rubert-tiny2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--save_path", default="./models/emotion_model")
    args = parser.parse_args()

    # DataModule
    dm = EmotionDataModule(
        data_dir=args.data_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    # Model
    model = BertMultiLabelClassifier(
        model_name=args.model_name,
        num_labels=10,
        learning_rate=args.learning_rate,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)

    # Saving
    trainer.save_checkpoint(f"{args.save_path}.ckpt")
    # Saving model and tokenizer in transformers format
    model.model.save_pretrained(args.save_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()