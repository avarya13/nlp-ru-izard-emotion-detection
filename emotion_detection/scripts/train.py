import hydra
import lightning as L
from omegaconf import DictConfig
from src.data.emotion_datamodule import EmotionDataModule
from src.models.multilabel_classifier import MultiLabelClassifier

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", default="./data/ru-izard-emotions")
    # parser.add_argument("--model_name", default="cointegrated/rubert-tiny2")
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--max_length", type=int, default=128)
    # parser.add_argument("--learning_rate", type=float, default=5e-5)
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--num_workers", type=int, default=0)
    # parser.add_argument("--accelerator", default="auto")
    # parser.add_argument("--save_path", default="./models/emotion_model")
    # args = parser.parse_args()


    L.seed_everything(cfg.seed)

    # DataModule
    dm = EmotionDataModule(
        data_dir=cfg.data.data_dir,
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        max_length=cfg.data.max_length,
        num_workers=cfg.data.num_workers,
    )

    # Model
    model = MultiLabelClassifier(**cfg.model)

    # Trainer
    # trainer = L.Trainer(
    #     max_epochs=args.epochs,
    #     accelerator=args.accelerator,
    #     log_every_n_steps=10,
    # )
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        log_every_n_steps=cfg.train.log_every_n_steps,
        default_root_dir=cfg.paths.save_dir,
    )
    trainer.fit(model, dm)

    # Saving model and tokenizer in transformers format
    save_path = f"{cfg.paths.save_dir}/{cfg.model.model_name.replace('/', '_')}"
    model.model.save_pretrained(save_path)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()