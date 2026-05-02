import hydra
import lightning as L
from omegaconf import DictConfig
from src.data.emotion_datamodule import EmotionDataModule
from src.models.multilabel_classifier import MultiLabelClassifier

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

    # Model
    model = MultiLabelClassifier(**cfg.model)

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