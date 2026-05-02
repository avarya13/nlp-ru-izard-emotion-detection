import os
import hydra
import lightning as L
from transformers import AutoTokenizer
from omegaconf import DictConfig
from src.data.emotion_datamodule import EmotionDataModule
from src.models.multilabel_classifier import MultiLabelClassifier
import mlflow.pytorch


mlflow.pytorch.autolog(log_every_n_epoch=1, log_models=False)


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

    logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name="emotion_classification",
        tracking_uri=cfg.paths.mlflow_dir,
        tags={
            "project": "ru_izard_emotion_detection",
            "user": os.getenv("USER", "unknown"),
        },
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
        logger=logger,
    )
    trainer.fit(model, dm)

    # Saving model and tokenizer in transformers format
    save_path = f"{cfg.paths.save_dir}/{cfg.model.model_name.replace('/', '_')}"
    model.model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

    with mlflow.start_run(run_id=logger.run_id):
        mlflow.transformers.log_model(
            transformers_model={
                "model": model.model,
                "tokenizer": tokenizer,
            },  # tokenizer нужно сохранить в dm
            artifact_path="model",
            registered_model_name=f"{cfg.model.model_name.replace('/', '_')}",
        )


if __name__ == "__main__":
    main()
