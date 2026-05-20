import os
from datetime import datetime

import hydra
import lightning as L

# import mlflow.pytorch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig
from src.data.emotion_datamodule import EmotionDataModule
from src.models.multilabel_classifier import MultiLabelClassifier
from src.utils.dvc_pull import dvc_pull
from transformers import AutoTokenizer

# mlflow.pytorch.autolog(log_every_n_epoch=1, log_models=False)
# TODO: fix mlflow formatting


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")

    dvc_pull()

    # DataModule
    dm = EmotionDataModule(
        data_dir=cfg.data.data_dir,
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        # max_length=cfg.data.max_length,
        num_workers=cfg.data.num_workers,
    )

    logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name="emotion_classification",
        tracking_uri=cfg.paths.mlflow_dir,
        tags={
            "project": f"ru_izard_{cfg.model.model_name}",
            "user": os.getenv("USER", "unknown"),
        },
    )

    # Model
    model_params = {k: v for k, v in cfg.model.items() if k != "epochs"}
    model = MultiLabelClassifier(**model_params)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_monitor = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        filename="best-{epoch}-{val_auroc:.2f}",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    trainer = L.Trainer(
        max_epochs=cfg.model.epochs,
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        # gradient_clip_val=cfg.train.gradient_clip_val,
        # accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        log_every_n_steps=cfg.train.log_every_n_steps,
        default_root_dir=cfg.paths.save_dir,
        logger=logger,
        callbacks=[lr_monitor, ckpt_monitor],
    )
    trainer.fit(model, dm)

    # Saving model and tokenizer in transformers format
    save_path = (
        f"{cfg.paths.save_dir}/{cfg.model.model_name.replace('/', '_')}/{timestamp}"
    )
    model.model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
