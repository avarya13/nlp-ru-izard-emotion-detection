import os
import subprocess
from datetime import datetime
from pathlib import Path

import lightning as L
import mlflow.pytorch
from eval.plot_metrics import save_all_plots
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig
from transformers import AutoTokenizer
from utils.dvc_pull import dvc_pull

from data.emotion_datamodule import EmotionDataModule

from .multilabel_classifier import MultiLabelClassifier


def run_train(cfg: DictConfig):
    L.seed_everything(cfg.seed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")

    dvc_pull(remote=cfg.data.remote_name, target=cfg.data.dvc_target)

    dm = EmotionDataModule(
        data_dir=cfg.data.data_dir,
        model_name=cfg.model.model_name,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name="emotion_detection",
        tracking_uri=cfg.paths.mlflow_uri,
        tags={
            "project": f"ru_izard_{cfg.model.model_name}",
            "user": os.getenv("USER", "unknown"),
        },
    )
    csv_logger = CSVLogger(save_dir="logs", name="train_logs")

    commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    logger.log_hyperparams(
        {
            "git_commit": commit_id,
            "model_name": cfg.model.model_name,
            "batch_size": cfg.data.batch_size,
            "learning_rate": cfg.model.learning_rate,
            "epochs": cfg.model.epochs,
        }
    )

    exclude_keys = {"epochs", "triton_name", "activation", "triton_url"}
    model_params = {k: v for k, v in cfg.model.items() if k not in exclude_keys}
    model = MultiLabelClassifier(**model_params)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_monitor = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        filename="best-{epoch}-{val_f1_macro:.2f}",
        monitor="val_f1_macro",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    trainer = L.Trainer(
        max_epochs=cfg.model.epochs,
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        log_every_n_steps=cfg.train.log_every_n_steps,
        default_root_dir=cfg.paths.save_dir,
        logger=[logger, csv_logger],
        callbacks=[lr_monitor, ckpt_monitor],
    )
    trainer.fit(model, dm)

    csv_path = Path(csv_logger.log_dir) / "metrics.csv"
    if csv_path.exists():
        save_all_plots(csv_path, Path("plots"), cfg.model.model_name, timestamp)

    save_path = (
        f"{cfg.paths.save_dir}/{cfg.model.model_name.replace('/', '-')}/{timestamp}"
    )
    model.model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    from utils.hydra_utils import compose_cfg

    cfg = compose_cfg("config")
    mlflow.pytorch.autolog(log_every_n_epoch=1, log_models=False)
    run_train(cfg)
