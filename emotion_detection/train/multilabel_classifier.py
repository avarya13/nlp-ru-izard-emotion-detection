from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from eval.metrics import compute_f1_macro, compute_f1_micro
from torchmetrics import F1Score, Precision, Recall
from torchmetrics.classification import MultilabelAUROC, MultilabelRankingLoss
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from .focal_loss import FocalLoss


class MultiLabelClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = "cointegrated/rubert-tiny2",
        num_labels: int = 10,
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        scheduler_steps: int = -1,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        scheduler_type: str = "none",
        warmup_ratio: float = 0.0,
        activation: str = "sigmoid",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="multi_label_classification"
        )

        # loss function
        if loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        self.train_f1_macro = F1Score(
            task="multilabel", num_labels=num_labels, average="macro"
        )
        self.train_f1_micro = F1Score(
            task="multilabel", num_labels=num_labels, average="micro"
        )
        self.train_precision_macro = Precision(
            task="multilabel", num_labels=num_labels, average="macro"
        )
        self.train_recall_macro = Recall(
            task="multilabel", num_labels=num_labels, average="macro"
        )

        self.train_auc = MultilabelAUROC(num_labels=num_labels, average="macro")

        self.train_rank_loss = MultilabelRankingLoss()

        self.val_f1_macro = F1Score(
            task="multilabel", num_labels=num_labels, average="macro"
        )
        self.val_f1_micro = F1Score(
            task="multilabel", num_labels=num_labels, average="micro"
        )
        self.val_precision_macro = Precision(
            task="multilabel", num_labels=num_labels, average="macro"
        )
        self.val_recall_macro = Recall(
            task="multilabel", num_labels=num_labels, average="macro"
        )

        self.val_auc = MultilabelAUROC(num_labels=num_labels, average="macro")

        self.val_rank_loss = MultilabelRankingLoss()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        logits = outputs.logits

        if self.hparams.activation == "softmax":
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits)

        self.train_f1_macro(probs, batch["labels"].int())
        self.train_f1_micro(probs, batch["labels"].int())
        self.train_precision_macro(probs, batch["labels"].int())
        self.train_recall_macro(probs, batch["labels"].int())
        self.train_auc(probs, batch["labels"].int())
        self.train_rank_loss(probs, batch["labels"].int())

        labels_np = batch["labels"].int().cpu().detach().numpy()
        probs_np = probs.cpu().detach().numpy()
        custom_train_f1_macro = compute_f1_macro(labels_np, probs_np)
        custom_train_f1_micro = compute_f1_micro(labels_np, probs_np)

        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(
            "train_f1_macro",
            self.train_f1_macro,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train_f1_micro",
            self.train_f1_micro,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "custom_train_f1_macro",
            custom_train_f1_macro,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "custom_train_f1_micro",
            custom_train_f1_micro,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train_precision_macro",
            self.train_precision_macro,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train_recall_macro",
            self.train_recall_macro,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log("train_auc", self.train_auc, on_epoch=True, prog_bar=True)
        self.log("train_rank_loss", self.train_rank_loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        logits = outputs.logits
        loss = outputs.loss

        if self.hparams.activation == "softmax":
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits)

        self.val_f1_macro(probs, batch["labels"].int())
        self.val_f1_micro(probs, batch["labels"].int())
        self.val_precision_macro(probs, batch["labels"].int())
        self.val_recall_macro(probs, batch["labels"].int())
        self.val_auc(probs, batch["labels"].int())
        self.val_rank_loss(probs, batch["labels"].int())

        labels_np = batch["labels"].int().cpu().detach().numpy()
        probs_np = probs.cpu().detach().numpy()
        custom_val_f1_macro = compute_f1_macro(labels_np, probs_np)
        custom_val_f1_micro = compute_f1_micro(labels_np, probs_np)

        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(
            "val_f1_macro",
            self.val_f1_macro,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "val_f1_micro",
            self.val_f1_micro,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "val_precision_macro",
            self.val_precision_macro,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "val_recall_macro",
            self.val_recall_macro,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "custom_val_f1_macro",
            custom_val_f1_macro,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "custom_val_f1_micro",
            custom_val_f1_micro,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log("val_auc", self.val_auc, on_epoch=True, prog_bar=True)
        self.log("val_rank_loss", self.val_rank_loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler_type = self.hparams.scheduler_type

        print(self.hparams)

        if scheduler_type == "exponential":
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 0.5**epoch
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

        elif scheduler_type == "linear_warmup":
            total_steps = (
                self.hparams.scheduler_steps
                if self.hparams.scheduler_steps > 0
                else self.trainer.estimated_stepping_batches
            )
            warmup_steps = int(self.hparams.warmup_ratio * total_steps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        else:
            return optimizer
