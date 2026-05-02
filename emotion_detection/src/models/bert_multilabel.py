import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import lightning as L
from torchmetrics import AUROC, F1Score

class BertMultiLabelClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = "cointegrated/rubert-tiny2",
        num_labels: int = 10,
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        scheduler_steps: int = -1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        # metrics
        self.train_auroc = AUROC(task="multilabel", num_labels=num_labels)
        self.val_auroc = AUROC(task="multilabel", num_labels=num_labels)
        self.test_auroc = AUROC(task="multilabel", num_labels=num_labels)
        self.val_f1 = F1Score(task="multilabel", num_labels=num_labels, average="macro")
        self.test_f1 = F1Score(task="multilabel", num_labels=num_labels, average="macro")

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        # logging
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        self.train_auroc(probs, batch["labels"].int())
        self.log("train_auroc", self.train_auroc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        self.val_auroc(probs, batch["labels"].int())
        self.val_f1(probs, batch["labels"].int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auroc", self.val_auroc, prog_bar=True)
        self.log("val_f1_macro", self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        self.test_auroc(probs, batch["labels"].int())
        self.test_f1(probs, batch["labels"].int())
        self.log("test_auroc", self.test_auroc, prog_bar=True)
        self.log("test_f1_macro", self.test_f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler_steps > 0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.scheduler_steps
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return optimizer