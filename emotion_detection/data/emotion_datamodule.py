import lightning as L
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


class PreTokenizedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v for k, v in self.encodings[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class EmotionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        model_name: str,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = load_from_disk(self.data_dir)
        self.train_df = dataset["train"].to_pandas()
        self.val_df = dataset["validation"].to_pandas()
        self.test_df = dataset["test"].to_pandas()

        emotion_cols = [
            "neutral",
            "joy",
            "sadness",
            "anger",
            "enthusiasm",
            "surprise",
            "disgust",
            "fear",
            "guilt",
            "shame",
        ]
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def tokenize_texts(texts):
            encodings = []
            for text in texts:
                encoding = tokenizer(
                    str(text), truncation=True, padding=False, return_tensors="pt"
                )

                encoding = {k: v.squeeze(0) for k, v in encoding.items()}
                encodings.append(encoding)
            return encodings

        train_encodings = tokenize_texts(self.train_df["text"])
        val_encodings = tokenize_texts(self.val_df["text"])
        test_encodings = tokenize_texts(self.test_df["text"])

        self.train_dataset = PreTokenizedDataset(
            train_encodings, self.train_df[emotion_cols].values
        )
        self.val_dataset = PreTokenizedDataset(
            val_encodings, self.val_df[emotion_cols].values
        )
        self.test_dataset = PreTokenizedDataset(
            test_encodings, self.test_df[emotion_cols].values
        )

    def train_dataloader(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        collate_fn = DataCollatorWithPadding(tokenizer)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        collate_fn = DataCollatorWithPadding(tokenizer)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        collate_fn = DataCollatorWithPadding(tokenizer)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )
