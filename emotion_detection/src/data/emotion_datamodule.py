import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import lightning as L

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

class EmotionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/ru-izard-emotions",
        model_name: str = "cointegrated/rubert-tiny2",
        batch_size: int = 32,
        max_length: int = 128,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        from datasets import load_from_disk
        dataset = load_from_disk(self.data_dir)
        self.train_df = dataset["train"].to_pandas()
        self.val_df = dataset["validation"].to_pandas()
        self.test_df = dataset["test"].to_pandas()

        emotion_cols = ['neutral', 'joy', 'sadness', 'anger', 'enthusiasm',
                        'surprise', 'disgust', 'fear', 'guilt', 'shame']
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.train_dataset = EmotionDataset(
            self.train_df['text'], self.train_df[emotion_cols].values,
            tokenizer, self.max_length
        )
        self.val_dataset = EmotionDataset(
            self.val_df['text'], self.val_df[emotion_cols].values,
            tokenizer, self.max_length
        )
        self.test_dataset = EmotionDataset(
            self.test_df['text'], self.test_df[emotion_cols].values,
            tokenizer, self.max_length
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)