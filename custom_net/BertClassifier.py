import argparse
import logging

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_cosine_schedule_with_warmup
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.utilities import rank_zero_only

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]["text"]
        label = self.data.iloc[index]["label"]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encoding["input_ids"], encoding["attention_mask"], label


class TextClassificationModel(LightningModule):
    def __init__(self, num_classes, learning_rate, warmup_steps, max_epochs, max_length):
        super().__init__()
        self.save_hyperparameters()
        self.bert, self.vocab = get_pytorch_kobert_model()
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("train_loss", loss)
        self.log("train_acc", accuracy(logits, labels))
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("val_loss", loss)
        self.log("val_acc", accuracy(logits, labels))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.max_epochs * len(self.train_dataloader()),
        )
        return [optimizer], [scheduler]


def main(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    train_data = pd.read_csv(args.train_data_path)
    val_data = pd.read_csv(args.val_data_path)
    test_data = pd.read_csv(args.test_data_path)

    encoder = LabelEncoder()
    encoder.fit(train_data["class"])
    train_data["label"] = encoder.transform(train_data["class"])
    val_data["label"] = encoder.transform(val_data["class"])
    test_data["label"] = [0 for _ in range(len(test_data))]

    tokenizer = get_tokenizer()
    train_dataset = TextClassificationDataset(train_data, tokenizer, args.max_length)
    val_dataset = TextClassificationDataset(val_data, tokenizer, args.max_length)
    test_dataset = TextClassificationDataset(test_data, tokenizer, args.max_length)

    model = TextClassificationModel(
        num_classes=len(encoder.classes_),
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        max_length=args.max_length,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    logger_callback = TensorBoardLogger(save_dir=args.log_dir, name="logs")

    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger_callback,
    )

    trainer.fit(model, train_dataset, val_dataset)

    trainer.test(model, test_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="./train.csv")
    parser.add_argument("--val_data_path", type=str, default="./val.csv")
    parser.add_argument("--test_data_path", type=str, default="./test.csv")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=20)
    args = parser.parse_args()

    main(args)
