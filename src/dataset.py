"""PyTorch Dataset that turns sample-level rows into tokenized model inputs."""

import pandas as pd
from torch.utils.data import Dataset


class MedHalluDataset(Dataset):
    """One item = one (question, answer) pair with label 1 = hallucinated, 0 = ground truth.

    Items are NOT padded here; padding is done per batch by DataCollatorWithPadding,
    so each batch is only as long as its longest sequence (dynamic padding).
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256, use_knowledge: bool = False):
        if use_knowledge:
            self.first = (df["question"] + " " + df["knowledge"]).tolist()
        else:
            self.first = df["question"].tolist()
        self.second = df["answer"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # Sentence-pair encoding: [CLS] first [SEP] answer [SEP]
        enc = self.tokenizer(
            self.first[idx],
            self.second[idx],
            truncation=True,          # cut the longer segment first if over max_length
            max_length=self.max_length,
        )
        item = dict(enc)              # input_ids, token_type_ids, attention_mask
        item["labels"] = self.labels[idx]
        return item
