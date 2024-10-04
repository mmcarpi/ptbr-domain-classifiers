import polars as pl
from torch.utils.data import Dataset


def tokenize(dataset_path, tokenizer, tokenizer_args):
    dataset = (
        pl.read_parquet(dataset_path)
        .select(["text", pl.col("domain").to_physical()])
        .sample(fraction=1.0, shuffle=True, seed=42)
    )
    tokenized = tokenizer(dataset["text"].to_list(), **tokenizer_args)
    ids = tokenized.input_ids
    labels = dataset["domain"]
    mydataset = MyDataset(ids, labels)
    return mydataset


class MyDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ids = self.ids[idx]
        labels = self.labels[idx]
        return ids, labels

    def save(self, path):
        out = pl.DataFrame(dict(ids=self.ids, labels=self.labels))
        out.write_parquet(path)

    @classmethod
    def load(cls, path):
        data = pl.read_parquet(path)
        return cls(data["ids"].to_list(), data["labels"].to_list())


# def teste():
#     from transformers import AutoTokenizer

#     tokenizer = AutoTokenizer.from_pretrained(
#         "PORTULAN/albertina-1b5-portuguese-ptbr-encoder-256"
#     )
#     ids, labels = tokenize(
#         "Data/caroldb-train-sentences.parquet",
#         tokenizer=tokenizer,
#         tokenizer_args=dict(padding="max_length", truncation=True, max_length=512),
#     )

#     dataset = MyDataset(ids, labels)
#     return dataset
