from pathlib import Path
import polars as pl
from polyglot.text import Text


def split_into_sentences(raw_text):
    return [sent.string for sent in Text(raw_text).sentences]


df = pl.read_parquet("Data/caroldb.parquet")

df = df.with_columns(
    pl.col("text").map_elements(split_into_sentences, return_dtype=pl.List(pl.String))
)
df = df.filter(pl.col("text").list.len() > 0)
df = df.explode("text")

print(df)

Path('./Data').mkdir(exist_ok=True)
df.write_parquet("Data/carol-domain-sents.parquet")
