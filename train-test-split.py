import polars as pl
from sklearn.model_selection import train_test_split

random_state = 42
min_sentence_length = 4
max_sentence_length = 256
hps_dataset_size = 100_000
df = pl.read_parquet("Data/caroldb-sentences.parquet")

domains = df["domain"].unique()

df = df.with_columns(pl.col("domain").cast(pl.Enum(domains)))
df = df.filter(
    (pl.col("text").str.len_chars() >= min_sentence_length)
    & (pl.col("text").str.len_chars() <= max_sentence_length)
)
df = df.unique()
number_of_texts = df.group_by("domain").len()["len"].min()

dfs = []
for domain in df["domain"].unique():
    dfs.append(
        df.filter(pl.col("domain") == domain).sample(
            number_of_texts, shuffle=True, seed=random_state
        )
    )

train, test = train_test_split(
    df, train_size=0.8, shuffle=True, random_state=random_state
)
hyper = train.sample(hps_dataset_size, shuffle=True)
hyper.write_parquet("Data/caroldb-hps-sentences.parquet")
train.write_parquet("Data/caroldb-train-sentences.parquet")
test.write_parquet("Data/caroldb-test-sentences.parquet")
