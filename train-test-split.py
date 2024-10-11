import polars as pl
from sklearn.model_selection import train_test_split

random_state = 42
min_sentence_length = 4
max_sentence_length = 256
hps_dataset_fraction = 0.01
df = pl.read_parquet("Data/caroldb-sentences.parquet")

domains = df["domain"].unique()

df = df.with_columns(pl.col("domain").cast(pl.Enum(domains)))
df = df.filter(
    (pl.col("text").str.len_chars() >= min_sentence_length)
    & (pl.col("text").str.len_chars() <= max_sentence_length)
)
df = df.unique()
number_of_texts = df.group_by("domain").len()["len"].min()

df_balanced = pl.DataFrame(schema=df.schema)
for domain in df["domain"].unique():
    df_balanced.vstack(
        df.filter(pl.col("domain") == domain).sample(
            number_of_texts, shuffle=True, seed=random_state
        ),
        in_place = True
    )

train, test = train_test_split(
    df_balanced, train_size=0.8, shuffle=True, random_state=random_state
)
hyper = train.sample(fraction=hps_dataset_fraction, shuffle=True)
hyper.write_parquet("Data/caroldb-hps-sentences.parquet")
train.write_parquet("Data/caroldb-train-sentences.parquet")
test.write_parquet("Data/caroldb-test-sentences.parquet")
