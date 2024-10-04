from collections import Counter
from time import time

import numpy as np
import polars as pl
from numpy.ma.core import argmax

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from tqdm import tqdm

import util


def extract_words(df, text, pattern=None):
    pattern = pattern or "\p{L}+"
    df = df.with_columns(pl.col(text).str.to_lowercase().str.extract_all(pattern))
    return df


def count_words(df, text):
    df = df.with_columns(count=pl.col(text).list.eval(pl.element().value_counts()))
    df = df.with_columns(
        pl.col("count").list.eval(pl.element().struct.rename_fields(["word", "count"]))
    )
    return df


def word_doc_to_id_doc(docs, id2word, vocab_size):
    docs = np.stack(
        [np.array([doc.get(id2word[i], 0) for i in range(vocab_size)]) for doc in docs]
    )
    return docs


def extract_features(id_docs, vocab_size):
    x = np.zeros((len(id_docs), vocab_size), dtype=np.int32)
    for i, doc in enumerate(id_docs):
        x[i, :] = np.array(doc)
    return x


def word_count_to_counter(df):
    docs = [
        Counter({w: c for w, c in d.struct.unnest().iter_rows()}) for d in df["count"]
    ]
    return docs


def prepare_training_data(df, text, label, max_domain_vocab_size=20_000):
    df = df.group_by(label).agg(pl.col(text).str.join(" "))
    df = extract_words(df, text)
    df = count_words(df, text)
    docs = word_count_to_counter(df)
    docs = [
        Counter({w: c for w, c in doc.most_common(max_domain_vocab_size)})
        for doc in docs
    ]
    vocab = sum(docs, Counter())
    id2word = {i: w for i, w in enumerate(sorted(vocab))}
    id_docs = word_doc_to_id_doc(docs, id2word, len(vocab))

    return df, id_docs, vocab, id2word


def batched_predict(df, text, id2word, vocab_size, batch_size=256):
    df = extract_words(df, text)
    df = count_words(df, text)
    docs = word_count_to_counter(df)
    # docs = word_doc_to_id_doc(docs, id2word, vocab_size)
    for batch in range(0, len(docs), batch_size):
        docs_batch = docs[batch : batch + batch_size]
        docs_batch = word_doc_to_id_doc(docs_batch, id2word, vocab_size)
        yield docs_batch


def main():
    seed = 42

    max_domain_vocab_size = 25_000
    num_test_splits = 10
    text = "text"
    label = "domain"

    t = time()

    df_train = pl.read_parquet("Data/caroldb-train-sentences.parquet")  # .sample(10000)
    df_test = pl.read_parquet(
        "Data/caroldb-test-sentences.parquet"
    )  # .sample(fraction=1.0, shuffle=True)#.sample(10000)
    print(f"{time() - t:.2f}", "read_parquet")

    df, x_train, vocab, id2word = prepare_training_data(
        df_train, text, label, max_domain_vocab_size
    )
    y_train = np.array(df[label].to_physical())
    labels = np.unique(y_train)

    print(f"{time() - t:.2f}", "prepare_training_data")
    print(f"{len(vocab)=}")

    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    print(f"{time() - t:.2f}", "clf.fit")

    df_test = df_test.sample(fraction=1.0, shuffle=True, seed=seed)
    batch_size = len(df_test) // num_test_splits

    print(f"{len(df_test)=} {batch_size=}")

    results = []
    compute_metrics = util.create_compute_metrics(len(labels), argmax_first=False)

    # let's rewrite this loop to split the test set and apply the batched compute in each batched

    for batch_start in range(0, len(df_test), batch_size):
        df_test_batched = df_test[batch_start : batch_start + batch_size]
        y_pred = []
        y_true = np.asarray(df_test_batched[label].to_physical())
        for x_test in tqdm(
            batched_predict(df_test_batched, text, id2word, len(vocab), 256)
        ):
            y_pred.append(clf.predict(x_test))

        y_pred = np.concatenate(y_pred)
        eval_pred = y_pred, y_true
        results.append(compute_metrics(eval_pred))

    # for batch_start in range(0, len(df_test), batch_size):
    #     test_batch = df_test[batch_start : batch_start + batch_size]
    #     y_true = np.array(test_batch[label].to_physical())
    #     y_pred = []
    #     for x_test_batch in tqdm(
    #         batched_predict(test_batch, text, id2word, len(vocab), 128)
    #     ):
    #         y_pred.append(clf.predict(x_test_batch))
    #
    #     y_pred = np.concatenate(y_pred)
    #
    #     eval_pred = y_pred, y_true
    #     results.append(compute_metrics(eval_pred))

    # y_true_oh = np.zeros((batch_size, len(labels)), dtype=np.int32)
    # y_true_oh[np.arange(y_true.size), y_true] = 1
    #
    # y_pred_oh = np.zeros((batch_size, len(labels)), dtype=np.int32)
    # y_pred_oh[np.arange(y_pred.size), y_pred] = 1

    # out["acc"].append(metrics.accuracy_score(y_true, y_pred))
    # out["precision_score_micro"].append(
    #     metrics.precision_score(y_true, y_pred, average="micro", labels=labels)
    # )
    # out["precision_score_macro"].append(
    #     metrics.precision_score(y_true, y_pred, average="macro", labels=labels)
    # )
    # breakpoint()
    # out["precision_score_samples"].append(
    #     metrics.precision_score(y_true, y_pred, average="samples", labels=labels)
    # )
    # out["recall_score_micro"].append(
    #     metrics.recall_score(y_true, y_pred, average="micro", labels=labels)
    # )
    # out["recall_score_macro"].append(
    #     metrics.recall_score(y_true, y_pred, average="macro", labels=labels)
    # )
    # out["recall_score_samples"].append(
    #     metrics.recall_score(y_true, y_pred, average="samples", labels=labels)
    # )
    # out["f1_score_micro"].append(
    #     metrics.f1_score(y_true, y_pred, average="micro", labels=labels)
    # )
    # out["f1_score_macro"].append(
    #     metrics.f1_score(y_true, y_pred, average="macro", labels=labels)
    # )
    # out["f1_score_samples"].append(
    #     metrics.f1_score(y_true, y_pred, average="samples", labels=labels)
    # )
    # out["cm"].append(
    #     metrics.confusion_matrix(y_true, y_pred, labels=labels).flatten()
    # )

    out = pl.DataFrame(results)
    print(out)
    out.write_parquet("results/nb.parquet")
    print(f"Done in {time()-t:.2f} seconds")


if __name__ == "__main__":
    main()
