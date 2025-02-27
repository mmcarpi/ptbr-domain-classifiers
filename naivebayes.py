import pickle
from pathlib import Path

import polars as pl
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

import util


def train_naive_bayes_tfidf(dataset, text_column, label_column):
    df = dataset.to_polars()
    x = df[text_column]
    y = df[label_column]

    vectorizer = TfidfVectorizer(
        max_features=30_000,
        analyzer="word",
        token_pattern=r"\b[a-zà-ú]+\b",
        lowercase=True,
        ngram_range=(1, 1),
        min_df=5,
        max_df=0.95,
    )
    x_tfidf = vectorizer.fit_transform(x)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    classifier = MultinomialNB()
    classifier.fit(x_tfidf, y)

    return classifier, vectorizer, encoder


def evaluate_naive_bayes_tfidf(
    classifier, vectorizer, encoder, compute_metrics, dataset, text_column, label_column
):
    df = dataset.to_polars()
    x_test = df[text_column]
    y_true = encoder.transform(df[label_column])

    x_test_tfidf = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test_tfidf)

    return pl.DataFrame(dict(y_pred=y_pred, y_true=y_true))


if __name__ == "__main__":
    dataset = load_dataset("carolina-c4ai/carol-domain-sents", split=["train", "test"])

    classifier, vectorizer, encoder = train_naive_bayes_tfidf(
        dataset[0], "text", "domain"
    )

    compute_metrics = util.create_compute_metrics(5, argmax_first=False)
    result = evaluate_naive_bayes_tfidf(
        classifier,
        vectorizer,
        encoder,
        compute_metrics,
        dataset[1],
        "text",
        "domain",
    )
    rootpath = Path("Models/baseline/naive-bayes")
    rootpath.mkdir(parents=True, exist_ok=True)
    result.write_csv(rootpath / "eval.csv")

    with open(rootpath / "classifier.pkl", "wb") as classifier_file:
        pickle.dump(classifier, classifier_file)

    with open(rootpath / "vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    with open(rootpath / "encoder.pkl", "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)
