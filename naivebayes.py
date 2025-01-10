import json

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
        token_pattern=r"\b[a-z]+\b",
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

    metrics = compute_metrics(y_pred, y_true)

    return metrics


if __name__ == "__main__":
    dataset = load_dataset("mmcarpi/caroldb-sentences", split=["train", "test"])

    classifier, vectorizer, encoder = train_naive_bayes_tfidf(
        dataset[0], "text", "domain"
    )

    compute_metrics = util.create_compute_metrics(5, argmax_first=False)
    metrics = evaluate_naive_bayes_tfidf(
        classifier,
        vectorizer,
        encoder,
        compute_metrics,
        dataset[1],
        "text",
        "domain",
    )

    with open("results/cnnb.json", "w") as metric_file:
        json.dump(metrics, metric_file, indent=True)
