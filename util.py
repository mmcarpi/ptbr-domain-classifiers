import numpy as np
from datasets import load_dataset
from sklearn import metrics


def load_caroldb(seed=42):
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": "Data/caroldb-train-sentences.parquet",
            "test": "Data/caroldb-test-sentences.parquet",
        },
    )

    dataset_tmp = dataset["train"].train_test_split(
        train_size=0.9, shuffle=True, seed=seed
    )
    dataset["train"] = dataset_tmp["train"]
    dataset["eval"] = dataset_tmp["test"]
    dataset["test"] = dataset["test"]
    dataset = dataset.rename_column("domain", "label")
    dataset = dataset.remove_columns(["source_typology", "carolina_typology"])

    return dataset


def unique_labels(dataset, labels="labels"):
    labels = set.union(*[set(v[labels]) for v in dataset.values()])
    return labels


def create_label2id(labels):
    label2id = {l: i for i, l in enumerate(labels)}
    return label2id


def create_preprocess_function(tokenizer, label2id):
    def preprocess_function(examples):
        tokens = tokenizer(examples["text"], truncation=True, padding=True)
        tokens["labels"] = [label2id[label] for label in examples["label"]]
        return tokens

    return preprocess_function


def to_one_hot(y, num_labels):
    y_onehot = np.zeros((len(y), num_labels), dtype=np.int32)
    y_onehot[np.arange(y.size), y] = 1
    return y_onehot


def create_compute_metrics(num_labels, argmax_first=False):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if argmax_first:
            predictions = np.argmax(predictions, axis=1)

        result = dict()
        result["cm"] = (
            metrics.confusion_matrix(y_true=labels, y_pred=predictions)
            .flatten()
            .tolist()
        )
        labels = to_one_hot(labels, num_labels)
        predictions = to_one_hot(predictions, num_labels)
        result["accuracy"] = metrics.accuracy_score(y_true=labels, y_pred=predictions)

        result["precision"] = metrics.precision_score(
            y_true=labels, y_pred=predictions, average=None
        ).tolist()
        result["precision_micro"] = metrics.precision_score(
            y_true=labels, y_pred=predictions, average="micro"
        )
        result["precision_macro"] = metrics.precision_score(
            y_true=labels, y_pred=predictions, average="macro"
        )
        result["precision_samples"] = metrics.precision_score(
            y_true=labels, y_pred=predictions, average="samples"
        )

        result["recall"] = metrics.recall_score(
            y_true=labels, y_pred=predictions, average=None
        ).tolist()
        result["recall_micro"] = metrics.recall_score(
            y_true=labels, y_pred=predictions, average="micro"
        )
        result["recall_macro"] = metrics.recall_score(
            y_true=labels, y_pred=predictions, average="macro"
        )
        result["recall_samples"] = metrics.recall_score(
            y_true=labels, y_pred=predictions, average="samples"
        )

        result["f1"] = metrics.f1_score(
            y_true=labels, y_pred=predictions, average=None
        ).tolist()
        result["f1_micro"] = metrics.f1_score(
            y_true=labels, y_pred=predictions, average="micro"
        )
        result["f1_macro"] = metrics.f1_score(
            y_true=labels, y_pred=predictions, average="macro"
        )
        result["f1_samples"] = metrics.f1_score(
            y_true=labels, y_pred=predictions, average="samples"
        )

        return result

    return compute_metrics
