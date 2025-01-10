import json

import numpy as np
from sklearn import metrics


def to_one_hot(y, num_labels):
    y_onehot = np.zeros((len(y), num_labels), dtype=np.int32)
    y_onehot[np.arange(y.size), y] = 1
    return y_onehot


def create_compute_metrics(num_labels, argmax_first=False):
    def compute_metrics(predictions, labels):
        if argmax_first:
            predictions = predictions.argmax(axis=1)

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


def save_metrics(metrics, file_name):
    with open(file_name, "w") as metric_file:
        json.dump(metrics, metric_file, indent=True)
