import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import polars as pl
import optuna
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


NUM_JOBS = 4

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_dataset_splits():
    dataset = load_dataset("carolina-c4ai/carol-domain-sents")
    train = dataset["train"].to_polars()

    hps = dataset["hps"].train_test_split(test_size=0.1)
    hps_train = hps["train"].to_polars()
    hps_test = hps["test"].to_polars()
    test = dataset["test"].to_polars()

    X_train, y_train = train["text"], train["domain"]
    X_test, y_test = test["text"], test["domain"]
    X_hps_train, y_hps_train = hps_train["text"], hps_train["domain"]
    X_hps_test, y_hps_test = hps_test["text"], hps_test["domain"]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    y_hps_train = encoder.transform(y_hps_train)
    y_hps_test = encoder.transform(y_hps_test)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_hps_train,
        y_hps_train,
        X_hps_test,
        y_hps_test,
        encoder,
    )


def create_objective(
    model_name, X_hps_train, y_hps_train, X_hps_test, y_hps_test, seed
):
    """
    Creates the objective function for Optuna to optimize.
    """

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        Defines the hyperparameter search space, creates a pipeline,
        trains it, and evaluates it on the validation set.
        """
        # --- Hyperparameter Search Space ---
        # 1. TfidfVectorizer hyperparameters
        tfidf_ngram_range = trial.suggest_categorical("tfidf_ngram_range", [1, 2, 3])
        tfidf_min_df = trial.suggest_int("tfidf_min_df", 2, 10, step=1)
        tfidf_max_df = trial.suggest_float("tfidf_max_df", 0.5, 0.95, step=0.05)

        # 2. Classifier hyperparameters
        if model_name == "svm":
            classifier_C = trial.suggest_float("svm_C", 1e-6, 1e4, log=True)
            classifier_obj = SGDClassifier(
                alpha=classifier_C, loss="hinge", random_state=seed, n_jobs=NUM_JOBS
            )
        elif model_name == "naive-bayes":
            classifier_alpha = trial.suggest_float("nb_alpha", 1e-2, 1.0, log=True)
            classifier_obj = MultinomialNB(alpha=classifier_alpha)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        pipeline = Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="word",
                        token_pattern=r"\b[a-zà-ú]+\b",
                        lowercase=True,
                        ngram_range=(1, tfidf_ngram_range),
                        min_df=tfidf_min_df,
                        max_df=tfidf_max_df,
                    ),
                ),
                ("classifier", classifier_obj),
            ]
        )

        pipeline.fit(X_hps_train, y_hps_train)

        y_pred = pipeline.predict(X_hps_test)
        accuracy = accuracy_score(y_hps_test, y_pred)
        return accuracy

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Train a text classification model with hyperparameter tuning."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["svm", "naive-bayes"],
        help="The classification model to train ('svm' or 'naive-bayes').",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="The number of Optuna trials to run for hyperparameter search.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for hps dataset splitting"
    )
    args = parser.parse_args()

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_hps_train,
        y_hps_train,
        X_hps_test,
        y_hps_test,
        encoder,
    ) = load_dataset_splits()

    print(f"Starting hyperparameter search for {args.model.upper()} using Optuna...")
    print(f"   Running for {args.trials} trials. This may take a while.")

    start_time = time.time()
    objective_func = create_objective(
        args.model, X_hps_train, y_hps_train, X_hps_test, y_hps_test, args.seed
    )
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name=f"{args.model}_tuning"
    )
    study.optimize(objective_func, n_trials=args.trials, show_progress_bar=True)
    tuning_duration = time.time() - start_time

    print(f"\nHyperparameter search complete in {tuning_duration:.2f} seconds.")
    print(f"   Best validation accuracy: {study.best_value:.4f}")
    print("   Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"     - {key}: {value}")
    print("-" * 30)

    # 3. Train final model with best hyperparameters
    print("Training final model with best hyperparameters on the full training data")
    best_params = study.best_params

    # Create the final pipeline with the optimal parameters
    if args.model == "svm":
        final_classifier = SGDClassifier(
            alpha=best_params["svm_C"],
            loss="hinge",
            random_state=args.seed,
            max_iter=2000,
        )
    else:  # naive_bayes
        final_classifier = MultinomialNB(alpha=best_params["nb_alpha"])

    final_vectorizer = TfidfVectorizer(
        ngram_range=(1, best_params["tfidf_ngram_range"]),
        min_df=best_params["tfidf_min_df"],
        max_df=best_params["tfidf_max_df"],
        analyzer="word",
        token_pattern=r"\b[a-zà-ú]+\b",
        lowercase=True,
    )
    final_pipeline = Pipeline(
        [("vectorizer", final_vectorizer), ("classifier", final_classifier)]
    )

    # Train on the training set
    final_pipeline.fit(X_train, y_train)
    print("Final model trained successfully.")

    # 4. Evaluate the final model on the test set
    print("Evaluating final model on the test set...")
    y_pred_test = final_pipeline.predict(X_test)
    rootpath = Path("Models/baseline/" + args.model)
    rootpath.mkdir(parents=True, exist_ok=True)
    result = pl.DataFrame(dict(y_pred=y_pred_test, y_true=y_test))
    result.write_csv(rootpath / "eval.csv")

    test_accuracy = accuracy_score(y_test, y_pred_test)

    with open(rootpath / "classifier.pkl", "wb") as classifier_file:
        pickle.dump(final_classifier, classifier_file)

    with open(rootpath / "vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(final_vectorizer, vectorizer_file)

    with open(rootpath / "encoder.pkl", "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)

    # Just create the predictions and save it into the disk

    print("\n--- FINAL PERFORMANCE ON TEST SET ---")
    print(f"Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    # Get target names for a more readable report
    print(classification_report(y_test, y_pred_test))


if __name__ == "__main__":
    main()
