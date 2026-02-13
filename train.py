import argparse
import json
import os

import joblib
import pandas as pd
# training the model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
# pipeline
from sklearn.pipeline import Pipeline
# code carbon
from codecarbon.emissions_tracker import EmissionsTracker


# ----------------------------------------------------------------------------------------------------------------- #


def load_data(path: str) -> pd.DataFrame:
    """
    Loads and cleans the dataset from a CSV file.
    """
    df = pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")
    # Standardize column names used below
    col_map = {c.lower(): c for c in df.columns}
    text_col = col_map.get("statement", col_map.get("text", None))
    label_col = col_map.get("status", col_map.get("label", None))
    if text_col is None or label_col is None:
        raise ValueError(
            f"Expected a text column ('statement' or 'text') and a label column ('status' or 'label'). Found: {list(df.columns)}"
        )
    df = df.rename(columns={text_col: "text", label_col: "label"})
    # Drop unnamed index columns if present
    for c in list(df.columns):
        if c.lower().startswith("unnamed"):
            df = df.drop(columns=[c])
    # Clean
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df


# build a pipeline, TfID Vectorisation & Logistic Resgression
def build_pipeline() -> Pipeline:
    """
    Constructs the Scikit-learn pipeline with a vectorizer and a classifier.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),  # both single and two word phrases for n gram
                    min_df=3,     # very rare words
                    max_df=0.95,  # very common words
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",  # must be balanced, it tells the model to pay extra attention to less
                    # frequent labels, which is very important for imbalanced datasets
                    n_jobs=-1,  # Use all available CPU cores
                ),
            ),
        ]
    )


def main():
    """
    Main function to run the data loading, training, and evaluation pipeline.
    """
    parser = argparse.ArgumentParser(description="Train MindCue text classifier")
    parser.add_argument("--data_path", default="Combined Data.csv",
                        help="CSV with columns: statement/text, status/label")
    parser.add_argument("--model_out", default="models/mindcue_textclf.joblib")
    parser.add_argument("--labels_out", default="models/labels.json")
    # FIX: Added a 'help' string. The error occurs if this argument definition is missing, for metrics JSON.
    parser.add_argument("--metrics_out", default="static/performance_metrics.json",
                        help="Path to save model performance metrics JSON.")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--subsample", type=int, default=0, help="Optional: train on N rows (stratified). 0 = full")
    args = parser.parse_args()

    # --- Initialize the tracker manually ---
    tracker = EmissionsTracker(output_dir="reports/", project_name="mindcue_training")

    try:
        # --- Start tracking emissions ---
        tracker.start()

        # Ensure the output directory for the model exists
        os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

        print("Loading data...")
        df = load_data(args.data_path)

        # Optional stratified subsample for faster training on weak machines
        if args.subsample and args.subsample > 0 and args.subsample < len(df):
            print(f"Subsampling to {args.subsample} rows...")
            df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(
                frac=min(1.0, args.subsample / len(df)), random_state=args.random_state
            ))
            df = df.sample(frac=1, random_state=args.random_state).reset_index(drop=True)

        X = df["text"].values
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )

        print("Building and training the pipeline...")
        pipe = build_pipeline()
        pipe.fit(X_train, y_train)

        # Evaluate
        print("Evaluating model performance...")
        y_pred = pipe.predict(X_test)
        try:
            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            report = classification_report(y_test, y_pred, digits=3)
            print(f"\nAccuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}")
            print("\nClassification Report:\n", report)
            # ---  Save metrics to a JSON file, for final webpage ---
            metrics = {
                "accuracy": acc,
                "macro_f1_score": macro_f1
            }
            with open(args.metrics_out, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            print(f"Saved performance metrics to {args.metrics_out}")
            # ----------------------------------------
        except Exception as e:
            print(f"Evaluation failed: {e}")

        # Save artifacts
        print("Saving model and labels...")
        joblib.dump(pipe, args.model_out)
        labels = list(sorted(pd.Series(y).unique().tolist()))
        with open(args.labels_out, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        print(f"Saved model to {args.model_out}\nSaved labels to {args.labels_out}")
    finally:
        # --- Stop tracking and save the report ---
        # This `finally` block ensures the tracker stops even if an error occurs.
        _ = tracker.stop()
        print("Emissions report saved to 'reports/' directory.")


if __name__ == "__main__":
    main()
