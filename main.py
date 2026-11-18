"""
main.py
========
Entry point for the 10-class image classification project.

Pipeline:
  1. Data preparation  (load images, annotate, train/test split)
  2. Train each of the 6 classifiers on the training set
  3. Evaluate each classifier on the test set
  4. Save summary metrics (CSV + plots)

Usage:
  python main.py
"""

import os
import time
import joblib

from data_preparation import prepare, load_splits, DATASET_DIR
from classifiers      import get_classifiers
from evaluation       import evaluate_classifier, save_summary

MODELS_DIR  = r"E:\photos ml\project\models"
RESULTS_DIR = r"E:\photos ml\project\results"


def main():
    print("\n" + "#" * 65)
    print("  10-CLASS IMAGE CLASSIFICATION PROJECT")
    print("  Algorithms: LR | DT | Naive Bayes | KNN | RF | SVC")
    print("#" * 65)

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: Data Preparation ──────────────────────────────────────────────
    print("\n[STEP 1]  Data Preparation")
    npy_exists = all(
        os.path.exists(os.path.join(DATASET_DIR, f))
        for f in ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy")
    )

    if npy_exists:
        print("  Pre-saved splits found. Loading from disk …")
        X_train, X_test, y_train, y_test = load_splits()
    else:
        print("  Running full data preparation pipeline …")
        X_train, X_test, y_train, y_test = prepare()

    print(f"\n  Train set : {X_train.shape[0]} samples")
    print(f"  Test  set : {X_test.shape[0]}  samples")
    print(f"  Features  : {X_train.shape[1]} (flattened 64x64 grayscale pixels)")

    # ── Step 2 & 3: Train + Evaluate each classifier ─────────────────────────
    print("\n[STEP 2 & 3]  Training and Evaluation")

    classifiers  = get_classifiers()
    results_list = []

    for name, model in classifiers.items():
        print(f"\n  >> Training: {name} …")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"     Training time: {elapsed:.2f}s")

        # Save trained model
        safe = name.replace(" ", "_").replace("/", "_")
        model_path = os.path.join(MODELS_DIR, f"{safe}.pkl")
        joblib.dump(model, model_path)
        print(f"     Model saved -> {model_path}")

        # Evaluate
        metrics = evaluate_classifier(name, model, X_test, y_test)
        metrics["Train_Time_s"] = round(elapsed, 2)
        results_list.append(metrics)

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    print("\n[STEP 4]  Performance Summary")
    df_summary = save_summary(results_list)

    print("\n" + "#" * 65)
    print("  PROJECT COMPLETE")
    print(f"  Results saved in : {RESULTS_DIR}")
    print(f"  Models  saved in : {MODELS_DIR}")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    main()
