"""
predict.py
===========
Test / predict a SINGLE IMAGE using all 6 trained models.

Usage:
  python predict.py <path_to_image>

Example:
  python predict.py "E:/photos ml/WIN_20261020_09_44_03_Pro.jpg"
  python predict.py "C:/Users/PC/Desktop/test.jpg"
"""

import sys
import os
import cv2
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from data_preparation import extract_features, CLASS_NAMES

MODELS_DIR = r"E:\photos ml\project\models"

MODEL_FILES = {
    "Logistic Regression"      : "Logistic_Regression.pkl",
    "Decision Tree"            : "Decision_Tree.pkl",
    "Naive Bayes"              : "Naive_Bayes.pkl",
    "K-Nearest Neighbours"     : "K-Nearest_Neighbours.pkl",
    "Random Forest"            : "Random_Forest.pkl",
    "Support Vector Classifier": "Support_Vector_Classifier.pkl",
}


def preprocess_image(image_path):
    """Extract features from image — same pipeline as training."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"\n[ERROR] Cannot read image: {image_path}")
        sys.exit(1)
    return extract_features(img, augment=False).reshape(1, -1)


def predict_all(image_path):
    print("\n" + "=" * 60)
    print(f"  IMAGE: {os.path.basename(image_path)}")
    print("=" * 60)

    features = preprocess_image(image_path)
    votes    = {}

    print(f"\n  {'Classifier':<30}  {'Predicted Class':<15}  {'Confidence'}")
    print(f"  {'-'*30}  {'-'*15}  {'-'*10}")

    for name, pkl_file in MODEL_FILES.items():
        model_path = os.path.join(MODELS_DIR, pkl_file)

        if not os.path.exists(model_path):
            print(f"  {name:<30}  [model not found: {model_path}]")
            continue

        model       = joblib.load(model_path)
        pred_id     = model.predict(features)[0]
        pred_label  = CLASS_NAMES[pred_id]

        # Confidence (probability if available)
        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(features)[0]
            confidence = f"{proba[pred_id]*100:.1f}%"
        else:
            confidence = "N/A"

        print(f"  {name:<30}  {pred_label:<15}  {confidence}")
        votes[pred_label] = votes.get(pred_label, 0) + 1

    # ── Majority vote ─────────────────────────────────────────────────────────
    best_class = max(votes, key=votes.get)
    print(f"\n  {'-'*55}")
    print(f"  MAJORITY VOTE  ->  {best_class}  ({votes[best_class]}/6 classifiers agree)")
    print(f"  {'-'*55}")
    print(f"\n  Vote breakdown: {votes}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print('  python predict.py "E:\\photos ml\\WIN_20261020_09_44_03_Pro.jpg"')
        print("\nOR edit image_path below and run without arguments:\n")

        # ── EDIT THIS LINE to test a specific image ────────────────────────────
        image_path = "E:/photos ml/WIN_20261020_09_44_03_Pro.jpg"
        # ───────────────────────────────────────────────────────────────────────
    else:
        image_path = sys.argv[1]

    predict_all(image_path)
