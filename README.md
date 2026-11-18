# 10-Class Image Classification Project

## Overview
End-to-end machine-learning pipeline that classifies 500 images (10 classes × 50 images)
using six classic ML algorithms, with full performance evaluation.

## Project Structure
```
E:\photos ml\project\
├── main.py                  <- Entry point (run this)
├── data_preparation.py      <- Image loading, annotation, train/test split
├── classifiers.py           <- All 6 classifier definitions
├── evaluation.py            <- Metrics, confusion matrices, comparison chart
├── requirements.txt         <- Python dependencies
├── dataset\
│   ├── labels.csv           <- Annotation file (filename, label_id, label_name)
│   ├── X_train.npy          <- Training feature matrix
│   ├── X_test.npy           <- Test feature matrix
│   ├── y_train.npy          <- Training labels
│   └── y_test.npy           <- Test labels
├── models\
│   ├── Logistic_Regression.pkl
│   ├── Decision_Tree.pkl
│   ├── Naive_Bayes.pkl
│   ├── K-Nearest_Neighbours.pkl
│   ├── Random_Forest.pkl
│   └── Support_Vector_Classifier.pkl
└── results\
    ├── labels.csv            <- Data annotation
    ├── results_summary.csv   <- Accuracy / Precision / Recall / F1 for all models
    ├── performance_comparison.png  <- Grouped bar chart
    ├── cm_Logistic_Regression.png
    ├── cm_Decision_Tree.png
    ├── cm_Naive_Bayes.png
    ├── cm_K-Nearest_Neighbours.png
    ├── cm_Random_Forest.png
    └── cm_Support_Vector_Classifier.png
```

## Dataset Preparation
- **Source** : `E:\photos ml\` (530 JPG images, flat directory)
- **Selection** : First 500 images taken in alphabetical/timestamp order
- **Labelling** : Sequential assignment — images 1-50 → class_01, 51-100 → class_02, … 451-500 → class_10
- **Annotation** : Saved to `dataset/labels.csv` (filename, label_id, label_name)
- **Image loading** : OpenCV (`cv2.imread`) → grayscale → resize to 64×64 → flatten
- **Normalisation** : Pixel values divided by 255 (range 0–1)
- **Split** : Stratified 80% train / 20% test (`random_state=42`)

## Algorithms
| # | Algorithm                  | Key Hyperparameters                              |
|---|----------------------------|--------------------------------------------------|
| 1 | Logistic Regression (LR)   | C=1.0, solver=lbfgs, max_iter=2000               |
| 2 | Decision Tree (DT)         | criterion=gini, max_depth=15                     |
| 3 | Naive Bayes (GaussianNB)   | default (works with scaled features)             |
| 4 | K-Nearest Neighbours (KNN) | n_neighbors=7, metric=euclidean                  |
| 5 | Random Forest (RF)         | n_estimators=200, max_depth=20                   |
| 6 | Support Vector Classifier  | kernel=rbf, C=10, gamma=scale                    |

All algorithms are wrapped in a `StandardScaler → Classifier` sklearn Pipeline.

## Performance Evaluation
Each classifier is evaluated with:
- **Accuracy** (overall)
- **Precision** (weighted, per class)
- **Recall** (weighted, per class)
- **F1-Score** (weighted, per class)
- **Confusion Matrix** (heatmap PNG)
- **Classification Report** (per-class breakdown printed to console)

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
cd "E:\photos ml\project"
python main.py
```

Results are written to `E:\photos ml\project\results\`.
