"""
classifiers.py
===============
Defines and trains all 6 classification algorithms:
  1. Logistic Regression (LR)
  2. Decision Tree (DT)
  3. Naive Bayes (GaussianNB)
  4. K-Nearest Neighbours (KNN)
  5. Random Forest Classifier (RFC)
  6. Support Vector Classifier (SVC)

Each classifier is wrapped in a sklearn Pipeline that first applies
StandardScaler (required for LR, KNN, SVC) and then the estimator.
This ensures fair comparison without data-leakage.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model      import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier
from sklearn.naive_bayes        import GaussianNB
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.ensemble           import RandomForestClassifier
from sklearn.svm                import SVC


def get_classifiers():
    """
    Return a dict of {name: sklearn Pipeline} for all 6 classifiers.
    Hyperparameters are set to sensible defaults; max_iter raised for LR.
    """
    classifiers = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=42,
                )),
        ]),

        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(
                criterion="gini",
                max_depth=15,
                random_state=42,
            )),
        ]),

        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),

        "K-Nearest Neighbours": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=7,
                metric="euclidean",
            )),
        ]),

        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
            )),
        ]),

        "Support Vector Classifier": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=10.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                random_state=42,
            )),
        ]),
    }
    return classifiers
