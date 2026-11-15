from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

STUDENT_TARGET = "Performance Index"
CREDIT_TARGET = "Creditability"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPSILON = 1e-12
TREE_GRID = {"max_depth": [2, 3, 4, 5, 6, 8, None], "min_samples_leaf": [1, 5, 10, 20]}
KNN_GRID = {
    "kneighborsclassifier__n_neighbors": [5, 11, 21, 31, 51, 101],
    "kneighborsclassifier__weights": ["uniform", "distance"],
}


def load_student_performance(csv_path: str = "Student_Performance.csv") -> RegressionInput:
    frame = pd.read_csv(csv_path)
    frame["Extracurricular Activities"] = frame["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    features = frame.drop(columns=STUDENT_TARGET).to_numpy(dtype=float)
    target = frame[STUDENT_TARGET].to_numpy(dtype=float)
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return RegressionInput(train_x, train_y, test_x, test_y)


def load_german_credit(csv_path: str = "german.csv") -> ClassificationInput:
    frame = pd.read_csv(csv_path, sep=";")
    features = frame.drop(columns=CREDIT_TARGET).to_numpy(dtype=float)
    target = frame[CREDIT_TARGET].to_numpy(dtype=int)
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )
    return ClassificationInput(train_x, train_y, test_x, test_y)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1
    return (train - mean) / std, (test - mean) / std, mean, std


def to_raw_scale(weights: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, float]:
    raw_weights = weights / std
    return raw_weights, float(bias - raw_weights @ mean)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-logits))


def binary_cross_entropy(target: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return float(-np.mean(target * np.log(probabilities) + (1 - target) * np.log(1 - probabilities)))


def classification_metrics(target, probabilities, predictions) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
    )


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    train_x, test_x, mean, std = standardize(
        np.asarray(data.train_features, dtype=float), np.asarray(data.test_features, dtype=float)
    )
    target = np.asarray(data.train_target, dtype=float)
    weights = np.zeros(train_x.shape[1])
    bias = 0.0
    loss_history = []
    for _ in range(data.epochs):
        error = train_x @ weights + bias - target
        loss_history.append(float(np.mean(error**2)))
        weights -= data.learning_rate * 2 * train_x.T @ error / len(target)
        bias -= data.learning_rate * 2 * float(error.mean())
    raw_weights, raw_bias = to_raw_scale(weights, bias, mean, std)
    return RegressionResult(raw_weights, raw_bias, loss_history, test_x @ weights + bias)


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    train_x, test_x, mean, std = standardize(
        np.asarray(data.train_features, dtype=float), np.asarray(data.test_features, dtype=float)
    )
    target = np.asarray(data.train_target, dtype=float)
    weights = np.zeros(train_x.shape[1])
    bias = 0.0
    loss_history = []
    for _ in range(data.epochs):
        probabilities = sigmoid(train_x @ weights + bias)
        loss_history.append(binary_cross_entropy(target, probabilities))
        error = probabilities - target
        weights -= data.learning_rate * train_x.T @ error / len(target)
        bias -= data.learning_rate * float(error.mean())
    probabilities = sigmoid(test_x @ weights + bias)
    raw_weights, raw_bias = to_raw_scale(weights, bias, mean, std)
    return ClassificationResult(
        raw_weights, raw_bias, loss_history, probabilities, (probabilities >= 0.5).astype(int)
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression()),
        "decision_tree": GridSearchCV(
            DecisionTreeClassifier(random_state=RANDOM_STATE), TREE_GRID, scoring="roc_auc", cv=folds
        ),
        "knn": GridSearchCV(
            make_pipeline(StandardScaler(), KNeighborsClassifier()), KNN_GRID, scoring="roc_auc", cv=folds
        ),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(data.train_features, data.train_target)
        metrics[name] = classification_metrics(
            data.test_target, model.predict_proba(data.test_features)[:, 1], model.predict(data.test_features)
        )
    return BaselineComparisonResult(**metrics)
