import json

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from experiments import run_experiments
from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    binary_cross_entropy,
    classification_metrics,
    compare_classical_models,
    load_german_credit,
    load_student_performance,
    standardize,
    train_linear_regression,
    train_logistic_classifier,
)

RNG = np.random.default_rng(0)
LINE_X = np.column_stack([RNG.uniform(0, 10, 200), RNG.uniform(1000, 5000, 200)])
LINE_Y = 3 * LINE_X[:, 0] - 0.002 * LINE_X[:, 1] + 5


@pytest.fixture
def line():
    return train_linear_regression(RegressionInput(LINE_X[:150], LINE_Y[:150], LINE_X[150:], LINE_Y[150:]))


@pytest.fixture
def blobs():
    rng = np.random.default_rng(1)
    features = np.vstack([rng.normal(-4, 1, (100, 2)), rng.normal(4, 1, (100, 2))])
    target = np.repeat([0, 1], 100)
    order = rng.permutation(200)
    features, target = features[order], target[order]
    return ClassificationInput(features[:150], target[:150], features[150:], target[150:])


def test_standardize_uses_train_statistics():
    train_x, test_x, mean, std = standardize(np.array([[1.0, 7.0], [3.0, 7.0]]), np.array([[5.0, 9.0]]))
    assert train_x.tolist() == [[-1, 0], [1, 0]]
    assert test_x.tolist() == [[3, 2]]
    assert (mean.tolist(), std.tolist()) == ([2, 7], [1, 1])


def test_linear_regression_recovers_line(line):
    assert line.weights == pytest.approx([3, -0.002], abs=1e-3)
    assert line.bias == pytest.approx(5, abs=1e-2)
    assert line.predictions == pytest.approx(LINE_Y[150:], abs=1e-2)


def test_linear_regression_weights_in_raw_units(line):
    assert LINE_X[150:] @ line.weights + line.bias == pytest.approx(line.predictions)


def test_linear_regression_loss_history(line):
    assert len(line.loss_history) == 800
    assert all(type(loss) is float for loss in line.loss_history)
    assert np.all(np.diff(line.loss_history) <= 1e-12)
    assert line.loss_history[-1] < 1e-4


def test_linear_regression_constant_feature():
    features = np.column_stack([LINE_X[:, 0], np.ones(200)])
    result = train_linear_regression(RegressionInput(features, LINE_Y, features, LINE_Y, epochs=10))
    assert np.isfinite(result.weights).all()
    assert len(result.loss_history) == 10


@pytest.mark.parametrize(
    "target, probabilities, expected",
    [
        ([1, 0], [0.5, 0.5], np.log(2)),
        ([1, 0], [1, 0], 0),
        ([1], [0], -np.log(1e-12)),
    ],
)
def test_binary_cross_entropy(target, probabilities, expected):
    assert binary_cross_entropy(np.array(target), np.array(probabilities)) == pytest.approx(expected)


def test_logistic_classifier_separates_blobs(blobs):
    result = train_logistic_classifier(blobs)
    assert result.predictions.tolist() == blobs.test_target.tolist()
    assert ((result.probabilities >= 0) & (result.probabilities <= 1)).all()
    assert result.predictions.tolist() == (result.probabilities >= 0.5).astype(int).tolist()
    assert result.weights.shape == (2,)


def test_logistic_classifier_loss_history(blobs):
    result = train_logistic_classifier(blobs)
    assert len(result.loss_history) == 1000
    assert result.loss_history[0] == pytest.approx(np.log(2))
    assert np.all(np.diff(result.loss_history) <= 1e-12)


def test_classification_metrics():
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], [0, 1, 0, 1])
    assert metrics.roc_auc == pytest.approx(0.75)
    assert (metrics.accuracy, metrics.precision, metrics.recall, metrics.f1) == (0.5, 0.5, 0.5, 0.5)


def test_compare_classical_models(blobs):
    result = compare_classical_models(blobs)
    for metrics in (result.logistic_regression, result.decision_tree, result.knn):
        assert metrics.accuracy == pytest.approx(1)


def test_student_dataset():
    data = load_student_performance()
    assert data.train_features.shape == (8000, 5)
    result = train_linear_regression(data)
    exact = LinearRegression().fit(data.train_features, data.train_target)
    assert result.weights == pytest.approx(exact.coef_, rel=1e-3)
    assert np.mean((result.predictions - data.test_target) ** 2) == pytest.approx(4.0826, abs=1e-4)


def test_german_dataset():
    data = load_german_credit()
    assert data.train_features.shape == (800, 20)
    assert data.test_target.mean() == pytest.approx(0.7)
    result = train_logistic_classifier(data)
    own = classification_metrics(data.test_target, result.probabilities, result.predictions)
    assert own.roc_auc == pytest.approx(0.8081, abs=1e-4)
    assert own.accuracy == pytest.approx(0.77)
    assert compare_classical_models(data).logistic_regression.roc_auc == pytest.approx(own.roc_auc, abs=0.01)



def test_experiment_results_reproduce():
    with open("experiment_results.json") as file:
        saved = json.load(file)
    results = run_experiments()
    assert results["linear_regression"] == pytest.approx(saved["linear_regression"])
    assert results["classification"].keys() == saved["classification"].keys()
    for model, metrics in saved["classification"].items():
        assert results["classification"][model] == pytest.approx(metrics)
