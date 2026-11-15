import json
from dataclasses import asdict

import numpy as np

from linear_models import (
    classification_metrics,
    compare_classical_models,
    load_german_credit,
    load_student_performance,
    train_linear_regression,
    train_logistic_classifier,
)


def run_experiments() -> dict:
    regression = load_student_performance()
    linear = train_linear_regression(regression)
    classification = load_german_credit()
    logistic = train_logistic_classifier(classification)
    own = classification_metrics(classification.test_target, logistic.probabilities, logistic.predictions)
    return {
        "linear_regression": {"test_mse": float(np.mean((linear.predictions - regression.test_target) ** 2))},
        "classification": {"own_logistic": asdict(own), **asdict(compare_classical_models(classification))},
    }


if __name__ == "__main__":
    with open("experiment_results.json", "w") as file:
        json.dump(run_experiments(), file, indent=2)
        file.write("\n")
