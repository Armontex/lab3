import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from linear_models import (
    load_german_credit,
    load_student_performance,
    train_linear_regression,
    train_logistic_classifier,
)


def plot_losses() -> tuple[Figure, Figure]:
    mse, mse_axis = plt.subplots(figsize=(8, 5))
    mse_axis.plot(train_linear_regression(load_student_performance()).loss_history)
    mse_axis.set(title="Линейная регрессия, Student Performance", xlabel="Эпоха", ylabel="MSE на train", yscale="log")
    bce, bce_axis = plt.subplots(figsize=(8, 5))
    bce_axis.plot(train_logistic_classifier(load_german_credit()).loss_history)
    bce_axis.set(title="Логистическая регрессия, German Credit", xlabel="Эпоха", ylabel="BCE на train")
    for figure in (mse, bce):
        figure.axes[0].grid(alpha=0.3)
        figure.tight_layout()
    return mse, bce


if __name__ == "__main__":
    plot_losses()
    plt.show()
