def mean_squared_error(predictions: list[float], targets: list[float]) -> float:
    if not predictions:
        return 0.0
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)


def mse_gradient(prediction: float, target: float) -> float:
    return 2.0 * (prediction - target)
