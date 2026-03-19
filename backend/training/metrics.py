import math


def mae(predictions: list[float], targets: list[float]) -> float:
    if not predictions:
        return 0.0
    return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)


def rmse(predictions: list[float], targets: list[float]) -> float:
    if not predictions:
        return 0.0
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions))


def spearman_rank_correlation(predictions: list[float], targets: list[float]) -> float:
    if len(predictions) < 2:
        return 0.0
    pred_sorted = sorted(range(len(predictions)), key=lambda i: predictions[i])
    target_sorted = sorted(range(len(targets)), key=lambda i: targets[i])

    pred_rank = {idx: rank for rank, idx in enumerate(pred_sorted)}
    target_rank = {idx: rank for rank, idx in enumerate(target_sorted)}

    n = len(predictions)
    diff_sq = sum((pred_rank[i] - target_rank[i]) ** 2 for i in range(n))
    return 1.0 - ((6.0 * diff_sq) / (n * (n**2 - 1)))
