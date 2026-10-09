"""Independent float64 NumPy CE oracle; no production or Torch operations."""
import numpy as np


def independent(logits, mask, classes=None, targets=None):
    """Explicit supported rows and analytic derivative of mean cross entropy."""
    logits = np.asarray(logits, dtype=np.float64)
    gradient = np.zeros_like(logits)
    probabilities = np.zeros_like(logits)
    losses = []
    for row in range(len(logits)):
        ids = np.flatnonzero(mask[row])
        scores = logits[row, ids]
        centered = scores - max(scores)
        exponentials = np.exp(centered)
        probabilities[row, ids] = exponentials / sum(exponentials)
        logs = centered - np.log(sum(exponentials))
        if classes is not None:
            target_index = list(ids).index(classes[row])
            losses.append(-logs[target_index])
            gradient[row] = probabilities[row]
            gradient[row, classes[row]] -= 1
        else:
            weights = np.asarray(targets[row], dtype=np.float64)[ids]
            losses.append(-sum(weight * log for weight, log in zip(weights, logs) if weight > 0))
            gradient[row, ids] = probabilities[row, ids] * sum(weights) - weights
    return np.mean(losses), gradient / len(logits), probabilities
