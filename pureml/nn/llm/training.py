import numpy as np


def clip_gradients(parameters_and_gradients, max_norm: float) -> None:
    total_norm_squared = 0.0
    for _, grad in parameters_and_gradients:
        total_norm_squared += np.sum(grad**2)

    total_norm = np.sqrt(total_norm_squared)
    if total_norm <= max_norm:
        return

    scale = max_norm / (total_norm + 1e-12)
    for _, grad in parameters_and_gradients:
        grad *= scale
