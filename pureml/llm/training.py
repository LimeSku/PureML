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


def train_language_model_step(
    model,
    loss_fn,
    optimizer,
    x: list[int],
    y: np.ndarray,
    max_grad_norm: float | None = None,
) -> float:
    logits = model(x)
    loss = loss_fn(logits, y)
    dlogits = loss_fn.backward()

    model.backward(dlogits)
    params_grads = model.parameters_and_gradients()
    if max_grad_norm is not None:
        clip_gradients(parameters_and_gradients=params_grads, max_norm=max_grad_norm)
    optimizer.step(params_grads)
    return loss


def train_language_model_batch(
    model,
    loss_fn,
    optimizer,
    x_batch: list[list[int]],
    y_batch: list[np.ndarray],
    max_grad_norm: float,
):
    if len(x_batch) != len(y_batch):
        raise ValueError("x_batch and y_batch must have the same length")

    accumulated_gradients = None
    parameters = None
    total_loss = 0.0
    for x, y in zip(x_batch, y_batch):
        logits = model(x)
        total_loss += loss_fn(logits, y)
        dlogits = loss_fn.backward()

        model.backward(dlogits)
        params_grads = model.parameters_and_gradients()
        if accumulated_gradients is None:
            parameters = [parameter for parameter, _ in params_grads]
            accumulated_gradients = [np.zeros_like(grad) for _, grad in params_grads]

        for accumulated, (_, grad) in zip(
            accumulated_gradients,
            params_grads,
        ):
            accumulated += grad
        batch_size = len(x_batch)
        averaged_params_grads = []

        for param, accumulated in zip(parameters, accumulated_gradients):
            accumulated /= batch_size
            averaged_params_grads.append((param, accumulated))

        if max_grad_norm is not None:
            clip_gradients(
                parameters_and_gradients=averaged_params_grads, max_norm=max_grad_norm
            )
    optimizer.step(params_grads)
    return total_loss / batch_size
