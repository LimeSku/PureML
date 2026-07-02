import numpy as np


class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None
        self.targets = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        self.targets = targets
        # numerical stability ("avoid" exponential of large values)
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        n_samples = logits.shape[0]
        correct_class_probs = self.probs[np.arange(n_samples), targets]
        return float(-np.mean(np.log(correct_class_probs + 1e-12)))

    def backward(self) -> np.ndarray:
        n_samples = self.probs.shape[0]
        dlogits = self.probs.copy()
        dlogits[np.arange(n_samples), self.targets] -= 1
        dlogits /= n_samples
        return dlogits

    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        return self.forward(logits, targets)
