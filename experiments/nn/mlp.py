import numpy as np

from ml_models.neural_networks.mlp.classifier import MLPClassifier

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = np.array([0, 0, 1, 1])

model = MLPClassifier(input_dim=2, hidden_dims=[4], num_classes=2)

losses = model.fit(
    X=X,
    y=y,
    epochs=1000,
    learning_rate=0.1,
    log_every=100,
)

print("Initial loss:", losses[0])
print("Final loss:", losses[-1])
print("Predictions:", model.predict(X))
print("Targets:", y)
