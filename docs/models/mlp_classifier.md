# MLP Classifier

## Idea

A stack of dense layers with nonlinear activations.

```text
X -> Dense -> ReLU -> ... -> Dense -> logits
```

The final layer outputs raw class scores, not probabilities.

## Loss

Softmax cross-entropy:

```text
proba = softmax(logits)
loss = -log(proba[true_class])
```

Gradient wrt logits:

```text
dlogits = proba - one_hot(y)
```

This gradient is backpropagated through every layer.

## Backprop

Dense layer:

```text
out = X @ W + b
dW = X.T @ dout
db = sum(dout)
dX = dout @ W.T
```

ReLU:

```text
pass gradient only where input > 0
```

## Fit Steps

```text
1. Forward pass to get logits
2. Compute softmax cross-entropy
3. Backward pass from loss to layers
4. Update dense layer weights
5. Repeat by batch and epoch
```

## In Code

```text
pureml/neural_networks/mlp/classifier.py
pureml/neural_networks/mlp/layers.py
pureml/neural_networks/mlp/losses.py
experiments/neural_networks/mlp_mnist.py
```

