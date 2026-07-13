# Gradient Boosting Regressor

## Idea

Build the model additively:

```text
prediction = initial_prediction + learning_rate * sum(trees)
```

Each new tree learns what the current model is missing.

## Loss

Squared error:

```text
L = 1/2 * (y - prediction)^2
```

Gradient wrt prediction:

```text
dL/dprediction = prediction - y
```

Negative gradient:

```text
y - prediction
```

So for squared error, the next tree learns the residual.

## Fit Steps

```text
1. Start with mean(y)
2. Compute residual = y - prediction
3. Fit a regression tree on residual
4. Add tree prediction with learning_rate
5. Repeat
```

## In Code

```text
pureml/ensemble/gradient_boosting_regressor.py
experiments/ensemble/gradient_boosting_regressor_demo.py
```

