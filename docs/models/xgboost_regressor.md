# XGBoost-like Regressor

## Idea

Still boosting trees, but each tree uses gradient and Hessian information.

Instead of fitting raw residuals, the tree chooses splits using regularized gain.

## Loss

Squared error:

```text
L = 1/2 * (y - prediction)^2
```

For this loss:

```text
gradient = prediction - y
hessian = 1
```

## Leaf Value

For a leaf:

```text
G = sum(gradients)
H = sum(hessians)
```

Prediction added by the leaf:

```text
leaf_value = -G / (H + lambda)
```

## Split Gain

```text
gain = 1/2 * (
    G_left^2 / (H_left + lambda)
  + G_right^2 / (H_right + lambda)
  - G_parent^2 / (H_parent + lambda)
) - gamma
```

The best split is the split with the highest positive gain.

## Fit Steps

```text
1. Start with mean(y)
2. Compute gradients and hessians
3. Build a tree using gain
4. Add tree prediction with learning_rate
5. Repeat
```

## In Code

```text
pureml/ensemble/xgboost_regressor.py
pureml/supervised/tree_based/xgboost_tree_regressor.py
experiments/ensemble/xgboost_regressor_demo.py
```

