# PureML

Educational implementations of core machine learning models from scratch.

The goal of this repository is not to beat optimized libraries like scikit-learn, XGBoost, or LightGBM.  
The goal is to understand how classical machine learning algorithms work internally.

## Goals

- Implement machine learning models from scratch using NumPy.
- Keep the code readable and educational.
- Use a sklearn-like API when possible:
  - `fit(X, y)`
  - `predict(X)`
- Compare simple implementations with standard ML libraries later.
- Build the project progressively, from simple models to more advanced ones.

## Current State

### Implemented

- Basic regression metrics:
  - Mean Squared Error
  - Root Mean Squared Error
  - Mean Absolute Error

- Basic train/test split utility

- First tree-based model:
  - `DecisionTreeRegressor`

### Current working demo

The current decision tree demo runs successfully:

```bash
uv run experiments/decision_tree_demo.py
```

Current output:

```text
Predictions: [0. 0. 1. 1. 2. 2.]
MSE: 0.0
```

This means the simple decision tree can correctly fit a small toy regression dataset.

## Project Structure

```text
RewriteMachineLearning/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── ml_models/
│   ├── metrics/
│   │   └── regression.py
│   │
│   ├── model_selection/
│   │   └── train_test_split.py
│   │
│   └── supervised/
│       └── tree_based/
│           └── decision_tree_regressor.py
│
├── experiments/
│   └── decision_tree_demo.py
```

## Design Principles

### Clarity over performance

The code is intentionally simple.  
Some implementations may be slower than optimized libraries, but easier to understand.

### Small steps

Each model should be implemented progressively:

1. Make a minimal version work.
2. Add tests.
3. Add comments.
4. Compare with sklearn.
5. Refactor only when needed.

### Common model API

Models should follow this pattern:

```python
model = SomeModel(...)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Current Focus

The current focus is on tree-based models.

Implementation order:

1. Decision Tree Regressor
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. Second-order GBDT with gradients and hessians
5. LightGBM-like Regressor

## TODO

### Short-term

<!-- - [ ] Add more tests for `DecisionTreeRegressor` -->
<!-- - [ ] Test behavior with different `max_depth` values -->
<!-- - [ ] Add a noisier regression toy dataset -->
- [ ] Add `_node_mse()` helper inside the decision tree
- [ ] Compare `DecisionTreeRegressor` with `sklearn.tree.DecisionTreeRegressor`
- [ ] Add simple plotting for tree predictions

### Metrics

- [x] Mean Squared Error
- [x] Root Mean Squared Error
- [x] Mean Absolute Error
- [ ] R² score
- [ ] Mean Absolute Percentage Error
- [ ] RMSLE

### Model Selection

- [x] Basic train/test split
- [ ] KFold
- [ ] GroupKFold
- [ ] StratifiedKFold
- [ ] Cross-validation helper

### Tree-Based Models

- [x] Decision Tree Regressor
- [ ] Decision Tree Classifier
- [ ] Random Forest Regressor
- [ ] Random Forest Classifier
- [ ] Gradient Boosting Regressor
- [ ] Second-order GBDT
- [ ] LightGBM-like Regressor

### Losses

- [ ] Squared Error loss
- [ ] Absolute Error loss
- [ ] Logistic loss
- [ ] Asymmetric squared error
- [ ] Custom loss interface with:
  - `loss`
  - `grad`
  - `hess`

### Later

- [ ] Linear Regression
- [ ] Logistic Regression
- [ ] Ridge Regression
- [ ] KMeans
- [ ] PCA
- [ ] Simple MLP from scratch (done in class but want to rewrite it here !)

## Notes

This repository is for learning.  
The implementations are expected to be incomplete, inefficient, and sometimes simplified compared to production-grade ML libraries.

The main objective is to understand maths and advanced hidden concepts such as:

- splits
- thresholds
- leaves
- impurity
- loss functions
- gradients
- hessians
- boosting
- regularization
- overfitting
- validation
