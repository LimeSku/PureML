# PureML

Educational implementations of machine learning models from scratch, mainly to pass the time but also to get some deeper understanding.

## Current State

Implemented:

- Regression metrics:
  - Mean Squared Error
  - Root Mean Squared Error
  - Mean Absolute Error
- Basic train/test split utility
- Tree-based models:
  - Decision Tree Regressor / Classifier
  - Random Forest Classifier
  - Gradient Boosting Regressor / Classifier
  - XGBoost-like Regressor
- Basic neural network components:
  - dense layer
  - ReLU
  - softmax cross-entropy
  - MLP classifier with backpropagation
- TinyGPT-style language model components:
  - character tokenizer
  - language-model dataset
  - embeddings
  - causal self-attention
  - transformer block
  - next-token cross-entropy
  - NumPy backpropagation
  - Adam training loop
  - text generation

## Live Training Dashboard (TUI)

A [Textual](https://textual.textualize.io/) terminal UI that trains the models
live. Pick a model and a dataset from the sidebar, tune the hyperparameters, press
**Train**, and watch the loss curve (MLP) or the forest grow tree-by-tree (Random
Forest), then see the final train/test accuracy and a confusion matrix.

```bash
uv run pureml-tui          # or: uv run python -m pureml.tui
```

- **Models:** Decision Tree, Random Forest, MLP (grouped by family in the sidebar)
- **Datasets:** Iris, MNIST
- **Controls:** `Max train rows` caps the training set (keeps the from-scratch trees
  responsive on MNIST), plus per-model hyperparameters that update when you switch
  models — e.g. `Max depth` / `Trees` for the tree models, `Epochs` /
  `Learning rate` / `Batch size` for the MLP.

## Demos

```bash
uv run python experiments/tree_based/decision_tree_regressor_toy.py
uv run python experiments/tree_based/decision_tree_classifier_iris.py
uv run python experiments/neural_networks/mlp_toy.py
uv run python experiments/neural_networks/mlp_iris.py
uv run python experiments/neural_networks/mlp_mnist.py
uv run python experiments/ensemble/gradient_boosting_regressor_demo.py
uv run python experiments/ensemble/gradient_boosting_classifier_demo.py
uv run python experiments/ensemble/xgboost_regressor_demo.py
uv run python experiments/llm/dataset_windows.py
uv run python experiments/llm/tinygpt_generation.py
uv run python experiments/llm/tinygpt_training.py
uv run python experiments/llm/tinygpt_tiny_shakespeare.py
```

## Model Notes

```text
docs/models/gradient_boosting_regressor.md
docs/models/xgboost_regressor.md
docs/models/random_forest_classifier.md
docs/models/mlp_classifier.md
```

## Project Structure

```text
PureML/
├── datasets/
├── experiments/
│   ├── llm/
│   ├── neural_networks/
│   └── tree_based/
├── tests/
└── pureml/
    ├── datasets/          # shared Iris / MNIST loaders
    ├── ensemble/
    ├── metrics/
    ├── model_selection/
    ├── nn/
    │   ├── cnn/
    │   ├── llm/
    │   └── mlp/
    ├── optimizer/
    ├── supervised/
    │   └── tree_based/
    └── tui/               # Textual live training dashboard
```


## TODO

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
- [ ] StratifiedKFold
- [ ] Cross-validation helper

### Tree-Based Models

- [x] Decision Tree Regressor
- [x] Decision Tree Classifier
- [x] Random Forest Classifier
- [ ] Random Forest Regressor
- [x] Gradient Boosting Regressor
- [x] Gradient Boosting Classifier
- [x] XGBoost-like Regressor
- [ ] XGBoost-like Classifier

### Neural Networks

- [x] Dense layer
- [x] ReLU
- [x] Softmax cross-entropy
- [x] MLP classifier
- [x] Mini-batch training
- [ ] CNN classifier
- [ ] Clean up the flexible multi-layer MLP API
- [ ] Add simple tests
- [ ] Try regularization on MNIST

### LLM

- [x] Character tokenizer
- [x] Language-model dataset
- [x] Embeddings
- [x] Causal self-attention
- [x] Transformer block
- [x] TinyGPT forward pass
- [x] Next-token cross-entropy
- [x] TinyGPT NumPy backpropagation
- [x] Adam optimizer
- [x] Tiny Shakespeare training demo
- [ ] Save/load TinyGPT weights
- [ ] Mini-batch TinyGPT training
