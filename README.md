# PureML

Educational implementations of machine learning models from scratch, mainly to pass the time but also to get some deeper understanding.

## Current State

Implemented:

- Regression metrics:
  - Mean Squared Error
  - Root Mean Squared Error
  - Mean Absolute Error
- Basic train/test split utility
- `DecisionTreeRegressor`
- Basic neural network components:
  - dense layer
  - ReLU
  - softmax cross-entropy
  - MLP classifier with backpropagation
- TinyGPT-style forward-pass components:
  - character tokenizer
  - language-model dataset
  - embeddings
  - causal self-attention
  - transformer block
  - random text generation

## Demos

```bash
uv run python experiments/tree_based/decision_tree_regressor_toy.py
uv run python experiments/tree_based/decision_tree_classifier_iris.py
uv run python experiments/neural_networks/mlp_toy.py
uv run python experiments/neural_networks/mlp_iris.py
uv run python experiments/neural_networks/mlp_mnist.py
uv run python experiments/llm/dataset_windows.py
uv run python experiments/llm/tinygpt_generation.py
```

## Project Structure

```text
PureML/
├── datasets/
├── experiments/
│   ├── llm/
│   ├── neural_networks/
│   └── tree_based/
└── pureml/
    ├── metrics/
    ├── model_selection/
    ├── neural_networks/
    │   ├── llm/
    │   └── mlp/
    └── supervised/
        └── tree_based/
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
- [ ] Decision Tree Classifier
- [ ] Random Forest Regressor
- [ ] Gradient Boosting Regressor

### Neural Networks

- [x] Dense layer
- [x] ReLU
- [x] Softmax cross-entropy
- [x] MLP classifier
- [x] Mini-batch training
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
- [ ] Add cross-entropy loss demo
- [ ] Decide whether training stays NumPy or moves to PyTorch
