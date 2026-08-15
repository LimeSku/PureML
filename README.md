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
  - byte-pair tokenizer
  - language-model dataset
  - embeddings
  - causal self-attention
  - transformer block
  - next-token cross-entropy
  - NumPy backpropagation
  - Adam training loop
  - text generation
  - PyTorch backend with automatic CUDA, MPS, or CPU selection

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
uv run python experiments/llm/torchgpt_training.py shakespeare
uv run python experiments/llm/torchgpt_training.py shakespeare \
  --checkpoint-dir checkpoints/shakespeare
uv run python experiments/llm/torchgpt_training.py shakespeare \
  --resume checkpoints/shakespeare/last.pt
uv run python scripts/download_tiny_stories.py
uv run python scripts/prepare_discord_dataset.py path/to/discord-exports
uv run python experiments/llm/torchgpt_training.py tinystories \
  --checkpoint-dir checkpoints/tinystories
uv run python experiments/llm/torchgpt_generation.py \
  checkpoints/shakespeare/best.pt "ROMEO:"
```

`--checkpoint-dir` writes `last.pt` every `--save-every` steps and at the end
of training. It also updates `best.pt` whenever the validation loss improves.
When resuming, `--steps` is the total target step, not a number of additional
steps.

The TinyStories downloader retrieves the GPT-4 train and validation files from
the official Hugging Face dataset. They use about 2.25 GB of disk space. By
default, the training experiment loads the first 25 million training characters
and 2 million validation characters, trains a 1,024-token BPE vocabulary on the
first million training characters, and trains the model for 10,000 steps. Use
`--max-train-characters` and `--max-validation-characters` to change these
limits; a value of `0` loads the entire corresponding file.

TinyStories uses BPE by default. Shakespeare and the toy dataset keep the
character tokenizer unless `--tokenizer bpe` is passed. Configure BPE with
`--vocab-size` and `--tokenizer-training-characters`, or use
`--tokenizer character` to compare both approaches. Training checkpoints store
the tokenizer type and BPE merge rules, so resumed runs recover the exact same
tokenization automatically.

New TorchGPT runs share the token embedding matrix with the output projection
by default. Use `--no-tie-embeddings` for an untied ablation. The setting is
stored in checkpoints and restored automatically when training resumes; older
checkpoints remain untied.

### Discord chat corpus

`prepare_discord_dataset.py` converts one or more JSON exports produced by
[DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) into a plain
next-token corpus similar to Tiny Shakespeare:

```bash
uv run python scripts/prepare_discord_dataset.py path/to/discord-exports \
  --output datasets/discord/input.txt
```

Directories are searched recursively, so partitioned exports can be passed as a
single directory. Messages are sorted chronologically and deduplicated by Discord
message ID. Bots and system notifications are excluded by default, full URLs are
replaced with `<URL>`, and image, video, audio, file, sticker, and embed-only
messages receive textual placeholders. A gap of 60 minutes starts a new
conversation; configure it with `--session-gap-minutes`.

The generated text uses explicit speaker and conversation markers:

```text
<CONVERSATION>
<USER_0001>
First message
<USER_0002>
Reply
<END_CONVERSATION>
```

The output directory also receives `speakers.json`, which privately maps aliases
to Discord identities, and `stats.json`, which reports corpus and per-speaker
sizes without message contents. `datasets/discord/` is ignored by Git because all
three files contain or describe private data.

The resulting corpus can use the existing Shakespeare training path with BPE:

```bash
uv run python experiments/llm/torchgpt_training.py shakespeare \
  --path datasets/discord/input.txt \
  --tokenizer bpe \
  --vocab-size 2048 \
  --tokenizer-training-characters 2000000 \
  --batch-size 8 \
  --checkpoint-dir checkpoints/discord
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
    ├── llm/
    ├── nn/
    │   ├── cnn/
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
- [x] Save/load PyTorch TinyGPT training checkpoints
- [ ] Mini-batch TinyGPT training
