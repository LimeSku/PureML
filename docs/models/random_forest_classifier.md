# Random Forest Classifier

## Idea

Train many decision trees and combine their votes.

Each tree sees a different bootstrap sample of rows.

Each split can also see only a subset of features.

## Prediction

Each tree predicts one class:

```text
tree_1 -> class A
tree_2 -> class B
tree_3 -> class A
```

The forest predicts the most common class:

```text
prediction = majority_vote(tree_predictions)
```

Class probabilities are vote proportions:

```text
proba[class] = trees_voting_for_class / n_trees
```

## Why It Works

One tree has high variance.

Many different trees average out some of that variance.

Bootstrap rows and random feature subsets make the trees less correlated.

## Fit Steps

```text
1. Sample training rows with replacement
2. Fit a decision tree
3. Repeat for n_estimators
4. Predict by majority vote
```

## Extra

Out-of-bag samples are rows not used by a tree bootstrap sample.

They can estimate performance without a separate validation set.

## In Code

```text
pureml/ensemble/random_forest_classifier.py
experiments/ensemble/random_forest_classifier_iris.py
experiments/ensemble/random_forest_classifier_mnist.py
```

