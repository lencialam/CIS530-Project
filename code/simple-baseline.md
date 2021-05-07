# Simple Baseline
The goal for our model is to assign a label (0, 1) to a sentence to classify it has offensive language(0) or not(1) respectively.

## Model Description
We will use majority class classifier to establish a simple baseline. That is, every data point in the development set and testing set is expected to be the majority label (mode) - NOT (1) of the training set.

## Score

### Development Set
The simple baseline development f1 score is: 0.3992740471869329

### Test Set
The simple baseline testing f1 score is: 0.4189189189189189