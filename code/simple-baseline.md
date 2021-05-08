# Simple Baseline
The goal for our model is to assign a label (0, 1) to a sentence to classify it has offensive language(0) or not(1) respectively.

## Usage
To run this simple baseline model:
1. Download the needed data and store them in a folder named `data` inside our directory.
2. In the `code` directory, run `python simple-baseline.py`.
3. On completion, it will print out the f1 score on testing set and generate a result file named `simple_base_test_preds.txt` inside the `output` folder.
4. You can also use `python score.py simple_base_test_preds.txt` to get the testing f1 score that compares this result file with the original labels.

## Model Description
We will use majority class classifier to establish a simple baseline. That is, every data point in the development set and testing set is expected to be the majority label (mode) - NOT (1) of the training set.

## Score

### Development Set
The simple baseline development f1 score is: 0.3992740471869329

### Test Set
The simple baseline testing f1 score is: 0.4189189189189189
