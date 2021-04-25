# Baseline
The goal for our model is to assign a label (0, 1) to a sentence to classify it has offensive language(0) or not(1) respectively.

## Required Libraries
* numpy
* pandas
* scikit-learn
* nltk

## Usage
To run this baseline model:
1. Download the needed data and store them in a folder named `data`.
2. In the parent directory of `data`, run `python baseline.py`.
3. On completion, it will print out the f1 score on testing set and generate a result file named `test_preds.txt`.
4. You can also use `python score.py` to get the f1 score that compares this result file with the original labels.

## Score
The F1 score for this published baseline is: 0.8599695585996956

## Referenced Paper
