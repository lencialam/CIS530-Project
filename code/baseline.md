# Baseline
The goal for our model is to assign a label (0, 1) to a sentence to classify it has offensive language(0) or not(1) respectively.

## Required Libraries
* numpy
* pandas
* scikit-learn

## Usage
To run this baseline model:
1. Download the needed data and store them in a folder named `data` inside our directory.
2. In the `code` directory, run `python baseline.py`.
3. On completion, it will print out the f1 score on testing set and generate a result file named `base_test_preds.txt` inside the `output` folder.
4. You can also use `python score.py base_test_preds.txt` to get the testing f1 score that compares this result file with the original labels.

## Model Description
We use a simple Support Vector Machine model with count vectorization as our baseline. This published baseline scored a macro f1 score of 0.690 on the test dataset in our referenced paper.

## Score with baseline SVM model

### Development Set
The baseline development f1 score is: 0.6941975551544496

### Test Set
The baseline testing f1 score is: 0.7059404698698091

## Referenced Paper
```
@inproceedings{zampieri2019semeval,
  title = {Semeval-2019 task 6: Identifying and categorizing offensive language in social media (offenseval)},
  author = {Zampieri, Marcos and Malmasi, Shervin and Nakov, Preslav and Rosenthal, Sara and Farra, Noura and Kumar, Ritesh},
  booktitle = {Proceedings of the 13th International Workshop on Semantic Evaluation (SemEval-2019)},
  year = {2019},
  pages = {75-86}
}
```