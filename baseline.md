# Baseline
The goal for our model is to assign a label (0, 1) to a sentence to classify it has offensive language(0) or not(1) respectively.

## Required Libraries
* numpy
* pandas
* scikit-learn
* nltk
* gensim

## Usage
To run this baseline model:
1. Download the needed data and store them in a folder named `data`.
2. In the parent directory of `data`, run `python baseline.py` (or `python baseline_word2vec.py` for pretrained word2vec model).
3. On completion, it will print out the f1 score on testing set and generate a result file named `test_preds.txt`.
4. You can also use `python score.py` to get the testing f1 score that compares this result file with the original labels.

## Model Description
We use a simple Logistic Regression model with basic text preprocessing as our baseline. After getting the bag-of-words features, we experimented with two vectorization techniques - CountVectorizer and Google pretrained Word2Vec model.

## Score with CountVectorizer

### Development Set
The baseline development f1 score is: 0.6968686741773813

### Test Set
The baseline testing f1 score is: 0.7155976554481942

## Score with Google Pretrained Word2Vec

### Development Set
The baseline development f1 score is: 0.7040376544125482

### Test Set
The baseline testing f1 score is: 0.7180019674281342

## Referenced Paper
```
@inproceedings{liu-etal-2019-nuli,
    title = "{NULI} at {S}em{E}val-2019 Task 6: Transfer Learning for Offensive Language Detection using Bidirectional Transformers",
    author = "Liu, Ping and Li, Wen and Zou, Liang",
    booktitle = "Proceedings of the 13th International Workshop on Semantic Evaluation",
    month = Jun,
    year = "2019",
    address = "Minneapolis, Minnesota, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/S19-2011",
    doi = "10.18653/v1/S19-2011",
    pages = "87--91",
}
```