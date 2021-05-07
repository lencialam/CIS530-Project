# Extension 1 - Preprocess
emoji + wordsegment + misc preprocessing

## Required Libraries
* numpy
* pandas
* scikit-learn
* gensim
* emoji
* wordsegment

## Score with CountVectorizer

### Keep only alphanumeric characters
The baseline development f1 score is: 0.6989256340018821
The baseline testing f1 score is: 0.7122215334653552

### Only segment tranlated emoji & hashtags
The baseline development f1 score is: 0.6884154405256526
The baseline testing f1 score is: 0.7291016795695866

## Score with Google Pretrained Word2Vec

### Keep only alphanumeric characters
The baseline development f1 score is: 0.6842459098413616
The baseline testing f1 score is: 0.723913494117798

### Only segment tranlated emoji & hashtags
The baseline development f1 score is: 0.6739185522423332
The baseline testing f1 score is: 0.6894279093562852

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