# Extension 1 - Semantic Embeddings of Emoji and Hashtags
emoji + wordsegment + misc preprocessing

## Required Libraries
* numpy
* pandas
* scikit-learn
* gensim
* emoji
* wordsegment

## Score with Baseline SVM

### Keep only alphanumeric characters
The baseline development f1 score is: 0.6963252859626121
The baseline testing f1 score is: 0.6965725806451613

### Only segment tranlated emoji & hashtags
The baseline development f1 score is: 0.6960298906127418
The baseline testing f1 score is: 0.719242541499638

## Score with Fine-tuned Bert

### Keep only alphanumeric characters
The development f1 score is: 0.7625010105909936
The testing f1 score is: 0.8119120163612839

### Only segment tranlated emoji & hashtags
The development f1 score is: 0.7717349751011322
The testing f1 score is: 0.8162637377426447

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