# Extension 3 - Semantic Embeddings of Emoji and Hashtags
For this extension, we tried to incorporate a preprocessing procedure that translates emoji into their natural language description, and segments hashtags into comprehensible phrases.

## Required Libraries
* numpy
* pandas
* scikit-learn
* emoji
* wordsegment

## Usage

### Usage with Baseline SVM:
To run the baseline model with the emoji & hashtag comprehension:
1. Download the needed data and store them in a folder named `data` inside our directory.
2. In the `code` directory, run `python preprocess_baseline.py`.
3. On completion, it will print out the f1 score on testing set and generate a result file named `base_test_preds.txt` inside the `output` folder.
4. You can also use `python score.py preprocess_base_test_preds.txt` to get the testing f1 score that compares this result file with the original labels.

### Usage with BERT model
To run the BERT model with the emoji & hashtag comprehension:
1. Download the needed data.
2. Upload the data files to google colab, run the `bert.ipynb`.
3. On completion, it will print out the f1 score on both development set and testing set

To check out the scores:
Use `python score.py preprocessed_bert_test_preds.txt` inside `output` folder to get the testing f1 score that compares this result file with the original labels.

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