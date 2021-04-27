# BERT model
We used the transformers package from Hugging Face which will give us a pytorch interface for working with BERT. In Hugging Face library, there are many powerful pytorch interface for working with BERT. Here, we choose to use BertForSequenceClassification.

## Required Libraries
* numpy
* pandas
* nltk
* transformers
* torch.utils.data

## Usage
To run this baseline model:
1. Download the needed data and store them in a folder named `data`.
2. Upload the data to google colab, run the `BERT_model.ipynb`.
3. On completion, it will print out the f1 score on both development set and testing set

## Model Description
We use a BERT model with BERT Tokenizer, padding and masking techniques.

## Score with BERT model

### Development Set
The baseline development f1 score is: 0.74

### Test Set
The baseline testing f1 score is: 0.772


## Cite
```
Chris McCormick and Nick Ryan. (2019, July 22). BERT Fine-Tuning Tutorial with PyTorch. Retrieved from http://www.mccormickml.com
```