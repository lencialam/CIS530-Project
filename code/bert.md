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
1. Download the needed data.
2. Upload the data files to google colab, run the `bert.ipynb`.
3. On completion, it will print out the f1 score on both development set and testing set

## Model Description
We use a BERT model with BERT Tokenizer, padding and masking techniques.

## Score with the BERT model
The development f1 score is: 0.7340947516674741
The testing f1 score is: 0.7917311135898011

## Parameter Tuning
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
| optimizer | development | test
| ------------- | ------------- | ------------- 
|AdamW, lr = 1e-6, eps = 1e-8| 0.72 | 0.771(epoch=6)
|AdamW, lr = 1e-6, eps = 1e-6| 0.74 | 0.774(epoch=6)
|AdamW, lr = 1e-5, eps = 1e-8| 0.69 | 0.717(epoch=6)
|AdamW, lr = 1e-5, eps = 1e-6 | 0.74 | 0.779
|AdamW, lr = 1e-5, eps = 1e-4 | 0.75 | 0.779
|AdamW, lr = 1e-4, eps = 1e-8 | 0.71 | 0.755
|AdamW, lr = 1e-4, eps = 1e-6 | 0.71 | 0.734
|AdamW, lr = 1e-4, eps = 1e-4 | 0.73 | 0.752


## Cite
```
Chris McCormick and Nick Ryan. (2019, July 22). BERT Fine-Tuning Tutorial with PyTorch. Retrieved from http://www.mccormickml.com
```