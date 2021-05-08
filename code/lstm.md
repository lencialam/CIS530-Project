# Extension 1 - LSTM model
We designed a 5 layer LSTM model using GloVe_200d.

## Required Libraries
* numpy
* pandas
* nltk
* keras

## Usage
To run this model:
1. Download the needed data.
2. Upload the data files to google colab, run the `lstm.ipynb`.
3. On completion, it will print out the f1 score on both development set and testing set

To check out the scores:
Use `python score.py lstm_test_preds.txt` inside `output` folder to get the testing f1 score that compares this result file with the original labels.

## Model Description
First, we truncate and pad the input tweets so that all tweets have the same input length. In our model, we set the maximum length of all sequences to be 200, which is a reasonable length for tweets.

Our model has 5 layers. The first layer is the Embedding layer that uses 200 length vectors to represent each word. The next layer is the LSTM layer with 64 memory units (smart neurons). Next, we add a global max pooling1d layer to smooths out the model. Then, we add an optional dropout layer to help prevent overfitting, Finally, since it is a classification problem, we add a dense layer with a single neuron and a sigmoid activation function to make 0 or 1 predictions for the two classes (offensive or not offensive) in the problem.

We chose log loss as the loss function since it is a binary classification problem. We tried Adam and rmsprop optimizers, since both are efficient heuristics for the problem. 

## Score with the LSTM model
The development f1 score is: 0.0.73
The testing f1 score is: 0.77

## Parameter Tuning
Below are the results of our model using sigmoid as the activation function, with 64 memory units,training hyper-parameters batch_size = 128, max_seq_length = 200 and epochs = 5. We tuned on the Optimizer and Dropout rate:

| optimizer and drop rate | development | test
| ------------- | ------------- | ------------- 
|Adam, drop rate = N/A| 0.71 | 0.72
|Adam, drop rate = 0.1| 0.71 | 0.74
|Adam, drop rate = 0.5| 0.73 | 0.77
|rmsprop, drop rate = N/A | 0.67 | 0.69
|rmsprop, drop rate = 0.1 | 0.72 | 0.75
|rmsprop, drop rate = 0.5 | 0.73 | 0.76


## Cite
```
https://github.com/talhaanwarch/Offensive-Language-Detection
```