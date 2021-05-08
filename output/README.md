# Evaluation Metric

## We decided to use macro-averaged F1 score to evalutate the performance of our classifiers.

The usual metrics for evaulating text classification task use the confusion matrix, which is a specific table layout that allows visualization of the performance of an model and makes it very easy and straightforward to see which labels are being misclassified and if our classifier is returning the correct label or not. 

The confusion matrix can help calculate a series of values: 
- **True Positives** (TP, correctly predicts the positive labels)
- **False Positives** (FP, incorrectly predicts the positive labels)
- **True Negatives** (TN, correctly predicts the negative labels)
- **False Negatives** (FN, incorrectly predicts the negative labels)

And two other very useful values:
- **Presion**: TP / (TP + FP) -- the fraction of positive predictions that actually belong to the positive class
- **Recall**: TP / (TP + FN) -- the fraction of positive predictions out of all positive instances in the dataset

Since our dataset has a very imbalanced distribution, it would be improper to measure the model performance with a simple accuracy score, nor would it be proper to only look at either precision or recall. In order to optimize for both precision and recall, we could utilize the F1 score, which is the harmonic mean of these two metrics. 

The highest possible value of an F1 score is 1.0, which represents perfect precision and recall.

The lowest possible value is 0, this will occur if either the precision or the recall is zero.

**For this scoring metric, a higher score would be better.**

## Usage

To run the evaluation script, simply run:

`python score.py test_preds.txt`

where `test_preds.txt` should be the name of the target prediction result file. The F1 score of the test dataset would then be printed to the terminal.

## Stored Results

### Development Set

* `simple_base_dev_preds.txt`: prediction results for the development set with the simple baseline model in `simple-baseline.py`
* `base_dev_preds.txt`: prediction results for the development set with the simple baseline model in `simple-baseline.py`
* `lstm_dev_preds.txt`: prediction results for the development set with our first extension (LSTM model) in `lstm.ipynb`
* `bert_dev_preds.txt`: prediction results for the development set with our second extension (BERT model) in `bert.ipynb`
* `preprocess_base_dev_preds.txt`: prediction results for the development set with the published baseline model and our third extension (Semantic Representation of Emoji and Hashtags) in `preprocess_baseline.py`
* `preprocess_bert_dev_preds.txt`: prediction results for the development set with our second extension (BERT model) and our third extension (Semantic Representation of Emoji and Hashtags) in `preprocess_baseline.py`

### Testing Set

* `simple_base_test_preds.txt`: prediction results for the testing set with the simple baseline model in `simple-baseline.py`
* `base_test_preds.txt`: prediction results for the testing set with the simple baseline model in `simple-baseline.py`
* `lstm_test_preds.txt`: prediction results for the testing set with our first extension (LSTM model) in `lstm.ipynb`
* `bert_test_preds.txt`: prediction results for the testing set with our second extension (BERT model) in `bert.ipynb`
* `preprocess_base_test_preds.txt`: prediction results for the testing set with the published baseline model and our third extension (Semantic Representation of Emoji and Hashtags) in `preprocess_baseline.py`
* `preprocess_bert_test_preds.txt`: prediction results for the testing set with our second extension (BERT model) and our third extension (Semantic Representation of Emoji and Hashtags) in `preprocess_baseline.py`
