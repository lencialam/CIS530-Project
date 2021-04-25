import pandas as pd
from sklearn.metrics import f1_score

# Test
test_labels = pd.factorize(pd.read_csv('data/labels-levela.csv', header=None).iloc[:,-1])[0]
with open("test_preds.txt", "r") as f:
    test_preds = [int(pred) for pred in f.read().strip().split("\n")]
fscore = f1_score(test_preds, test_labels)
print("The F1 score is:", fscore)