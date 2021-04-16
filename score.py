import pandas as pd
from sklearn.metrics import f1_score

# Test
test_labels = pd.read_csv("test.csv")["class"]
with open("test_preds.txt", "r") as f:
    test_preds = [int(pred) for pred in f.read().strip().split("\n")]
fscore = f1_score(test_preds, test_labels, average='micro')
print("The F1 score is:", fscore)