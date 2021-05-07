import sys
import pandas as pd
from sklearn.metrics import f1_score

test_file = "base_test_preds.txt"

if len(sys.argv) > 2:
    print("Wrong number of parameters, only one file at a time.")
elif len(sys.argv) == 2:
    test_file = sys.argv[1]

if "dev_" in test_file: 
    # deal with dev set results as well
    original_file = "../data/dev.csv"
else: 
    # default to test again test
    original_file = "../data/test.csv"

# Test
test_labels = pd.read_csv(original_file)["class"]
with open(test_file, "r") as f:
    test_preds = [int(pred) for pred in f.read().strip().split("\n")]
fscore = f1_score(test_labels, test_preds, average='macro')
print("The F1 score is:", fscore)