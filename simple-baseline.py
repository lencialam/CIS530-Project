import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

if __name__ == "__main__":
    # Load train data
    train = pd.read_csv('data/olid-training-v1.0.tsv', sep="\t")
    train_data = train['tweet']
    train_labels = pd.factorize(train['subtask_a'])[0]
    majority = stats.mode(train_labels)[0][0]
    print("Majority class is:", majority)   # OFF=0, NOT=1

    # Load test data
    test_data = pd.read_csv('data/testset-levela.tsv', sep="\t")['tweet']
    test_labels = pd.factorize(pd.read_csv('data/labels-levela.csv', header=None).iloc[:,-1])[0]
    # Test
    test_preds = [majority] * len(test_data)
    # Write to result file
    with open("test_preds.txt", "w") as f:
        f.write("\n".join(map(str, test_preds)))
    # Print the fscore
    test_fscore = f1_score(test_labels, test_preds)
    print("The simple baseline f1 score is:", test_fscore)