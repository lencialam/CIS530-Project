import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

if __name__ == "__main__":
    # Load train data
    train_df = pd.read_csv('../data/train.csv')
    train_data, train_labels = train_df["tweet"], train_df["class"]
    majority = train_labels.mode()[0]
    print("Majority class is:", majority)   # OFF=0, NOT=1

    # Ecaluate with Train
    train_preds = [majority] * len(train_data)
    train_fscore = f1_score(train_labels, train_preds, average='macro')
    print("The simple baseline training f1 score is:", train_fscore)

    # Evaluation with dev
    dev_df = pd.read_csv('../data/dev.csv')
    dev_data, dev_labels = dev_df["tweet"], dev_df["class"]
    dev_preds = [majority] * len(dev_data)
    # Write to result file
    with open("../output/simple_base_dev_preds.txt", "w") as f:
        f.write("\n".join(map(str, dev_preds)))
    # Print the fscore
    dev_fscore = f1_score(dev_labels, dev_preds, average='macro')
    print("The simple baseline development f1 score is:", dev_fscore)

    # Evaluation with test
    test_df = pd.read_csv('../data/test.csv')
    test_data, test_labels = test_df["tweet"], test_df["class"]
    test_preds = [majority] * len(test_data)
    # Write to result file
    with open("../output/simple_base_test_preds.txt", "w") as f:
        f.write("\n".join(map(str, test_preds)))
    # Print the fscore
    test_fscore = f1_score(test_labels, test_preds, average='macro')
    print("The simple baseline testing f1 score is:", test_fscore)