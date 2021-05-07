import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

if __name__ == "__main__":
    # Load train data
    train_df = pd.read_csv('../data/train.csv')
    train_data, train_labels = train_df["tweet"], train_df["class"]
    # load dev data
    dev_df = pd.read_csv('../data/dev.csv')
    dev_data, dev_labels = dev_df["tweet"], dev_df["class"]
    # load test data
    test_df = pd.read_csv('../data/test.csv')
    test_data, test_labels = test_df["tweet"], test_df["class"]

    # Transform into vectorized features
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    X_dev = vectorizer.transform(dev_data)
    X_test = vectorizer.transform(test_data)

    # Training
    classifier = svm.LinearSVC()
    classifier.fit(X_train, train_labels)

    # Evaluation with dev
    dev_preds = classifier.predict(X_dev)
    # Write to result file
    with open("../output/base_dev_preds.txt", "w") as f:
        f.write("\n".join(map(str, dev_preds)))
    # Print the fscore
    dev_fscore = f1_score(dev_labels, dev_preds, average='macro')
    print("The baseline development f1 score is:", dev_fscore)
    print(classification_report(dev_labels, dev_preds))

    # Evaluation with test
    test_preds = classifier.predict(X_test)
    # Write to result file
    with open("../output/base_test_preds.txt", "w") as f:
        f.write("\n".join(map(str, test_preds)))
    # Print the fscore
    test_fscore = f1_score(test_labels, test_preds, average='macro')
    print("The baseline testing f1 score is:", test_fscore)
    print(classification_report(test_labels, test_preds))