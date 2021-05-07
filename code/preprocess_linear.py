import re
import emoji
import wordsegment
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

wordsegment.load()

def preprocess(tweet):
    tweet = tweet.lower()
    # limit consecutive @user
    tweet = re.sub(r"(@user ){3,}", "@user @user @user ", tweet)
    # replace "url" with "html" for embedding
    tweet = tweet.replace("url", "html")
    # translate emoji into words
    tweet = emoji.demojize(tweet)
    # segment hashtag & emoji translations
    tokens = []
    for token in tweet.split(" "):
        # # Remove all non-alphanumeric characters
        # tokens += wordsegment.segment(token)
        # Only deal with translated emoji & hashtags
        if re.match(r"(:[a-z_-]+:)|(#[a-z]+)", token):
            tokens += wordsegment.segment(token)
        else:
            tokens.append(token)
    return " ".join(tokens)

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

    # Creating the training corpus
    train_data = train_data.apply(lambda x: preprocess(x))
    # Creating the development corpus
    dev_data = dev_data.apply(lambda x: preprocess(x))
    # Creating the testing corpus
    test_data = test_data.apply(lambda x: preprocess(x))

    # Transform into vectorized features
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    X_dev = vectorizer.transform(dev_data)
    X_test = vectorizer.transform(test_data)

    # Training
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train, train_labels)

    # Evaluation with dev
    dev_preds = classifier.predict(X_dev)
    # Write to result file
    with open("../output/preprocess_dev_preds.txt", "w") as f:
        f.write("\n".join(map(str, dev_preds)))
    # Print the fscore
    dev_fscore = f1_score(dev_labels, dev_preds, average='macro')
    print("The baseline development f1 score is:", dev_fscore)
    print(classification_report(dev_labels, dev_preds))

    # Evaluation with test
    test_preds = classifier.predict(X_test)
    # Write to result file
    with open("../output/preprocess_test_preds.txt", "w") as f:
        f.write("\n".join(map(str, test_preds)))
    # Print the fscore
    test_fscore = f1_score(test_labels, test_preds, average='macro')
    print("The baseline testing f1 score is:", test_fscore)
    print(classification_report(test_labels, test_preds))