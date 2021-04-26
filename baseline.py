import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('wordnet')
# load stopwords
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(tweet):
    tweet = tweet.lower()
    tweet = tweet.replace(r"@user", "")
    tweet = tweet.replace(r"@[\w\-]+", "")
    tweet = tweet.replace(r"[^A-Za-z]", " ")
    # remove url
    tweet = tweet.replace(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "")
    tokens = tweet.split(" ")
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if not token in stop_words]
    return " ".join(tokens)

if __name__ == "__main__":
    # Load train data
    train_df = pd.read_csv('data/train.csv')
    train_data, train_labels = train_df["tweet"], train_df["class"]
    # load dev data
    dev_df = pd.read_csv('data/dev.csv')
    dev_data, dev_labels = dev_df["tweet"], dev_df["class"]
    # load test data
    test_df = pd.read_csv('data/test.csv')
    test_data, test_labels = test_df["tweet"], test_df["class"]

    # Creating the training corpus
    train_data = train_data.apply(lambda x: preprocess(x))
    # Creating the development corpus
    dev_data = dev_data.apply(lambda x: preprocess(x))
    # Creating the testing corpus
    test_data = test_data.apply(lambda x: preprocess(x))

    # Transform into vectorized features
    vectorizer = CountVectorizer()
    X_train= vectorizer.fit_transform(train_data)
    X_dev = vectorizer.transform(dev_data)
    X_test  = vectorizer.transform(test_data)

    # Training
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train, train_labels)

    # Evaluation with dev
    dev_preds = classifier.predict(X_dev)
    # Write to result file
    with open("dev_preds.txt", "w") as f:
        f.write("\n".join(map(str, dev_preds)))
    # Print the fscore
    dev_fscore = f1_score(dev_labels, dev_preds, average='macro')
    print("The baseline development f1 score is:", dev_fscore)
    print(classification_report(dev_labels, dev_preds))

    # Evaluation with test
    test_preds = classifier.predict(X_test)
    # Write to result file
    with open("test_preds.txt", "w") as f:
        f.write("\n".join(map(str, test_preds)))
    # Print the fscore
    test_fscore = f1_score(test_labels, test_preds, average='macro')
    print("The baseline testing f1 score is:", test_fscore)
    print(classification_report(test_labels, test_preds))