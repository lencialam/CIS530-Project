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
    # load train data
    train = pd.read_csv('data/olid-training-v1.0.tsv', sep="\t")
    train_data = train['tweet']
    train_labels = pd.factorize(train['subtask_a'])[0]   # OFF=0, NOT=1

    # load test data
    test_data = pd.read_csv('data/testset-levela.tsv', sep="\t")['tweet']
    test_labels = pd.factorize(pd.read_csv('data/labels-levela.csv', header=None).iloc[:,-1])[0]

    # Creating the training corpus
    train_data = train_data.apply(lambda x: preprocess(x))
    # Creating the testing corpus
    test_data = test_data.apply(lambda x: preprocess(x))

    # Transform into vectorized features
    vectorizer = CountVectorizer()
    X_train= vectorizer.fit_transform(train_data)
    X_test  = vectorizer.transform(test_data)

    # Training
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train, train_labels)

    # Evaluation
    test_preds = classifier.predict(X_test)
    # Write to result file
    with open("test_preds.txt", "w") as f:
        f.write("\n".join(map(str, test_preds)))
    # Print the fscore
    test_fscore = f1_score(test_labels, test_preds)
    print("The baseline f1 score is:", test_fscore)
    print(classification_report(test_labels, test_preds))