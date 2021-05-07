import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import downloader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('wordnet')
# load stopwords
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

w2v = downloader.load('word2vec-google-news-300')

def preprocess(tweet):
    tweet = tweet.lower()
    tweet = re.sub("@user", "", tweet)
    tweet = re.sub(r"[^A-Za-z]", " ", tweet)
    # tweet = re.sub("url", "", tweet) # remove url
    tokens = tweet.split(" ")
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token and not token in stop_words]
    embedding = np.zeros((300,))
    if not tokens:
        return embedding
    for token in tokens:
        if token in w2v:
            embedding += w2v[token]
    return embedding

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
    X_train = pd.DataFrame(train_data.apply(lambda x: preprocess(x)).tolist())
    # Creating the development corpus
    X_dev = pd.DataFrame(dev_data.apply(lambda x: preprocess(x)).tolist())
    # Creating the testing corpus
    X_test = pd.DataFrame(test_data.apply(lambda x: preprocess(x)).tolist())

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