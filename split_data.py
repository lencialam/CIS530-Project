import pandas as pd
from sklearn.model_selection import train_test_split

# Load train data
train = pd.read_csv('downloaded_data/olid-training-v1.0.tsv', sep="\t")
train_data = train['tweet']
train_labels = pd.Series(pd.factorize(train['subtask_a'])[0])  # OFF=0, NOT=1
train_df = pd.concat([train_data, train_labels], axis=1)
train_df.columns = ['tweet', 'class']

train_df, dev_df = train_test_split(train_df, test_size=0.1)
print("train_df:", train_df.shape)
print("dev_df:", dev_df.shape)

# Load test data
test_data = pd.read_csv('downloaded_data/testset-levela.tsv', sep="\t")['tweet']
test_labels = pd.Series(pd.factorize(pd.read_csv('downloaded_data/labels-levela.csv', header=None).iloc[:,-1])[0])
test_df = pd.concat([test_data, test_labels], axis=1)
test_df.columns = ['tweet', 'class']
print("test_df:", test_df.shape)

train_df.to_csv("data/train.csv", index=False)
dev_df.to_csv("data/dev.csv", index=False)
test_df.to_csv("data/test.csv", index=False)