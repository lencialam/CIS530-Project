import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('labeled_data.csv', index_col=0)

train_df, test_df = train_test_split(df, test_size=0.2)
dev_df, test_df = train_test_split(test_df, test_size=0.5)

train_df.to_csv("train.csv", index=False)
dev_df.to_csv("dev.csv", index=False)
test_df.to_csv("test.csv", index=False)