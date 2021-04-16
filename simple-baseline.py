import pandas as pd

# Train
train_df = pd.read_csv("train.csv")
majority = train_df["class"].mode()[0]
print("Majority class is:", majority)

# Dev
dev_df = pd.read_csv("dev.csv")
dev_preds = [majority] * len(dev_df)
with open("dev_preds.txt", "w") as f:
    f.write("\n".join(map(str, dev_preds)))

# Test
test_df = pd.read_csv("test.csv")
test_preds = [majority] * len(test_df)
with open("test_preds.txt", "w") as f:
    f.write("\n".join(map(str, test_preds)))