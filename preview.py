

import pandas as pd

# Load the training data
df = pd.read_csv("mitbih_train.csv", header=None)

# Show the shape and first 5 rows
print("Shape of the dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Count label values
print("\nLabel distribution:")
print(df.iloc[:, -1].value_counts())
