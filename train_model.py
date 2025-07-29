import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the training data
df = pd.read_csv("mitbih_train.csv", header=None)

# Rename last column as 'label'
df.rename(columns={187: "label"}, inplace=True)

# Convert labels to binary: 0 = normal, 1 = abnormal
df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

# Split into features and target
X = df.drop("label", axis=1)
y = df["label"]

# Split into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))

# Save the trained model
joblib.dump(clf, "ekg_model.pkl", compress=3)
print("✅ Model saved to ekg_model.pkl")
