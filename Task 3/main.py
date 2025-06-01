import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Dataset URL and local paths
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
ZIP_PATH = "bank-additional.zip"
EXTRACTED_FOLDER = "bank-additional"
CSV_FILE = os.path.join(EXTRACTED_FOLDER, "bank-additional-full.csv")

# Download and extract the dataset if not already present
if not os.path.isfile(CSV_FILE):
    if not os.path.isfile(ZIP_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("Download complete.")
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall()
    print("Extraction complete.")

# Load dataset
df = pd.read_csv(CSV_FILE, sep=';')
print(f"Dataset loaded with shape: {df.shape}")

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Target variable: 'y' (yes/no) → binary encode: yes=1, no=0
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Separate features and target
X = df.drop(columns=['y'])
y = df['y']

# Encode categorical columns using LabelEncoder
cat_cols = X.select_dtypes(include=['object']).columns
print("Categorical columns to encode:", list(cat_cols))

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split data into train and test sets (80-20 stratified split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Build Decision Tree classifier with max_depth and min_samples_leaf to reduce overfitting
clf = DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=20)
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Plot feature importances
feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
feature_importances.plot(kind='bar', color='cornflowerblue')
plt.title('Feature Importances')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Plot the decision tree
plt.figure(figsize=(20,12))
plot_tree(clf,
          feature_names=X.columns,
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()

# Print the decision rules as text
tree_rules = export_text(clf, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n")
print(tree_rules)
