import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logR import LogisticRegression

# Source: https://github.com/AssemblyAI-Community

# Load breast cancer dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target # type: ignore

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Initialize the logistic regression classifier
clf = LogisticRegression(lr=0.01)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    """
    Computes the accuracy of the predictions.

    Parameters:
    y_pred (list): Predicted labels.
    y_test (numpy array): True labels.

    Returns:
    float: Accuracy as the ratio of correct predictions to total predictions.
    """
    return np.sum(y_pred == y_test) / len(y_test)

# Calculate and print the accuracy of the model
acc = accuracy(y_pred, y_test)
print(f"Accuracy: {acc:.2f}")
