import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_likelihoods = defaultdict(lambda: defaultdict(lambda: 1))  # Use Laplace smoothing
        self.classes = []

    def train(self, X, y):
        # X is the training data (features), y is the labels (classes)
        self.classes = np.unique(y)  # Unique class labels
        total_instances = len(y)
        
        # Calculate prior probabilities P(C) for each class
        for c in self.classes:
            class_count = np.sum(y == c)
            self.class_priors[c] = class_count / total_instances
        
        # Calculate likelihood P(x|C) for each feature given a class
        for c in self.classes:
            # Get the instances for class c
            instances_in_class = X[y == c]
            total_features_in_class = len(instances_in_class)
            
            # For each feature, calculate likelihood
            for j in range(X.shape[1]):
                feature_values, counts = np.unique(instances_in_class[:, j], return_counts=True)
                for value, count in zip(feature_values, counts):
                    # Likelihood with Laplace smoothing
                    self.feature_likelihoods[j][(value, c)] = (count + 1) / (total_features_in_class + len(feature_values))

    def predict(self, X):
        predictions = []
        for instance in X:
            posteriors = {}
            # For each class, calculate the posterior probability
            for c in self.classes:
                posterior = np.log(self.class_priors[c])  # Start with prior
                for j, feature_value in enumerate(instance):
                    # Multiply by the likelihood for each feature
                    if (feature_value, c) in self.feature_likelihoods[j]:
                        posterior += np.log(self.feature_likelihoods[j][(feature_value, c)])
                    else:
                        # If unseen feature value, apply smoothing
                        posterior += np.log(1 / (len(self.classes) + 1))
                posteriors[c] = posterior
            # Choose the class with the highest posterior probability
            predictions.append(max(posteriors, key=posteriors.get))
        return predictions

# Example usage:

# Training data: X (features), y (labels)
X_train = np.array([
    [1, 'S'],
    [1, 'M'],
    [1, 'M'],
    [1, 'S'],
    [0, 'S'],
    [0, 'M'],
    [0, 'M'],
    [0, 'L'],
    [0, 'L'],
])

y_train = np.array(['T', 'T', 'F', 'F', 'F', 'F', 'T', 'T', 'T'])

# Create and train Naive Bayes model
nb = NaiveBayes()
nb.train(X_train, y_train)

# New data to predict
X_test = np.array([
    [0, 'S'],
    [1, 'M'],
])

# Make predictions
predictions = nb.predict(X_test)
print(predictions)
