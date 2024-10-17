import numpy as np

# Source: https://github.com/AssemblyAI-Community

def sigmoid(x):
    """
    Computes the sigmoid of x.

    Parameters:
    x (numpy array): Input array or scalar.

    Returns:
    numpy array: Sigmoid of input.
    """
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    """
    A simple implementation of logistic regression classifier.

    Parameters:
    lr (float): Learning rate for gradient descent.
    n_iters (int): Number of iterations for training.

    Attributes:
    weights (numpy array): Coefficients of the features.
    bias (float): Bias term.
    """

    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initializes the logistic regression model with learning rate and iterations.
        
        Parameters:
        lr (float): Learning rate for gradient descent.
        n_iters (int): Number of iterations to run the gradient descent.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Trains the logistic regression model using gradient descent.

        Parameters:
        X (numpy array): Training data (samples x features).
        y (numpy array): Target values.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights
        self.bias = 0  # Initialize bias

        # Gradient Descent
        for _ in range(self.n_iters):
            # Linear model: X * weights + bias
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predicts binary class labels for given input data.

        Parameters:
        X (numpy array): Input data (samples x features).

        Returns:
        list: Predicted class labels (0 or 1).
        """
        linear_pred = np.dot(X, self.weights) + self.bias # type: ignore
        y_pred = sigmoid(linear_pred)
        # Convert probabilities to class labels
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred
