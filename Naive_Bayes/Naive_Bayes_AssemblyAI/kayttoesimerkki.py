from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from NB import NaiveBayes

# Luodaan esimerkkiaineisto
X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)

# Jaetaan aineisto opetus- ja testidataan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Luodaan NaiveBayes-luokan olio ja sovitetaan opetusdataan
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Ennustetaan testidatalle ja tulostetaan tarkkuus
predictions = nb.predict(X_test)
accuracy = np.sum(y_test == predictions) / len(y_test)
print("Naive Bayes classification accuracy", accuracy)