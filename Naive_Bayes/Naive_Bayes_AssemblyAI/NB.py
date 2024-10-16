import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        # Tallennetaan näytteiden ja piirteiden määrä
        n_samples, n_features = X.shape
        # Tallennetaan luokat ja luokkien määrä
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Lasketaan jokaiselle luokalle keskiarvo, varianssi ja priori (todennäköisyys)
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Käydään jokainen luokka läpi ja lasketaan keskiarvo, varianssi ja priori
        for idx, c in enumerate(self._classes):
            # Suodatetaan ne rivit X:stä, jotka kuuluvat luokkaan c
            X_c = X[y == c]
            # Lasketaan keskiarvo ja varianssi piirteille luokassa c
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            # Lasketaan priorin todennäköisyys luokalle c (luokan esiintymisten määrä / kaikkien näytteiden määrä)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        # Ennustetaan jokaiselle näytteelle luokka kutsumalla _predict-metodia
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Lista posteriori-todennäköisyyksistä eri luokille
        posteriors = []

        # Lasketaan posteriori-todennäköisyys jokaiselle luokalle
        for idx, c in enumerate(self._classes):
            # Lasketaan priorin logaritmi luokalle
            prior = np.log(self._priors[idx])
            # Lasketaan todennäköisyystiheysfunktion (pdf) logaritmien summa
            posterior = np.sum(np.log(self._pdf(idx, x)))
            # Lisätään priori posterioriin
            posterior = posterior + prior
            # Lisätään posteriori luokan listaan
            posteriors.append(posterior)

        # Palautetaan se luokka, jolla on suurin posteriori-todennäköisyys
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        # Haetaan keskiarvo ja varianssi luokalle
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        # Lasketaan pdf:n osoittaja (numerator)
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        # Lasketaan pdf:n nimittäjä (denominator)
        denominator = np.sqrt(2 * np.pi * var)
        # Palautetaan pdf:n arvo
        return numerator / denominator


"""
# Testaus
if __name__ == "__main__":
    # Tuodaan tarvittavat kirjastot
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # Määritellään funktio tarkkuuden laskemiseen
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Luodaan esimerkkiaineisto
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    # Jaetaan aineisto opetus- ja testidataan
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Luodaan NaiveBayes-luokan olio ja sovitetaan opetusdataan
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    # Ennustetaan testidatalle
    predictions = nb.predict(X_test)

    # Tulostetaan luokittelun tarkkuus
    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))

"""