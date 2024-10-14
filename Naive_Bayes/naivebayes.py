import numpy as np
import pandas as pd

# Lataa datasetti (korvaa 'your_dataset.csv' omalla dataset-tiedostollasi)
data = pd.read_csv('autodata2.csv')

# Oletetaan, että viimeinen sarake on kohde
X = data.iloc[:, :-1].values  # Ominaisuudet
y = data.iloc[:, -1].values   # Kohde

# Jaa datasetti koulutus- ja testijoukkoihin
def split_train_test(X, y, test_size=0.2, random_seed=None):
    # Jos satunnaissiementä annetaan, asetetaan se
    if random_seed is not None:
        np.random.seed(random_seed)
    # Luodaan indeksit ja sekoitetaan ne satunnaisesti
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    # Määritetään jakokohta, jonka mukaan data jaetaan koulutus- ja testijoukkoihin
    split_point = int(X.shape[0] * (1 - test_size))
    # Jaetaan ominaisuudet ja kohteet koulutus- ja testijoukkoihin
    X_train, X_test = X[indices[:split_point]], X[indices[split_point:]]
    y_train, y_test = y[indices[:split_point]], y[indices[split_point:]]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_seed=42)

# Gaussin Naive Bayes -toteutus
class GaussianNB:
    def fit(self, X, y):
        # Tallennetaan luokat, jotka ovat datassa
        self.classes = np.unique(y)
        self.means = {}  # Sanakirja keskiarvoille
        self.variances = {}  # Sanakirja variansseille
        self.priors = {}  # Sanakirja ennakkotodennäköisyyksille
        
        # Lasketaan jokaiselle luokalle keskiarvot, varianssit ja ennakkotodennäköisyydet
        for c in self.classes:
            X_c = X[y == c]  # Valitaan kaikki rivit, jotka kuuluvat tiettyyn luokkaan
            self.means[c] = np.mean(X_c, axis=0)  # Lasketaan keskiarvo jokaiselle ominaisuudelle
            self.variances[c] = np.var(X_c, axis=0)  # Lasketaan varianssi jokaiselle ominaisuudelle
            self.priors[c] = X_c.shape[0] / X.shape[0]  # Lasketaan ennakkotodennäköisyys
    
    def _gaussian_probability(self, x, mean, variance):
        # Gaussin todennäköisyyslaskenta yksittäiselle arvolle
        eps = 1e-6  # Siloitustermi jakamisen nollalla välttämiseksi
        factor = 1.0 / np.sqrt(2.0 * np.pi * variance + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance + eps))
        return factor * exponent
    
    def predict(self, X):
        # Ennustetaan kohdearvot kaikille riveille
        y_pred = [self._predict_instance(x) for x in X]
        return np.array(y_pred)
    
    def _predict_instance(self, x):
        # Ennustetaan yksittäisen havainnon kohde
        posteriors = []
        
        # Lasketaan posterioritodennäköisyydet jokaiselle luokalle
        for c in self.classes:
            prior = np.log(self.priors[c])  # Lasketaan ennakkotodennäköisyyden logaritmi
            class_conditional = np.sum(np.log(self._gaussian_probability(x, self.means[c], self.variances[c])))  # Lasketaan luokkaehdollinen todennäköisyys
            posterior = prior + class_conditional  # Yhdistetään ennakkotodennäköisyys ja luokkaehdollinen todennäköisyys
            posteriors.append(posterior)
        
        # Palautetaan luokka, jolla on suurin posterioritodennäköisyys
        return self.classes[np.argmax(posteriors)]

# Luodaan Gaussin Naive Bayes -malli
nb_model = GaussianNB()

# Koulutetaan malli
nb_model.fit(X_train, y_train)

# Tehdään ennusteet
y_pred = nb_model.predict(X_test)

# Arvioidaan malli
def accuracy(y_true, y_pred):
    # Lasketaan tarkkuus vertaamalla oikeita arvoja ennustettuihin arvoihin
    return np.sum(y_true == y_pred) / len(y_true)

def classification_report(y_true, y_pred):
    # Luodaan luokitusraportti, joka sisältää tarkkuuden, palautuksen ja F1-pisteet
    classes = np.unique(y_true)
    report = ""
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))  # Oikein positiiviset
        fp = np.sum((y_true != c) & (y_pred == c))  # Väärin positiiviset
        fn = np.sum((y_true == c) & (y_pred != c))  # Väärin negatiiviset
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Tarkkuus
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Palautus
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # F1-pisteet
        report += f"Luokka {c} -> Tarkkuus: {precision:.2f}, Palautus: {recall:.2f}, F1-pisteet: {f1:.2f}\n"
    return report

accuracy_value = accuracy(y_test, y_pred)
print(f'Tarkkuus: {accuracy_value:.2f}')
print('Luokitusraportti:')
print(classification_report(y_test, y_pred))