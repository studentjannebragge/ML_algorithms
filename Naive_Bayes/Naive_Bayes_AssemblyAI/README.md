# Naive Bayes -Classifier

Tämä projekti sisältää yksinkertaisen Naive Bayes -luokittimen toteutuksen Python-kielellä. Luokitin perustuu Gaussin jakaumaan ja soveltuu luokittelutehtäviin, joissa halutaan erotella eri luokkia toisistaan piirteiden perusteella.

## Käyttö

Koodi sisältää `NaiveBayes`-luokan, joka toteuttaa Naive Bayes -algoritmin. Tämä luokka koostuu seuraavista metodeista:

- `fit(X, y)`: Sovittaa mallin opetusdataan, eli laskee jokaiselle luokalle keskiarvot, varianssit ja priori-todennäköisyydet.
- `predict(X)`: Ennustaa annetuille syötteenä oleville piirteille luokat.

Lisäksi koodi sisältää testausosan, jossa käytetään scikit-learn-kirjaston luomaa esimerkkiaineistoa mallin toiminnan testaamiseksi.

## Riippuvuudet

Projektissa tarvittavat kirjastot:

- `numpy`: matemaattisiin laskutoimituksiin.
- `sklearn`: opetus- ja testiaineistojen luomiseen sekä niiden jakamiseen.

Voit asentaa tarvittavat kirjastot seuraavalla komennolla:

```
pip install numpy scikit-learn
```

## Käyttöesimerkki

Käyttääksesi NaiveBayes-luokitinta, voit seurata alla olevaa esimerkkiä:

```python
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
```

## Lisätietoja

Naive Bayes -algoritmi perustuu Bayesin teoreemaan ja oletukseen siitä, että piirteet ovat toisistaan riippumattomia (naive oletus). Tämä implementaatio käyttää Gaussin jakaumaa todennäköisyystiheyden laskemiseen, joten se soveltuu erityisesti tilanteisiin, joissa piirteiden jakauma on likimain normaalijakautunut.

## Huomioitavaa

Tämä toteutus on yksinkertaistettu versio Naive Bayes -luokittimesta, eikä se sisällä kaikkia optimointeja tai virheenkäsittelyä, joita tuotantotason ratkaisu vaatisi. Koodi on tarkoitettu opetus- ja demonstraatiotarkoituksiin.

## Lähde

How to implement Naive Bayes from scratch with Python. AssemblyAI (https://www.youtube.com/watch?v=TLInuAorxqE&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=7)
