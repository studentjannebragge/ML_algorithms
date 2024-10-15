# K-Nearest Neighbors (KNN) Toteutus

Tämä repositorio sisältää yksinkertaisen toteutuksen k-Nearest Neighbors (KNN) -algoritmista Pythonilla. Toteutus on kirjoitettu alusta alkaen, jotta voidaan havainnollistaa KNN:n perusperiaatteita ilman ulkoisia kirjastoja, kuten `scikit-learn`.

## Yleiskatsaus

K-Nearest Neighbors (KNN) on valvottu oppimisalgoritmi, jota käytetään sekä luokittelu- että regressiotehtäviin. Se toimii etsimällä `k` lähintä datan pistettä (naapuria) syötenäytteeseen ja määrittämällä tuloksen enemmistöluokan (luokittelussa) tai keskiarvon (regressiossa) perusteella. Tässä toteutuksessa keskitytään luokitteluongelmaan.

Perusidea on laskea etäisyys syötenäytteen ja jokaisen harjoitusnäytteen välillä Euklidisen etäisyyden kaavaa käyttäen, sitten valita `k` lähintä naapuria ja käyttää enemmistöäänestystä luokan ennustamiseen.

## Koodin rakenne

### Tiedostot
- **knn.py**: Sisältää KNN-luokittelijan toteutuksen.
- **train.py**: Lataa Iris-datasarjan, jakaa datan harjoitus- ja testijoukkoihin, visualisoi datan ja käyttää KNN-luokittelijaa ennusteiden tekemiseen.

### KNN-luokka
KNN-toteutus koostuu seuraavista osista:

1. **`__init__(self, k=3)`**: Alustaa luokittelijan naapureiden lukumäärällä `k`. Oletuksena `k` on asetettu arvoon 3.

2. **`fit(self, X, y)`**: Tallentaa harjoitusdatan piirteet (`X`) ja luokat (`y`) myöhempää ennustamista varten.

3. **`predict(self, X)`**: Ennustaa annettujen syötteiden (`X`) luokat. Tämä metodi käyttää sisäisesti `_predict`-metodia ennustaakseen jokaisen näytteen erikseen.

4. **`_predict(self, x)`**: Ennustaa yhden syötenäytteen (`x`) luokan etsimällä `k` lähintä naapuria ja käyttämällä enemmistöäänestystä ennustetun luokan määrittämiseen.

### Euklidinen etäisyys
- **`euclidean_distance(x1, x2)`** -funktio laskee Euklidisen etäisyyden kahden pisteen välillä, ja sitä käytetään määrittämään syötenäytteen ja harjoitusnäytteiden välinen etäisyys.

## Käyttö

Voit käyttää tätä KNN-toteutusta ajamalla `iris_dataset_load.py`-skriptin. Skripti seuraa näitä vaiheita:

1. Lataa Iris-datasarjan, joka on yleisesti käytetty datasarja luokittelutehtäviin.
2. Jakaa datasarjan harjoitus- ja testijoukkoihin.
3. Visualisoi datasarjan hajontakaaviolla.
4. Kouluttaa KNN-luokittelijan harjoitusdatan avulla.
5. Tekee ennusteita testijoukosta ja tulostaa tulokset.

### Esimerkki
```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from knn import KNN

# Lataa Iris-datasarja
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Jaa data harjoitus- ja testijoukkoihin
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Alusta ja kouluta KNN-luokittelija
clf = KNN(k=5)
clf.fit(X_train, y_train)

# Tee ennusteita
ennusteet = clf.predict(X_test)
print(ennusteet)
```

## Vaatimukset
- Python 3.x
- NumPy
- Matplotlib (datan visualisointiin)
- scikit-learn (datasarjojen käsittelyyn)

## Kuinka ajaa
1. Kloonaa repositorio.
2. Asenna tarvittavat paketit komennolla `pip install -r requirements.txt`.
3. Aja `train.py`-skripti nähdäksesi KNN-luokittelijan toiminnassa.

## Lähde
How to implement KNN from scratch with Python, AssemblyAI (https://www.youtube.com/watch?v=rTEtEy5o3X0) 15.10.2024
