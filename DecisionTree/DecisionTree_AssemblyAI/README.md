# Päätöspuu Classifier

Tämä arkisto sisältää Pythonilla alusta alkaen toteutetun päätöspuu-luokittelijan. Toteutus sisältää ydintoiminnallisuuden, jota tarvitaan päätöspuumallin kouluttamiseen, entropian ja informaatiovahvistuksen laskemiseen sekä ennusteiden tekemiseen uudelle datalle.

## Ominaisuudet

- Päätöspuu-luokittelijan mukautettu toteutus.
- Tukee useita pysäytysehtoja, kuten enimmäissyvyys ja solmun jakamiseen tarvittavien minimiesimerkkien määrä.
- Käyttää entropiaa informaatiovahvistuksen laskemiseen parhaiden jakojen löytämiseksi.

## Vaatimukset

- Python 3.x
- NumPy (lukuarvojen käsittelyyn)

Tarvittavan kirjaston voit asentaa seuraavalla komennolla:

```sh
pip install numpy
```

## Luokkarakenne

Koodi koostuu seuraavista luokista:

### Node-luokka

`Node`-luokka edustaa päätöspuun solmua.

- **Attribuutit:**
  - `feature`: Ominaisuuden indeksi, jota käytetään jakamiseen.
  - `threshold`: Kynnysarvo, jolla data jaetaan.
  - `left`: Viittaus vasempaan lapsisolmuun.
  - `right`: Viittaus oikeaan lapsisolmuun.
  - `value`: Solmun arvo, jos se on lehtisolmu.
- **Metodit:**
  - `is_leaf_node()`: Tarkistaa, onko solmu lehtisolmu.

### DecisionTree-luokka

`DecisionTree`-luokka käytetään päätöspuun rakentamiseen.

- **Attribuutit:**
  - `min_samples_split`: Vähimmäismäärä esimerkkejä, joita tarvitaan solmun jakamiseen.
  - `max_depth`: Puun enimmäissyvyys.
  - `n_features`: Ominaisuuksien määrä, jotka otetaan huomioon jakamisessa.
  - `root`: Päätöspuun juurisolmu.
- **Metodit:**
  - `fit(X, y)`: Sovittaa päätöspuun annettuun dataan.
  - `_grow_tree(X, y, depth)`: Kasvattaa päätöspuuta rekursiivisesti.
  - `_best_split(X, y, feat_idxs)`: Etsii parhaan ominaisuuden ja kynnyksen jakamiselle.
  - `_information_gain(y, X_column, threshold)`: Laskee informaatiovahvistuksen tietylle jaolle.
  - `_split(X_column, split_thresh)`: Jakaa datan annetun kynnyksen perusteella.
  - `_entropy(y)`: Laskee annettujen luokkien entropian.
  - `_most_common_label(y)`: Löytää yleisimmän luokan annetuista luokista.
  - `predict(X)`: Ennustaa annettujen syötteiden luokat.
  - `_traverse_tree(x, node)`: Kulkee puun läpi tehdäkseen ennusteen yksittäiselle esimerkille.

## Käyttöesimerkki

Tässä on yksinkertainen esimerkki `DecisionTree`-luokan käytöstä:

```python
import numpy as np
from decision_tree import DecisionTree

# Esimerkkidata
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]) # X sisältää neljä esimerkkiä, joilla on kaksi ominaisuutta
y = np.array([0, 1, 0, 1]) # y sisältää näiden esimerkkien luokat (0 tai 1)

# Alustetaan ja koulutetaan päätöspuu
model = DecisionTree(max_depth=3)
model.fit(X, y)

# Tehdään ennusteita
predictions = model.predict(X)
print("Ennusteet:", predictions)
```

## Lähde

How to implement Decision Trees from scratch with Python. AssemblyAI. (https://www.youtube.com/watch?v=NxEHSAfFlK8&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=5)

