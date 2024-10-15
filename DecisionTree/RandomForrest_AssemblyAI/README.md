# Päätöspuu classfier ja Random Forest

Tämä arkisto sisältää Pythonilla alusta alkaen toteutetun päätöspuu-luokittelijan sekä satunnaismetsä-mallin. Toteutus sisältää ydintoiminnallisuuden, jota tarvitaan päätöspuumallin ja satunnaismetsän kouluttamiseen, entropian ja informaatiovahvistuksen laskemiseen sekä ennusteiden tekemiseen uudelle datalle.

## Ominaisuudet

- Päätöspuu- ja satunnaismetsä-luokittelijoiden mukautetut toteutukset.
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

### RandomForest-luokka

`RandomForest`-luokka edustaa satunnaismetsää, joka koostuu useista päätöspuista.

- **Attribuutit:**
  - `n_trees`: Puun määrä satunnaismetsässä.
  - `max_depth`: Yksittäisten puiden enimmäissyvyys.
  - `min_samples_split`: Vähimmäismäärä esimerkkejä, joita tarvitaan solmun jakamiseen.
  - `n_features`: Ominaisuuksien määrä, jotka otetaan huomioon jakamisessa.
  - `trees`: Lista päätöspuista, jotka muodostavat satunnaismetsän.
- **Metodit:**
  - `fit(X, y)`: Sovittaa satunnaismetsän annettuun dataan kasvattamalla useita päätöspuita.
  - `_bootstrap_samples(X, y)`: Luo satunnaisen otoksen alkuperäisestä datasta päätöspuun kouluttamista varten.
  - `_most_common_label(y)`: Löytää yleisimmän luokan annetuista luokista.
  - `predict(X)`: Ennustaa annettujen syötteiden luokat käyttämällä kaikkien päätöspuiden ennusteita.

## Vaiheittainen selitys koodin toiminnasta

1. **Päätöspuun luominen (`DecisionTree`):**
   - `DecisionTree`-luokka luo päätöspuun mallin, joka oppii jakamaan datan sen ominaisuuksien perusteella. 
   - `fit(X, y)`-metodia käytetään mallin sovittamiseen annettuun dataan. Se aloittaa puun kasvattamisen kutsumalla `_grow_tree(X, y, depth)`-metodia, joka jakaa dataa rekursiivisesti perustuen informaatiovahvistukseen.
   - `_best_split(X, y, feat_idxs)`-metodi löytää parhaan ominaisuuden ja kynnyksen, jotka tuottavat suurimman informaatiovahvistuksen, kun solmu jaetaan. Puuta kasvatetaan niin kauan, kunnes pysäytysehdot täyttyvät (esim. maksimisyvyys saavutetaan).
   - Ennusteet tehdään kulkemalla puun läpi `_traverse_tree(x, node)`-metodilla ja määrittämällä, mikä luokka lehtisolmussa on.

2. **Satunnaismetsän luominen (`RandomForest`):**
   - `RandomForest`-luokka luo satunnaismetsän, joka koostuu useista päätöspuista. Tämä parantaa mallin tarkkuutta ja yleistettävyyttä.
   - `fit(X, y)`-metodia käytetään kasvattamaan useita päätöspuita. Jokaista päätöspuuta varten luodaan satunnainen otos alkuperäisestä datasta `_bootstrap_samples(X, y)`-metodilla, mikä auttaa estämään ylisovittamista.
   - `predict(X)`-metodi ennustaa annettujen syötteiden luokat ottamalla huomioon kaikkien päätöspuiden ennusteet ja valitsemalla yleisimmän luokan jokaiselle esimerkille.

3. **Esimerkkidata ja käyttö:**
   - `X` sisältää esimerkkidatan, jossa jokaisella esimerkillä on kaksi ominaisuutta.
   - `y` sisältää luokat, jotka vastaavat kutakin esimerkkiä (`0` tai `1`).
   - `RandomForest`-malli luodaan määrittelemällä puiden määrä (`n_trees`) ja puiden enimmäissyvyys (`max_depth`), minkä jälkeen malli sovitetaan dataan `fit(X, y)`-metodilla.
   - Lopuksi `predict(X)`-metodia käytetään tekemään ennusteita, jotka tulostetaan käyttäjälle.


## Käyttöesimerkki

Tässä on yksinkertainen esimerkki `RandomForest`-luokan käytöstä:

```python
import numpy as np
from random_forest import RandomForest

# Esimerkkidata
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([0, 1, 0, 1])

# Alustetaan ja koulutetaan satunnaismetsä
model = RandomForest(n_trees=5, max_depth=3)
model.fit(X, y)

# Tehdään ennusteita
predictions = model.predict(X)
print("Ennusteet:", predictions)
```

## Lähde

How to implement Random Forest from scratch with Python. AssemblyAI. (https://www.youtube.com/watch?v=kFwe2ZZU7yw)