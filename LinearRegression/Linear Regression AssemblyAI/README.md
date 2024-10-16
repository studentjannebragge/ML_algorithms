# Lineaarisen regressioin toteutus ja animaatio

Tämä projekti esittelee yksinkertaisen lineaarisen regressioin toteutuksen Pythonilla. Lisäksi se sisältää skriptin, joka visualisoi lineaarisen regressiomallin oppimisprosessin animaationa käyttäen `matplotlib`-kirjastoa. Koodipohja koostuu kahdesta pääskriptistä:

1. `lr.py`: Lineaarisen regressioin perusimplementaatio alusta alkaen.
2. `train.py`: Skripti, joka kouluttaa lineaarisen regressiomallin generoituun dataan ja animoi oppimisprosessin.

## Tiedostot

### 1. `lr.py`
Tämä tiedosto sisältää `LinearRegression`-luokan toteutuksen, joka suorittaa yksinkertaisen lineaarisen regressioin gradienttilaskennan avulla.

- **Luokka: `LinearRegression`**
  - `__init__(self, lr=0.001, n_iters=1000)`: Alustaa mallin oppimisnopeudella (`lr`) ja iteraatioiden määrällä (`n_iters`).
  - `fit(self, X, y)`: Kouluttaa mallin käyttämällä gradienttilaskentaa syötedatalle `X` ja sen arvoille `y`.
  - `predict(self, X)`: Ennustaa kohdearvot annetuille syötetiedoille `X`.

Toteutus olettaa, että syötedata on 2D-muodossa (n_samples, n_features).

### 2. `train.py`
Tämä tiedosto hoitaa lineaarisen regressiomallin koulutuksen syntetisoidulla datalla, joka on luotu `sklearn.datasets.make_regression`-funktiolla. Lisäksi se käyttää `matplotlib`-kirjastoa regressioviivan visualisoimiseen ja animaatioon oppimisprosessin aikana.

- **Keskeiset vaiheet:**
  - Syntetisoidun datan generointi: Datasetti generoidaan `make_regression`-funktiolla.
  - Datan jakaminen: Datasetti jaetaan koulutus- ja testijoukkoihin käyttämällä `train_test_split`-funktiota.
  - Visualisointi: Hajontakaavio luodaan datan visualisoimiseksi, ja lineaarisen regressioviivan tilaa päivitetään iteratiivisesti näyttämään, miten malli oppii ajan myötä.
  - Animaatio: `matplotlib.animation.FuncAnimation`-toimintoa käytetään oppimisprosessin animointiin.

### Käyttöohjeet

#### Vaatimukset

- Python 3.x
- `numpy`
- `scikit-learn`
- `matplotlib`

Voit asentaa tarvittavat paketit pipillä:

```bash
pip install numpy scikit-learn matplotlib
```
## Koodin suoritus

### Mallin kouluttaminen:

Voit kouluttaa mallin ja nähdä animaation suorittamalla `train.py`-skriptin:

```bash
python train.py
```

Tämä luo kuvaajan, joka näyttää koulutus- ja testidatan sekä animoidun regressioviivan, joka esittää mallin oppimisprosessia.


