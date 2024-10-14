*ChatGPT learning*

Tässä Python-koodissa rakennetaan päätöspuu S&P 500 -yhtiöiden historiallisten tuottojen ja markkina-arvojen perusteella. Käydään koodi vaiheittain läpi:

### 1. S&P 500 -osakkeiden tietojen lataaminen
```python
import pandas as pd
import yfinance as yf
import random

# Hae S&P 500 -osakkeiden ticker-symbolit
sp500_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
sp500_data = pd.read_csv(sp500_url)

# Listaa S&P 500 -yhtiöiden ticker-symbolit
tickers = sp500_data['Symbol'].tolist()
```
Tässä ladataan S&P 500 -yhtiöiden tiedot DataHubista, ja tallennetaan osakkeiden ticker-symbolit listalle.

### 2. Osaketietojen lataaminen ja tuottojen laskeminen
```python
# Määritä aikaväli, jolta haluat hakea dataa
start_date = '2023-01-01'
end_date = '2023-12-31'

# Lataa osakedata suoraan muistissa käsiteltäväksi
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
```
Tässä käytetään `yfinance`-kirjastoa historiallisten osakekurssien lataamiseen aikaväliltä 1.1.2023 - 31.12.2023.

### 3. Tuottojen ja markkina-arvojen laskeminen
```python
returns = {}
market_caps = {}

for ticker in tickers:
    try:
        # Lasketaan historiallinen tuotto aloitus- ja lopetushintojen perusteella
        start_price = data[ticker]['Adj Close'].iloc[0]
        end_price = data[ticker]['Adj Close'].iloc[-1]
        percentage_return = ((end_price - start_price) / start_price) * 100
        returns[ticker] = percentage_return

        # Haetaan markkina-arvo yfinance-kirjaston avulla
        stock_info = yf.Ticker(ticker).info
        market_caps[ticker] = stock_info.get('marketCap', None)
    except Exception as e:
        continue
```
Osakkeiden tuotto lasketaan vertailtaessa aloitus- ja päätöshintoja. Lisäksi haetaan markkina-arvo (Market Cap) jokaiselle osakkeelle.

### 4. Tuottojen ja markkina-arvojen yhdistäminen DataFrameksi
```python
# Muutetaan tuotto- ja markkina-arvot pandas DataFrameiksi
returns_df = pd.DataFrame(list(returns.items()), columns=['Ticker', 'Return'])
market_cap_df = pd.DataFrame(list(market_caps.items()), columns=['Ticker', 'MarketCap'])

# Yhdistetään tuotto- ja markkina-arvot yhdeksi DataFrameksi
data_df = pd.merge(returns_df, market_cap_df, on='Ticker')

# Poistetaan rivit, joilta puuttuu markkina-arvo
data_df = data_df.dropna()
```
Tuotto- ja markkina-arvot yhdistetään yhdeksi DataFrameksi, ja puuttuvat tiedot poistetaan.

### 5. Tuottojen luokittelu kategorioihin
```python
def categorize_return(return_value):
    if return_value > 30:
        return 1
    elif 30 >= return_value > 10:
        return 2
    elif 10 >= return_value >= 0:
        return 3
    else:
        return 4

data_df['ReturnCategory'] = data_df['Return'].apply(categorize_return)
```
Tässä lisätään uusi sarake, jossa osakkeiden tuotot luokitellaan neljään eri kategoriaan.

### 6. Manuaalinen koulutus- ja testijoukon jako
```python
def train_test_split_manual(X, y, test_size=0.3, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    data = list(zip(X, y))
    random.shuffle(data)

    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    return list(X_train), list(X_test), list(y_train), list(y_test)

X = data_df[['MarketCap', 'Return']].values.tolist()
y = data_df['ReturnCategory'].tolist()

X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.3, random_state=42)
```
Koulutus- ja testijoukot jaetaan manuaalisesti `train_test_split_manual`-funktion avulla.

### 7. Päätöspuun rakentaminen
```python
def split_dataset(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion ** 2
        gini += (1.0 - score) * (size / n_instances)
    return gini

def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = split_dataset(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left)
        split(node['left'], max_depth, min_size, depth+1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right)
        split(node['right'], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root
```
Tässä määritellään päätöspuun rakentamiseen liittyvät toiminnot, kuten parhaan jakokohdan etsiminen ja puun jakaminen.

### 8. Ennustaminen ja arviointi
```python
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def decision_tree_predict(tree, X_test):
    predictions = []
    for row in X_test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions

def calculate_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / float(len(y_true)) * 100.0

# Yhdistetään koulutusdata (X_train, y_train)
train_data = [list(X_train[i]) + [y_train[i]] for i in range(len(X_train))]

# Rakennetaan päätöspuu
tree = build_tree(train_data, max_depth=3, min_size=1)

# Ennustetaan testidatalla
y_pred = decision_tree_predict(tree, X_test)

# Arvioidaan tarkkuus
accuracy = calculate_accuracy(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}%")
```
Päätöspuun avulla ennustetaan testijoukon luokat ja lasketaan mallin tarkkuus.

### Raportin tulkitseminen:

- **Tarkkuuden laskeminen**: `calculate_accuracy()`-funktio vertaa todellisia luokkia (y_test) mallin ennusteisiin (y_pred) ja laskee oikein ennustettujen osakkeiden osuuden prosentteina.
  
  - Oikein ennustettujen määrä jaetaan kaikkien ennusteiden määrällä ja kerrotaan 100:lla, jotta saadaan prosentuaalinen tarkkuus:
    \[
    \text{Tarkkuus (\%)} = \left( \frac{\text{Oikein ennustetut}}{\text{Kaikki ennusteet}} \right) \times 100
    \]

- **Tulosten näyttäminen**: Tämän jälkeen tulostetaan mallin ennustama luokka ja todellinen luokka (muutamia rivejä testidatasta) taulukkomuodossa, jotta voit helposti tarkastella, kuinka hyvin ennusteet vastaavat todellisia luokkia.

### Esimerkkituloste:

```plaintext
Model Accuracy: 99.33%
   True Category  Predicted Category
0              1                   1
1              4                   4
2              1                   1
3              1                   1
4              2                   2
```

- **Model Accuracy**: Tarkkuus voi olla esimerkiksi 99,33 %, mikä tarkoittaa, että malli ennusti oikein noin 99,33 % tapauksista testidatassa.
- **True Category** ja **Predicted Category**: Tämä taulukko näyttää muutaman esimerkkirivin testidatasta, jossa näet mallin ennusteen (`Predicted Category`) ja todellisen luokan (`True Category`).

### Parannukset:

1. **Syvyyden ja minimiryhmäkoon säätö**: Voit säätää päätöspuun maksimisyvyyttä ja minimiryhmäkokoa seuraamalla, miten ne vaikuttavat tarkkuuteen. Esimerkiksi:
   - **max_depth**: Rajaa, kuinka syvälle puu voi mennä.
   - **min_size**: Määrittää pienimmän sallitun ryhmän koon, jolloin päätöspuu ei enää jaa dataa.

2. **Visualisointi**: Jos haluat päätöspuun visualisoinnin, voimme rakentaa yksinkertaisen tulostuksen näyttämään, miten data jakautuu eri solmuihin.

### Yhteenveto
Tässä koodissa käytetään päätöspuuta ennustamaan S&P 500 -osakkeiden tuottoluokkaa markkina-arvon ja tuottojen perusteella. Päätöspuun rakennus ja jakopisteen valinta tehdään manuaalisesti, ja mallin tarkkuus lasketaan vertaamalla ennustettuja arvoja todellisiin luokkiin.

Tämä koodi tarjoaa hyvän esimerkin siitä, kuinka päätöspuita voidaan hyödyntää kategoristen ennusteiden tekemisessä käyttäen historiallista osakedataa.
