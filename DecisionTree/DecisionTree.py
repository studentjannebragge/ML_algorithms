import pandas as pd
import yfinance as yf
import random

# Hae S&P 500 -osakkeiden ticker-symbolit
sp500_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
sp500_data = pd.read_csv(sp500_url)

# Listaa S&P 500 -yhtiöiden ticker-symbolit
tickers = sp500_data['Symbol'].tolist()

# Määritä aikaväli, jolta haluat hakea dataa
start_date = '2023-01-01'
end_date = '2023-12-31'

# Lataa osakedata suoraan muistissa käsiteltäväksi
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Lasketaan prosentuaalinen tuotto ja haetaan markkina-arvot
returns = {}  # Säilyttää osakkeiden tuottojen laskut
market_caps = {}  # Säilyttää osakkeiden markkina-arvot

# Käydään läpi jokainen osake ja lasketaan tuotto sekä haetaan markkina-arvo
for ticker in tickers:
    try:
        # Lasketaan historiallinen tuotto aloitus- ja lopetushintojen perusteella
        start_price = data[ticker]['Adj Close'].iloc[0]
        end_price = data[ticker]['Adj Close'].iloc[-1]
        percentage_return = ((end_price - start_price) / start_price) * 100
        returns[ticker] = percentage_return
        
        # Haetaan markkina-arvo (Market Cap) yfinance-kirjaston avulla
        stock_info = yf.Ticker(ticker).info
        market_caps[ticker] = stock_info.get('marketCap', None)  # Jos ei ole arvoa, aseta None
        
    except Exception as e:
        # Jos data puuttuu tai jotain menee pieleen, jatka seuraavaan osakkeeseen
        continue

# Muutetaan tuotto- ja markkina-arvot pandas DataFrameiksi
returns_df = pd.DataFrame(list(returns.items()), columns=['Ticker', 'Return'])
market_cap_df = pd.DataFrame(list(market_caps.items()), columns=['Ticker', 'MarketCap'])

# Yhdistetään tuotto- ja markkina-arvot yhdeksi DataFrameksi
data_df = pd.merge(returns_df, market_cap_df, on='Ticker')

# Poistetaan rivit, joilta puuttuu markkina-arvo
data_df = data_df.dropna()

# Funktio luokittelee tuoton perusteella neljään eri kategoriaan
def categorize_return(return_value):
    if return_value > 30:
        return 1  # Yli 30% tuotto
    elif 30 >= return_value > 10:
        return 2  # 30% - 10% tuotto
    elif 10 >= return_value >= 0:
        return 3  # 10% - 0% tuotto
    else:
        return 4  # Negatiivinen tuotto

# Lisätään uusi sarake, jossa on tuottoluokat
data_df['ReturnCategory'] = data_df['Return'].apply(categorize_return)

# Funktio, joka jakaa datan koulutus- ja testijoukkoihin
def train_test_split_manual(X, y, test_size=0.3, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    # Yhdistetään selittävät muuttujat ja luokat
    data = list(zip(X, y))
    
    # Sekoitetaan data satunnaisesti
    random.shuffle(data)

    # Jaetaan data koulutus- ja testiosuuksiin
    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    return list(X_train), list(X_test), list(y_train), list(y_test)

# Ominaisuudet ja luokat muutetaan listoiksi
X = data_df[['MarketCap', 'Return']].values.tolist()
y = data_df['ReturnCategory'].tolist()

# Jaetaan data manuaalisesti koulutus- ja testidataan
X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.3, random_state=42)

# Funktio jakaa datasetin kahteen osaan perustuen indeksiin ja arvoon
def split_dataset(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Gini-impurityn laskeminen ryhmille
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        # Laske luokkien osuus ryhmässä
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion ** 2
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Funktio etsii datasetin parhaan mahdollisen jakokohdan
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

# Funktio luo lopullisen päätöksen (lehden) ryhmän perusteella
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Jatketaan puun jakamista, kunnes ehdot täyttyvät (maksimisyyvyys tai minimiryhmäkoko)
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

# Rakennetaan päätöspuu
def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Ennustetaan päätöspuun perusteella
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

# Ennustetaan testidatan perusteella
def decision_tree_predict(tree, X_test):
    predictions = []
    for row in X_test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions

# Laske tarkkuus vertaamalla todellisia luokkia ennusteisiin
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

# Tulostetaan muutama rivi tuloksista, jossa näkyvät ennusteet ja todelliset luokat
results = pd.DataFrame({'True Category': y_test, 'Predicted Category': y_pred})
print(results.head())
