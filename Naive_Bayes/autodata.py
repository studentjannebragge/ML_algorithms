import pandas as pd
import numpy as np

# Luodaan kuvitteellinen autodata-dataframe, joka sisältää tuhannen auton tiedot
np.random.seed(42)  # Satunnaissiementäminen toistettavuuden takaamiseksi

data = {
    'Hinta': np.random.randint(10000, 50000, size=1000),  # Hinta 10 000 - 50 000
    'Ikä': np.random.randint(1, 15, size=1000),  # Ikä 1 - 15 vuotta
    'Kilometrit': np.random.randint(5000, 200000, size=1000),  # Kilometrit 5 000 - 200 000
    'Teho (hv)': np.random.randint(70, 300, size=1000),  # Teho 70 - 300 hevosvoimaa
    'Polttoaine': np.random.choice([0, 1], size=1000),  # Polttoaine: 0 = Bensiini, 1 = Diesel
    'Vaihteisto': np.random.choice([0, 1], size=1000),  # Vaihteisto: 0 = Automaatti, 1 = Manuaali
    'Luokka': np.random.choice([0, 1, 2], size=1000)  # Luokka: 0 = Pieni, 1 = Keskikokoinen, 2 = Iso
}

df = pd.DataFrame(data)

# Tallennetaan datasetti CSV-tiedostoksi
df.to_csv('autodata2.csv', index=False)

# Tulostetaan muutama rivi varmistaaksemme, että kaikki on kunnossa
print(df.head())