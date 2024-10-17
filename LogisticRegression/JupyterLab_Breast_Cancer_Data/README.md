
# Rintasyövän ennustaminen logistisella regressiolla

## Johdanto

Tässä projektissa opetellaan kouluttamaan logistista regressiomallia, jonka avulla voidaan luokitella rintasyövän kasvaimet joko pahanlaatuisiksi (M) tai hyvänlaatuisiksi (B) solumittauksien perusteella. Logistinen regressio on yksinkertainen ja tehokas koneoppimismenetelmä kaksiluokkaisiin luokittelutehtäviin, mikä tekee siitä sopivan valinnan tähän ongelmaan.

### Aineisto

Tässä projektissa käytettävä aineisto sisältää solujen yksityiskohtaisia mittauksia, kuten säde, rakenne, ympärysmitta ja pinta-ala. Näiden mittausten perusteella pyritään ennustamaan diagnoosi.

- **Lähde**: [Breast Cancer Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- **Luokiteltava muuttuja**: `diagnosis` (M = Pahanlaatuinen, B = Hyvänlaatuinen)
- **Ominaisuudet**: Erilaisia solun ominaisuuksia, kuten:
  - `radius_mean`
  - `texture_mean`
  - `perimeter_mean`
  - `area_mean`
  - Ja monia muita.

## Projektin rakenne

1. **Datan lataaminen**: Aineisto ladataan ja siistitään analyysiä varten.
2. **Tutkiva data-analyysi**: Aineistoa tutkitaan yksityiskohtaisesti, jotta ymmärretään ominaisuuksien jakaumat.
3. **Mallin rakentaminen**: Logistinen regressiomalli koulutetaan aineistolla ennustamaan, onko kasvain pahanlaatuinen vai hyvänlaatuinen.
4. **Mallin arviointi**: Mallin suorituskykyä arvioidaan tavallisilla mittareilla, kuten tarkkuudella, tarkkuus (precision) ja muistilla (recall).

## Aloittaminen

Seuraa näitä ohjeita projektin suorittamiseksi:

### Esivaatimukset

Varmista, että sinulla on asennettuna Python ja tarvittavat kirjastot:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Notebookin suorittaminen

1. Kloonaa tämä repositorio:
   ```bash
   git clone <repository-url>
   ```
2. Avaa ja suorita Jupyter-notebook:
   ```bash
   jupyter notebook Breast_Cancer_LogR.ipynb
   ```

## Tulokset

Logistinen regressiomalli saavuttaa korkean tarkkuuden rintasyövän kasvainten ennustamisessa, perustaen annettuihin mittaustietoihin.

## Lisenssi

Tämä projekti on lisensoitu MIT-lisenssillä.
