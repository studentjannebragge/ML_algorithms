# Logistic Regression Koneoppimismalli

Tämä projekti toteuttaa logistisen regressiomallin Python-koodilla. Se käyttää NumPy-kirjastoa matemaattisten laskutoimitusten suorittamiseen ja hyödyntää Breast Cancer -datasettiä ennustetarkkuuden testaamiseksi. Koodin päätiedostot ovat `logR.py` ja `train.py`.

## Tiedostot

### `logR.py`
Tiedosto sisältää `LogisticRegression`-luokan, joka toteuttaa logistisen regressioalgoritmin. Tämän luokan avulla voidaan sovittaa malli annettuihin datajoukkoihin ja tehdä ennusteita uusista datanäytteistä. Luokka toimii seuraavasti:
- **Sigmoid-funktio**: Muuntaa mallin lineaariset ennusteet todennäköisyyksiksi.
- **Parametrien oppiminen**: Malli oppii käyttämällä gradientin laskeutumista, jossa painot ja bias päivitetään iteratiivisesti oppimisnopeuden (learning rate) perusteella.
- **Ennustus**: Mallin avulla voidaan tehdä luokittelupäätöksiä ennustetun todennäköisyyden perusteella.

Koodi luokassa sisältää seuraavat tärkeät toiminnot:
- `fit(X, y)`: Sovittaa mallin annettuihin piirteisiin ja kohdeluokkiin.
- `predict(X)`: Ennustaa luokan uusille havainnoille käyttäen koulutettua mallia.

### `train.py`
Tämä tiedosto sisältää koodin, joka lataa Breast Cancer -datan, jakaa sen koulutus- ja testijoukkoihin, kouluttaa logistisen regressiomallin, ja arvioi mallin ennustustarkkuuden. Se käyttää sklearn-kirjastoa datan jakamiseen ja ladataan Breast Cancer -datan seuraavasti:
- **Datan lataaminen**: Breast Cancer -data ladataan sklearn-kirjaston avulla.
- **Datan jakaminen**: Data jaetaan koulutus- ja testijoukkoihin suhteessa 80/20.
- **Mallin koulutus**: Logistisen regression malli koulutetaan `X_train` ja `y_train` datalla.
- **Tarkkuuden laskenta**: Mallin tarkkuus lasketaan ennustettujen ja todellisten arvojen perusteella.

Koodissa käytetään `accuracy`-funktiota, joka palauttaa mallin tarkkuuden.

## Asennus

Tämän projektin suorittamiseksi tarvitset Pythonin sekä seuraavat kirjastot:
- `numpy`
- `sklearn`
- `matplotlib`

Voit asentaa kirjastot seuraavalla komennolla:

```bash
pip install numpy scikit-learn matplotlib

```
## Käyttö

1. Lataa tarvittavat tiedostot:
   - `logR.py`
   - `train.py`

2. Varmista, että sinulla on tarvittavat kirjastot asennettuna:

   ```bash
   pip install numpy scikit-learn matplotlib

    ```

3. Suorita kooditiedosto `train.py`:

   ```bash
   python train.py

   ```

4. Ohjelma suorittaa seuraavat toiminnot:
   - Lataa Breast Cancer -datasetin sklearn-kirjastosta.
   - Jakaa datan koulutus- ja testijoukkoihin suhteessa 80/20.
   - Kouluttaa logistisen regressiomallin koulutusdatalla.
   - Ennustaa testijoukon luokat käyttäen koulutettua mallia.
   - Laskee mallin tarkkuuden vertaamalla ennusteita todellisiin luokkiin.
   - Tulostaa mallin ennustustarkkuuden testidatalle.

5.  Ohjelman tulostama tarkkuus on luku, joka edustaa prosenttiosuutta oikein luokitelluista testidatan näytteistä verrattuna todellisiin luokkiin. Tarkkuus on arvo välillä 0 ja 1, missä 1 tarkoittaa 100 % tarkkuutta.

## Lähde
https://www.youtube.com/watch?v=YYEJ_GUguHw&t=1s