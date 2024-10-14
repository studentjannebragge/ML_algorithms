Tässä koodissa on useita funktioita, jotka yhdessä toteuttavat Gaussin Naive Bayes -luokittimen. Koodi koostuu muun muassa datan jakamisesta koulutus- ja testijoukkoihin, mallin opettamisesta, sekä mallin avulla tehtävistä ennusteista ja arvioinnista.

Ensimmäinen merkittävä funktio on `split_train_test`, joka jakaa datan koulutus- ja testijoukkoihin. Tämä funktio sekoittaa indeksit satunnaisesti ja jakaa datan haluttuun osaan koulutusta ja testiä varten, mahdollistaen mallin oikeanlaisen arvioinnin.

Seuraavaksi on `GaussianNB`-luokka, joka toteuttaa Gaussin Naive Bayes -mallin. Luokassa on useita metodeja, kuten `fit`, `predict`, `_predict_instance`, ja `_gaussian_probability`. 

- `fit`-metodi kouluttaa mallin. Se laskee kullekin luokalle keskiarvot, varianssit ja ennakkotodennäköisyydet, jotka ovat mallin perusta. Näitä arvoja käytetään luokittelussa.
  
- `_gaussian_probability` laskee yksittäisen ominaisuuden todennäköisyyden Gaussin jakauman avulla, mikä on olennainen osa Naive Bayesin laskentaa.

- `predict`-metodi tekee ennusteita annetuille syötteille. Se hyödyntää `_predict_instance`-metodia, joka laskee jokaiselle luokalle todennäköisyyden (posterioritodennäköisyyden) ja valitsee suurimman todennäköisyyden omaavan luokan.

Lopuksi koodissa on kaksi arviointifunktiota: `accuracy`, joka laskee ennusteen tarkkuuden vertaamalla oikeita ja ennustettuja arvoja, sekä `classification_report`, joka muodostaa luokitteluraportin, sisältäen tietoja kuten tarkkuuden, palautuksen ja F1-pisteet.

Yhdessä nämä funktiot muodostavat kokonaisuuden, joka mahdollistaa datan käsittelyn, mallin opettamisen ja sen arvioimisen.

## Koodin antama raportti lyhyesti

Naive Bayes -koodin testituloksia arvioidaan tulostettujen "Tarkkuus" ja "Luokitusraportti" -tietojen perusteella. Tässä on ohje, kuinka voit tulkita näitä tuloksia:

1. **Tarkkuus (Accuracy)**:
   - Tarkkuus kertoo, kuinka monta prosenttia ennusteista oli oikein. Se lasketaan jakamalla oikein ennustettujen tapausten määrä kaikilla tapauksilla.
   - Esimerkiksi, jos tulostettu tarkkuus on `0.85`, tämä tarkoittaa, että 85 % ennusteista oli oikein. Mitä suurempi tarkkuus, sitä parempi malli on onnistunut tehtävässään.

2. **Luokitusraportti (Classification Report)**:
   Luokitusraportti sisältää tarkempia tietoja kunkin luokan ennusteiden laadusta. Se sisältää seuraavat mittarit:

   - **Tarkkuus (Precision)**:
     - Tarkkuus kertoo, kuinka moni ennustetuista tietyn luokan tapauksista oli oikein.
     - Se lasketaan kaavalla: oikein positiiviset (True Positives, TP) / (oikein positiiviset + väärin positiiviset (False Positives, FP)).
     - Korkea tarkkuus tarkoittaa, että mallilla on vähän virheellisiä positiivisia ennusteita.

   - **Palautus (Recall)**:
     - Palautus kertoo, kuinka moni tietyn luokan todellisista tapauksista löydettiin oikein.
     - Se lasketaan kaavalla: oikein positiiviset (TP) / (oikein positiiviset + väärin negatiiviset (False Negatives, FN)).
     - Korkea palautus tarkoittaa, että malli löytää suurimman osan tietyn luokan tapauksista.

   - **F1-pisteet (F1-score)**:
     - F1-pisteet ovat tarkkuuden ja palautuksen painotettu keskiarvo. Se lasketaan kaavalla: 2 * (tarkkuus * palautus) / (tarkkuus + palautus).
     - F1-pisteet ovat hyödyllisiä, kun halutaan tasapaino tarkkuuden ja palautuksen välillä, erityisesti silloin kun datassa on epätasapaino eri luokkien välillä.

   - **Luokat**:
     - Raportissa on listattuna kaikki luokat (esimerkiksi `0: Pieni auto`, `1: Keskikokoinen auto`, `2: Iso auto`), ja näille annetaan tarkkuus, palautus ja F1-pisteet.
     - Tämä antaa käsityksen siitä, miten malli suoriutui eri luokkien ennustamisesta. Jos jotkut luokat saavat matalat pisteet, voi olla tarpeen kerätä lisää dataa tai harkita toisenlaista mallia.

Yhteenvetona: Tarkkuus antaa yleiskuvan mallin suorituskyvystä, kun taas luokitusraportti tarjoaa syvällisemmän analyysin siitä, miten hyvin malli toimii kussakin luokassa. Korkeat tarkkuus-, palautus- ja F1-arvot viittaavat hyvään ennustekykyyn. Jos jokin näistä arvoista on alhainen, se saattaa viitata mallin heikkouteen kyseisessä luokassa, ja se voi olla merkki mallin optimointitarpeesta.

# Esimerkkitulos ja sen analyysi

## raportti antoi seuraavat tulokset:

Luokka Pieni auto -> Tarkkuus: 0.22, Palautus: 0.11, F1-pisteet: 0.15
Luokka Keskikokoinen auto -> Tarkkuus: 0.27, Palautus: 0.25, F1-pisteet: 0.26
Luokka Iso auto -> Tarkkuus: 0.31, Palautus: 0.51, F1-pisteet: 0.39

Tässä raportissa on luokitteluindikaattorit kolmelle eri autoluokalle: "Pieni auto", "Keskikokoinen auto" ja "Iso auto". Analysoidaan tarkemmin, mitä tarkkuus, palautus ja F1-pisteet tarkoittavat tässä yhteydessä:

### 1. **Tarkkuus (Precision)**
Tarkkuus kuvaa sitä, kuinka monta mallin tekemää ennustetta tietystä luokasta oli oikeasti oikein kyseiseen luokkaan kuuluvia. Se antaa siis tiedon siitä, kuinka moni mallin ennustama "positiivinen" oli todella positiivinen.

- **"Pieni auto" -> Tarkkuus: 0.22**: Mallin tekemistä "Pieni auto" -ennusteista vain 22 % oli oikein.
- **"Keskikokoinen auto" -> Tarkkuus: 0.27**: Mallin tekemistä "Keskikokoinen auto" -ennusteista 27 % oli oikein.
- **"Iso auto" -> Tarkkuus: 0.31**: Mallin tekemistä "Iso auto" -ennusteista 31 % oli oikein.

Korkea tarkkuus tarkoittaa, että mallilla on vähän väärin positiivisia ennusteita, kun taas alhainen tarkkuus viittaa siihen, että malli tekee usein virheellisiä positiivisia ennusteita.

### 2. **Palautus (Recall)**
Palautus kuvaa sitä, kuinka moni tietyn luokan todellisista tapauksista malli löysi oikein. Se siis mittaa mallin kykyä löytää kaikki kyseisen luokan esimerkit.

- **"Pieni auto" -> Palautus: 0.11**: Malli tunnisti vain 11 % kaikista "Pieni auto" -tapauksista oikein.
- **"Keskikokoinen auto" -> Palautus: 0.25**: Malli tunnisti 25 % kaikista "Keskikokoinen auto" -tapauksista oikein.
- **"Iso auto" -> Palautus: 0.51**: Malli tunnisti 51 % kaikista "Iso auto" -tapauksista oikein.

Korkea palautus tarkoittaa, että malli löytää suurimman osan tietyn luokan tapauksista, kun taas alhainen palautus tarkoittaa, että malli jättää monia kyseisen luokan tapauksia huomiotta.

### 3. **F1-pisteet (F1-score)**
F1-pisteet ovat tarkkuuden ja palautuksen harmoninen keskiarvo. F1-pisteet ovat hyödyllisiä silloin, kun halutaan tasapaino tarkkuuden ja palautuksen välillä. Se auttaa arvioimaan mallia erityisesti, kun datassa on epätasapaino eri luokkien välillä.

- **"Pieni auto" -> F1-pisteet: 0.15**: Pieni F1-arvo viittaa siihen, että mallin suorituskyky on heikko sekä tarkkuuden että palautuksen osalta tässä luokassa.
- **"Keskikokoinen auto" -> F1-pisteet: 0.26**: Kohtalainen F1-arvo viittaa siihen, että mallin suorituskyky on parantunut, mutta on silti riittämätön, jotta malli olisi luotettava tässä luokassa.
- **"Iso auto" -> F1-pisteet: 0.39**: Tämä on korkein F1-piste, mikä tarkoittaa, että malli onnistui tässä luokassa paremmin kuin muissa, mutta suorituskyky on silti keskinkertainen.

### Yhteenveto
- **Pieni auto**: Mallin kyky ennustaa "Pieni auto" -luokkaa on heikko. Sekä tarkkuus että palautus ovat alhaisia, mikä tarkoittaa, että malli tekee virheellisiä ennusteita ja jättää monia tapauksia huomiotta.
- **Keskikokoinen auto**: Tämä luokka on hieman parempi kuin "Pieni auto", mutta tarkkuus ja palautus ovat silti matalia, mikä johtaa matalaan F1-arvoon.
- **Iso auto**: Mallin suorituskyky on paras tässä luokassa, mutta F1-pisteet (0.39) osoittavat, että malli tarvitsee vielä parannuksia ennustustarkkuuden ja palautuksen suhteen.

Näiden tulosten perusteella mallia voi parantaa esimerkiksi keräämällä lisää dataa, säätämällä mallin hyperparametreja tai kokeilemalla monimutkaisempia koneoppimismalleja, kuten päätöspuu- tai satunnaismetsäluokittimia.