# KMeans Klusteroinnin Animaatio

Tämä projekti sisältää KMeans-klusterointialgoritmin Pythonilla, jossa klusteroinnin eteneminen voidaan visualisoida animaation avulla. Koodi käyttää `matplotlib`-kirjastoa luomaan animaation, joka näyttää klustereiden ja centroidien kehityksen iteraatioiden aikana.

## Kuvaus

KMeans on yksinkertainen mutta tehokas klusterointialgoritmi, jota käytetään ryhmittelemään dataa `K` ryhmään. Algoritmi pyrkii löytämään keskipisteet (centroidit) kullekin ryhmälle siten, että klustereiden sisäinen varianssi minimoituu. Tässä projektissa toteutetaan KMeans-algoritmi ja lisätään mahdollisuus visualisoida klusterointiprosessi animaation avulla.

## Koodin Tiedostot

- **kmeans\_animation.py**: Sisältää KMeans-algoritmin toteutuksen ja klusterointiprosessin visualisoinnin animaationa.

## Kuinka Käyttää

1. **Asenna riippuvuudet**: Ennen kuin suoritat koodin, varmista, että sinulla on tarvittavat kirjastot asennettuna.

   ```sh
   pip install numpy matplotlib scikit-learn
   ```

2. **Suorita Koodi**: Voit suorittaa koodin komentoriviltä seuraavasti:

   ```sh
   python kmeans_animation.py
   ```

   Tämä luo animaation, joka näyttää, kuinka KMeans-algoritmi ryhmittelee datan klustereihin.

## Toiminnot

- **`KMeans`****-luokka**: Toteuttaa KMeans-klusterointialgoritmin.

  - `__init__(K, max_iters)`: Alustaa algoritmin klustereiden lukumäärän `K` ja suurimman iteraatioiden määrän `max_iters` perusteella.
  - `predict(X)`: Suorittaa klusteroinnin syötedatalle `X`.
  - `animate()`: Luo animaation klusterointiprosessin etenemisestä.

- **Animaatio**: Animaatio näyttää, kuinka pisteet ryhmitellään klustereihin ja kuinka centroidit liikkuvat kohti optimaalisia sijaintejaan iteraatioiden aikana.

## Esimerkki

Koodi käyttää `make_blobs`-funktiota `scikit-learn`-kirjastosta luomaan synteettistä dataa, jossa on kolme klusteria. Tämän jälkeen KMeans-algoritmi sovitetaan dataan, ja animaatio näyttää prosessin askel askeleelta.

## Riippuvuudet

- **numpy**: Matemaattisia operaatioita varten.
- **matplotlib**: Animaation luomista ja visualisointia varten.
- **scikit-learn**: Testidatan luomista varten.

## Tiedoston Rakenne

- **euclidean\_distance(x1, x2)**: Laskee euklidisen etäisyyden kahden pisteen välillä.
- **KMeans**-luokka, joka sisältää seuraavat metodit:
  - `predict(X)`: Suorittaa klusteroinnin datalle `X`.
  - `_get_cluster_labels(clusters)`: Palauttaa klusterin indeksin jokaiselle pisteelle.
  - `_create_clusters(centroids)`: Luo klusterit liittämällä pisteet lähimpään centroidiin.
  - `_closest_centroid(sample, centroids)`: Etsii lähimmän centroidin annetulle pisteelle.
  - `_get_centroids(clusters)`: Laskee centroidit klustereiden perusteella.
  - `_is_converged(centroids_old, centroids)`: Tarkistaa, ovatko centroidit konvergoituneet.
  - `animate()`: Luo animaation klusterointiprosessista.

## Lähde
How to implement K-Means from scratch with Python, AssemblyAI (https://www.youtube.com/watch?v=6UF5Ysk_2gk&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd&index=11)


