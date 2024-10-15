import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def euclidean_distance(x1, x2):
    """
    Laskee Euklidisen etäisyyden kahden pisteen välillä.

    Parametrit:
    x1, x2: np.ndarray
        Kaksi pistettä, joiden välinen etäisyys lasketaan.

    Palauttaa:
    float
        Euklidinen etäisyys pisteiden välillä.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    """
    KMeans-klusterointialgoritmi.

    Parametrit:
    K: int
        Klustereiden määrä.
    max_iters: int
        Suurin sallittu iteraatioiden määrä.
    """

    def __init__(self, K=5, max_iters=100):
        """
        Alustaa KMeans-luokan.

        Parametrit:
        K: int
            Klustereiden määrä.
        max_iters: int
            Suurin sallittu iteraatioiden määrä.
        """
        self.K = K
        self.max_iters = max_iters
        # Lista jokaiselle klusterille, jossa on siihen kuuluvat pisteet
        self.clusters = [[] for _ in range(self.K)]
        # Lista centroidipisteille
        self.centroids = []
        # Historia klustereista ja centroidipisteistä animaatiota varten
        self.history = []

    def predict(self, X):
        """
        Suorittaa KMeans-klusteroinnin syötedatalle.

        Parametrit:
        X: np.ndarray
            Syötedata, jossa on n_samples riviä ja n_features saraketta.

        Palauttaa:
        np.ndarray
            Jokaiselle pisteelle sen klusterin indeksi.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Alustetaan centroidit satunnaisilla pisteillä datasta
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        self.history.append((self.clusters, self.centroids.copy()))

        # Optimoidaan klusterit
        for _ in range(self.max_iters):
            # Liitetään pisteet lähimpään centroidiin (luodaan klusterit)
            self.clusters = self._create_clusters(self.centroids)

            # Tallennetaan klusterit ja centroidit animaatiota varten
            self.history.append((self.clusters.copy(), self.centroids.copy()))

            # Lasketaan uudet centroidit klustereiden perusteella
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Tarkistetaan, onko konvergoitu (eli centroidit eivät enää muutu)
            if self._is_converged(centroids_old, self.centroids):
                break

            # Tallennetaan klusterit ja centroidit animaatiota varten
            self.history.append((self.clusters.copy(), self.centroids.copy()))

        # Palautetaan jokaiselle pisteelle sen klusterin indeksi
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        """
        Antaa jokaiselle pisteelle sen klusterin indeksin.

        Parametrit:
        clusters: list
            Lista klustereista, joissa jokaisessa on klusteriin kuuluvien pisteiden indeksit.

        Palauttaa:
        np.ndarray
            Jokaiselle pisteelle sen klusterin indeksi.
        """
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        """
        Liittää pisteet lähimpään centroidiin.

        Parametrit:
        centroids: list
            Lista centroidipisteistä.

        Palauttaa:
        list
            Lista klustereista, joissa jokaisessa on klusteriin kuuluvien pisteiden indeksit.
        """
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        """
        Etsii lähimmän centroidin pisteelle.

        Parametrit:
        sample: np.ndarray
            Piste, jonka lähin centroidi etsitään.
        centroids: list
            Lista centroidipisteistä.

        Palauttaa:
        int
            Lähimmän centroidin indeksi.
        """
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        """
        Laskee klustereiden keskiarvon ja asettaa ne uusiksi centroidipisteiksi.

        Parametrit:
        clusters: list
            Lista klustereista, joissa jokaisessa on klusteriin kuuluvien pisteiden indeksit.

        Palauttaa:
        np.ndarray
            Uudet centroidipisteet.
        """
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        """
        Tarkistaa, onko centroidit konvergoituneet.

        Parametrit:
        centroids_old: list
            Edellisen iteraation centroidipisteet.
        centroids: list
            Nykyisen iteraation centroidipisteet.

        Palauttaa:
        bool
            True, jos centroidit ovat konvergoituneet, muuten False.
        """
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def animate(self):
        """
        Luo animaation klusteroinnin etenemisestä.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        def update(frame):
            # Päivitetään kuva joka iteraation jälkeen
            ax.clear()
            clusters, centroids = self.history[frame]

            # Piirretään klusterit
            for i, index in enumerate(clusters):
                points = self.X[index].T
                ax.scatter(*points)

            # Piirretään centroidit
            for point in centroids:
                ax.scatter(*point, marker="x", color="black", linewidth=2)

            ax.set_title(f"Iteration {frame + 1}")

        # Luodaan animaatio
        ani = animation.FuncAnimation(fig, update, frames=len(self.history), interval=600, repeat=False) # type: ignore
        plt.show()

# Testaus
if __name__ == "__main__":
    np.random.seed(42)
    from sklearn.datasets import make_blobs

    # Luodaan testidata, jossa on 10 klusteria
    X, y = make_blobs( # type: ignore
        centers=10, n_samples=500, n_features=2, shuffle=True, random_state=40
    )

    clusters = len(np.unique(y))

    # Luodaan ja sovitetaan KMeans-malli dataan
    k = KMeans(K=clusters, max_iters=150)
    y_pred = k.predict(X)

    # Animoidaan klusteroinnin eteneminen
    k.animate()