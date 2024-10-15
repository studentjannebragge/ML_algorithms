import numpy as np
from RandomForest import RandomForest

# Esimerkkidata
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([0, 1, 0, 1])

# Alustetaan ja koulutetaan satunnaismetsä
model = RandomForest(n_trees=5, max_depth=3)
model.fit(X, y)

# Tehdään ennusteita
predictions = model.predict(X)
print("Ennusteet:", predictions)