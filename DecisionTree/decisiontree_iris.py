# %%

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Ladataan esimerkkidata
data = load_iris()
X, y = data.data, data.target

# Päätöspuun malli
model = DecisionTreeClassifier()
model.fit(X, y)

# Piirretään päätöspuu
plt.figure(figsize=(12, 8))
tree.plot_tree(model, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()
# %%
