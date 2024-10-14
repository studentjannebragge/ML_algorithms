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

# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ladataan data ja jaetaan se koulutus- ja testijoukkoihin
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Päätöspuu ilman leikkaamista (ylisovitettu)
model_full = DecisionTreeClassifier()
model_full.fit(X_train, y_train)
y_pred_full = model_full.predict(X_test)
print(f"Täyden puun tarkkuus: {accuracy_score(y_test, y_pred_full):.2f}")

# Päätöspuu, jossa maksimisyvyys on rajoitettu (leikkaus)
model_pruned = DecisionTreeClassifier(max_depth=3)
model_pruned.fit(X_train, y_train)
y_pred_pruned = model_pruned.predict(X_test)
print(f"Leikatun puun tarkkuus: {accuracy_score(y_test, y_pred_pruned):.2f}")

# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ladataan Iris-data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Leikattujen mallien vertailu
parametrien_yhdistelmat = [
    {"max_depth": 3},
    {"min_samples_split": 4},
    {"min_samples_leaf": 5}
]

for params in parametrien_yhdistelmat:
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Leikkausparametrit {params} - Tarkkuus: {accuracy_score(y_test, y_pred):.2f}")


# %%
import matplotlib.pyplot as plt
from sklearn import tree

# Täysi päätöspuu ilman leikkaamista
full_tree = DecisionTreeClassifier()
full_tree.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
tree.plot_tree(full_tree, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("Täysi päätöspuu")
plt.show()

# Leikattu päätöspuu (syvyys rajoitettu)
pruned_tree = DecisionTreeClassifier(max_depth=3)
pruned_tree.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
tree.plot_tree(pruned_tree, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("Leikattu päätöspuu (max_depth=3)")
plt.show()

# %%
import numpy as np

# Eri maksimi syvyyksiä
syvyyksiä = np.arange(1, 10)

# Tallennetaan koulutus- ja testivirheet
train_errors = []
test_errors = []

for depth in syvyyksiä:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, y_train)
    
    # Lasketaan virheet
    train_errors.append(1 - accuracy_score(y_train, model.predict(X_train)))
    test_errors.append(1 - accuracy_score(y_test, model.predict(X_test)))

# Piirretään virheet
plt.plot(syvyyksiä, train_errors, label='Koulutusvirhe', marker='o')
plt.plot(syvyyksiä, test_errors, label='Testivirhe', marker='o')
plt.xlabel('Puun maksimi syvyys')
plt.ylabel('Virheprosentti')
plt.legend()
plt.title('Puun syvyyden vaikutus koulutus- ja testivirheisiin')
plt.show()


# %%
from sklearn.model_selection import cross_val_score

# Eri syvyyksien testaus
for depth in [3, 5, None]:  # None tarkoittaa täyttä syvyyttä ilman leikkaamista
    model = DecisionTreeClassifier(max_depth=depth)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Max_depth={depth} - Cross Validation - Keskimääräinen tarkkuus: {scores.mean():.2f}")

# %%
