import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lr import LinearRegression

# Luodaan data
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)  # type: ignore
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Alustetaan kuva
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_train, y_train, color="blue", s=10, label="Train data")
ax.scatter(X_test, y_test, color="red", s=10, label="Test data")
line, = ax.plot([], [], color='black', linewidth=2, label='Prediction')

ax.set_xlim(X.min(), X.max())
ax.set_ylim(y.min(), y.max())
ax.legend()

# Animaation funktio
def animate(i):
    reg = LinearRegression(lr=0.01)
    reg.fit(X_train[:i+1], y_train[:i+1])  # Käytetään vain osaa datasta koulutukseen
    y_pred_line = reg.predict(X)
    line.set_data(X, y_pred_line)
    return line,

# Luodaan animaatio
anim = FuncAnimation(fig, animate, frames=len(X_train), interval=100)

plt.show()

