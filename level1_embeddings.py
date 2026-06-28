"""
Уровень 1: обучаемый эмбеддинг "с нуля" на numpy.

Идея: E -- таблица (n_ягод, dim). Учим её предсказывать признаки ягоды.
Градиент обновляет строки E, и похожие по признакам ягоды стягиваются вместе.
Это ровно то, что делает nn.Embedding в torch, только видно каждую гайку.
"""
import numpy as np

np.random.seed(0)

# --- Данные: ягода -> бинарные признаки -------------------------------------
# признаки: [красная, тёмно-синяя/чёрная, кислая, растёт_низко, сложная(костянки)]
berries = [
    "клубника", "земляника", "малина", "ежевика", "черника",
    "голубика", "брусника", "клюква", "крыжовник", "смородина",
]
feat_names = ["красная", "тёмная", "кислая", "низкая", "костянки"]
Y = np.array([
    [1, 0, 0, 1, 0],  # клубника
    [1, 0, 0, 1, 0],  # земляника  (почти как клубника)
    [1, 0, 0, 0, 1],  # малина
    [0, 1, 0, 0, 1],  # ежевика    (костянки, как малина)
    [0, 1, 0, 1, 0],  # черника
    [0, 1, 0, 0, 0],  # голубика
    [1, 0, 1, 1, 0],  # брусника
    [1, 0, 1, 1, 0],  # клюква     (почти как брусника)
    [0, 0, 1, 0, 0],  # крыжовник
    [0, 1, 1, 0, 0],  # смородина
], dtype=float)

n, n_feat = Y.shape
dim = 8

# --- Параметры: таблица эмбеддингов E + линейный классификатор (W, b) -------
E = np.random.randn(n, dim) * 0.3      # <- ВОТ ЭТО и есть эмбеддинги
W = np.random.randn(dim, n_feat) * 0.3
b = np.zeros(n_feat)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

lr = 0.5
for epoch in range(4000):
    # forward (full-batch: все ягоды сразу)
    z = E @ W + b           # (n, n_feat)
    p = sigmoid(z)
    eps = 1e-9
    loss = -np.mean(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps))

    # backward (ручной бэкпроп)
    dz = (p - Y) / n        # grad BCE по z
    dW = E.T @ dz
    db = dz.sum(0)
    dE = dz @ W.T           # grad по строкам таблицы эмбеддингов

    W -= lr * dW
    b -= lr * db
    E -= lr * dE            # обновляем САМИ эмбеддинги

    if epoch % 1000 == 0:
        print(f"epoch {epoch:4d}  loss {loss:.4f}")

# --- Похожесть: косинус между эмбеддингами -----------------------------------
En = E / np.linalg.norm(E, axis=1, keepdims=True)
sim = En @ En.T

def top_neighbors(name, k=2):
    i = berries.index(name)
    order = np.argsort(-sim[i])
    return [(berries[j], round(float(sim[i, j]), 2)) for j in order if j != i][:k]

print("\nБлижайшие соседи по выученным эмбеддингам:")
for name in ["клубника", "малина", "брусника", "голубика"]:
    print(f"  {name:10s} -> {top_neighbors(name)}")

# --- Визуализация в 2D (PCA) -------------------------------------------------
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xy = PCA(n_components=2).fit_transform(E)
plt.figure(figsize=(7, 6))
plt.scatter(xy[:, 0], xy[:, 1], s=40)
for i, name in enumerate(berries):
    plt.annotate(name, (xy[i, 0], xy[i, 1]), fontsize=11,
                 xytext=(5, 4), textcoords="offset points")
plt.title("Уровень 1: эмбеддинги ягод (PCA в 2D)")
plt.tight_layout()
plt.savefig("./pictures/level1_pca.png", dpi=130)
print("\nГрафик сохранён: level1_pca.png")
