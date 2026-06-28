"""
Этап 1: Skip-gram (Word2Vec) с нуля на numpy.

Идея в одном предложении:
  у каждого слова есть вектор; мы двигаем векторы так, чтобы у слов,
  встречающихся в похожих контекстах, векторы стали похожими.

Здесь нет torch — две матрицы W_in и W_out и есть наши "эмбеддинги".
Это ровно то, что делает nn.Embedding, только всё видно руками.
"""

import re
import numpy as np

np.random.seed(0)

# ──────────────────────────────────────────────────────────────────────
# 1. Корпус. Пока крошечный и придуманный, но с осмысленными контекстами:
#    лесные ягоды соседствуют с "лес/собирать", садовые — с "сад/грядка",
#    красные — со словом "красная" и т.д. Этого хватит, чтобы увидеть структуру.
# ──────────────────────────────────────────────────────────────────────
corpus = [
    "клубника растёт в саду на грядке и бывает сладкая красная",
    "малина растёт в саду сладкая красная её кладут в варенье",
    "садовая клубника крупная сладкая красная ягода с грядки",
    "малина и клубника садовые сладкие ягоды для варенья",
    "черника растёт в лесу её собирают синяя лесная ягода",
    "голубика лесная синяя ягода её собирают в лесу как чернику",
    "черника и голубика синие лесные ягоды их собирают в лесу",
    "брусника красная лесная ягода её собирают в лесу осенью",
    "ежевика тёмная сладкая ягода растёт в лесу и в саду",
    "смородина чёрная и красная садовая ягода с куста кислая",
    "крыжовник зелёная кислая садовая ягода растёт на кусте",
    "из малины и клубники варят сладкое варенье",
    "из черники и голубики делают начинку синяя лесная ягода",
    "красную смородину и крыжовник собирают в саду на кусте",
    "лесную чернику и бруснику собирают в лесу",
]

# ──────────────────────────────────────────────────────────────────────
# 2. Токенизация и словарь
# ──────────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"[а-яё]+", text.lower())

tokens_per_sent = [tokenize(s) for s in corpus]
all_tokens = [t for sent in tokens_per_sent for t in sent]

from collections import Counter
counts = Counter(all_tokens)
vocab = sorted(counts)                       # детерминированный порядок
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
V = len(vocab)
print(f"Размер словаря: {V} слов")

# ──────────────────────────────────────────────────────────────────────
# 3. Обучающие пары (center -> context) в окне ±window
#    Это и есть "сигнал из контекста" — берём его прямо из текста, без меток.
# ──────────────────────────────────────────────────────────────────────
WINDOW = 2
pairs = []
for sent in tokens_per_sent:
    idxs = [word2idx[t] for t in sent]
    for pos, center in enumerate(idxs):
        lo = max(0, pos - WINDOW)
        hi = min(len(idxs), pos + WINDOW + 1)
        for ctx_pos in range(lo, hi):
            if ctx_pos != pos:
                pairs.append((center, idxs[ctx_pos]))
pairs = np.array(pairs)
print(f"Обучающих пар (center, context): {len(pairs)}")

# Распределение для негативного сэмплирования: частота^0.75 (трюк из Word2Vec)
freq = np.array([counts[idx2word[i]] for i in range(V)], dtype=np.float64)
neg_dist = freq ** 0.75
neg_dist /= neg_dist.sum()

# ──────────────────────────────────────────────────────────────────────
# 4. Параметры модели = два набора эмбеддингов
# ──────────────────────────────────────────────────────────────────────
DIM = 16
W_in = (np.random.rand(V, DIM) - 0.5) / DIM   # вектор слова как "центра"
W_out = (np.random.rand(V, DIM) - 0.5) / DIM  # вектор слова как "контекста"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ──────────────────────────────────────────────────────────────────────
# 5. Обучение: skip-gram с negative sampling, обычный SGD
# ──────────────────────────────────────────────────────────────────────
EPOCHS = 200
LR = 0.05
K = 5  # негативов на одну пару

for epoch in range(EPOCHS):
    np.random.shuffle(pairs)
    total_loss = 0.0
    for center, context in pairs:
        v_c = W_in[center]                      # (DIM,)
        negatives = np.random.choice(V, size=K, p=neg_dist)

        # позитив
        z_pos = v_c @ W_out[context]
        s_pos = sigmoid(z_pos)
        # негативы
        z_neg = W_out[negatives] @ v_c          # (K,)
        s_neg = sigmoid(z_neg)

        total_loss += -np.log(s_pos + 1e-9) - np.log(1 - s_neg + 1e-9).sum()

        # градиенты
        grad_v = (s_pos - 1) * W_out[context] + (s_neg[:, None] * W_out[negatives]).sum(0)
        grad_out_pos = (s_pos - 1) * v_c
        grad_out_neg = s_neg[:, None] * v_c[None, :]

        # шаг SGD
        W_in[center]    -= LR * grad_v
        W_out[context]  -= LR * grad_out_pos
        W_out[negatives] -= LR * grad_out_neg

    if (epoch + 1) % 40 == 0:
        print(f"эпоха {epoch+1:3d}  средний loss = {total_loss/len(pairs):.4f}")

# ──────────────────────────────────────────────────────────────────────
# 6. Готовые эмбеддинги = W_in. Смотрим ближайших соседей по косинусу.
# ──────────────────────────────────────────────────────────────────────
def normalize(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

E = normalize(W_in)

def neighbors(word, k=4):
    if word not in word2idx:
        return f"'{word}' нет в словаре"
    q = E[word2idx[word]]
    sims = E @ q
    order = np.argsort(-sims)
    out = [(idx2word[i], round(float(sims[i]), 3)) for i in order if i != word2idx[word]][:k]
    return out

print("\nБлижайшие соседи (косинусная близость):")
for berry in ["клубника", "черника", "крыжовник", "малина"]:
    print(f"  {berry:10s} -> {neighbors(berry)}")
