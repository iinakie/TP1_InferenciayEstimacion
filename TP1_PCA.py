import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ── PARÁMETROS ───────────────────────────────────────────────
DATA_SET = "dataset"
CLASES    = ["eritroblasto", "mieloblasto", "monocito", "plaqueta"]
N_IMG     = 500
K_VALS    = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 784]
SEMILLA      = 42
np.random.seed(SEMILLA)

# FUNCIONES ────────────────────────────────────────────────
# Cargo n imágenes de una clase y las aplano a vectores de 784.
def cargar_imagenes(clase, n=500):
    carpeta = os.path.join(DATA_SET, clase)
    archivos = [f for f in os.listdir(carpeta)
                if f.endswith(('.png', '.jpg', '.jpeg'))]
    elegidos = np.random.choice(archivos, size=n, replace=False)
    X = []
    for f in elegidos:
        img = Image.open(os.path.join(carpeta, f)).convert("L")
        X.append(np.array(img).flatten())
    return np.array(X, dtype=float)  # shape: (500, 784)

# hago PCA, calculo la media, autovals y autovecs de la matriz de covarianza.
def hacer_pca(X):
    mu    = X.mean(axis=0)
    X_c   = X - mu
    C     = np.cov(X_c, rowvar=False)           # (784, 784)
    autovals, autovecs = np.linalg.eigh(C)
    idx          = np.argsort(autovals)[::-1] # ordenar mayor a menor
    autovals  = autovals[idx]
    autovecs = autovecs[:, idx]
    return mu, autovals, autovecs, X_c

# proyecto y reconstruyo usando los K primeros autovecs
def reconstruir(X_c, autovecs, K):
    P_K   = autovecs[:, :K]   # (784, K)
    Y     = X_c @ P_K             # proyección: (500, K)
    X_rec = Y @ P_K.T             # reconstrucción: (500, 784)
    return X_rec

# EJERCICIO 1A: varianza explicada y MSE vs K ──────────────
resultados = {}

for clase in CLASES:
    print(f"Procesando: {clase}...")
    X = cargar_imagenes(clase, N_IMG)
    mu, autovals, autovecs, X_c = hacer_pca(X)

    var_total = autovals.sum()
    vars_exp, mses = [], []

    for K in K_VALS:
        # varianza explicada acumulada
        ve  = autovals[:K].sum() / var_total
        vars_exp.append(ve)

        # MSE entre imagen original y reconstruida
        X_rec = reconstruir(X_c, autovecs, K)
        mse   = np.mean((X_c - X_rec) ** 2)
        mses.append(mse)

    resultados[clase] = {"var_exp": vars_exp, "mse": mses,
                         "mu": mu, "autovals": autovals,
                         "autovecs": autovecs,
                         "X_c": X_c, "X": X}

# GRÁFICOS 1A ──────────────────────────────────────────────
colores = ['blue', 'red', 'green', 'grey']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, clase in enumerate(CLASES):
    axes[0].plot(K_VALS, resultados[clase]["var_exp"],
                 marker='o', label=clase, color=colores[i])
    axes[1].plot(K_VALS, resultados[clase]["mse"],
                 marker='o', label=clase, color=colores[i])

for ax, titulo, ylabel in zip(axes,
    ["Varianza explicada vs K", "MSE vs K"],
    ["Varianza explicada", "MSE"]):
    ax.set_xlabel("K (componentes principales)")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ejercicio1a.png", dpi=150)
plt.show()
print("Gráfico guardado: ejercicio1a.png")

# EJERCICIO 1B: reconstrucción visual ──────────────────────
fig, axes = plt.subplots(4, 3, figsize=(8, 12))

for i, clase in enumerate(CLASES):
    X   = resultados[clase]["X"]
    X_c = resultados[clase]["X_c"]
    mu  = resultados[clase]["mu"]
    av  = resultados[clase]["autovecs"]

    # imagen 0 de cada clase
    img_orig = X[0].reshape(28, 28)

    rec_50 = (reconstruir(X_c, av, K=50)[0]  + mu).reshape(28, 28)
    rec_2  = (reconstruir(X_c, av, K=2)[0]   + mu).reshape(28, 28)

    axes[i][0].imshow(img_orig, cmap='gray')
    axes[i][0].set_title(f"{clase}\nOriginal")
    axes[i][0].axis('off')

    axes[i][1].imshow(rec_50, cmap='gray')
    axes[i][1].set_title("K=50")
    axes[i][1].axis('off')

    axes[i][2].imshow(rec_2, cmap='gray')
    axes[i][2].set_title("K=2")
    axes[i][2].axis('off')

plt.suptitle("Ejercicio 1b: Reconstrucción por clase", fontsize=13)
plt.tight_layout()
plt.savefig("ejercicio1b.png", dpi=150)
plt.show()
print("Gráfico guardado: ejercicio1b.png")