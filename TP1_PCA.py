import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

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

# ########################################################################################

# Preparo X e y con todas las clases juntas 
X_total = np.vstack([resultados[c]["X"] for c in CLASES])
y_total = np.hstack([np.full(N_IMG, i)  for i, c in enumerate(CLASES)])

# EJERCICIO 2A: sin PCA, vectores completos ────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_total, y_total, test_size=0.3, stratify=y_total, random_state=42
)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
acc_full = accuracy_score(y_test, clf.predict(X_test))
print(f"Accuracy sin PCA (784 dims): {acc_full:.4f}")

# EJERCICIO 2B: con PCA para cada K ────────────────────────
mu_total = X_total.mean(axis=0)
X_total_c = X_total - mu_total
C_total = np.cov(X_total_c, rowvar=False)
avals, avecs = np.linalg.eigh(C_total)
idx = np.argsort(avals)[::-1]
avals, avecs = avals[idx], avecs[:, idx]

accs_pca = []

for K in K_VALS:
    P_K   = avecs[:, :K]
    X_red = X_total_c @ P_K

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_red, y_total, test_size=0.3, stratify=y_total, random_state=42
    )

    clf_k = KNeighborsClassifier(n_neighbors=5)
    clf_k.fit(X_tr, y_tr)
    acc_k = accuracy_score(y_te, clf_k.predict(X_te))
    accs_pca.append(acc_k)
    print(f"K={K:4d} → Accuracy: {acc_k:.4f}")

# ── GRÁFICO 2 ─────────────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.plot(K_VALS, accs_pca, marker='o', color='steelblue', label='Con PCA')
plt.axhline(acc_full, color='tomato', linestyle='--',
            label=f'Sin PCA ({acc_full:.4f})')
plt.xscale('log')
plt.xlabel("K (componentes principales)")
plt.ylabel("Accuracy")
plt.title("Ejercicio 2: Accuracy del clasificador k-nn vs K")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ejercicio2.png", dpi=150)
plt.show()
print("Gráfico guardado: ejercicio2.png")


# ########################################################################################
# ── EJERCICIO 3: Scatter plot 2D ─────────────────────────────
colores_clase = ['steelblue', 'tomato', 'seagreen', 'darkorchid']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, clase in enumerate(CLASES):
    X_clase = resultados[clase]["X"]

    # (a) dos primeros píxeles del vector original
    axes[0].scatter(X_clase[:, 0], X_clase[:, 1],
                    color=colores_clase[i], label=clase,
                    alpha=0.5, s=10)

    # (b) PCA con K=2
    X_clase_c = X_clase - mu_total
    Y_pca = X_clase_c @ avecs[:, :2]
    axes[1].scatter(Y_pca[:, 0], Y_pca[:, 1],
                    color=colores_clase[i], label=clase,
                    alpha=0.5, s=10)

axes[0].set_title("(a) Dos primeros píxeles originales")
axes[0].set_xlabel("Pixel 0")
axes[0].set_ylabel("Pixel 1")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("(b) PCA con K=2")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Ejercicio 3: Scatter plot 2D por clase", fontsize=13)
plt.tight_layout()
plt.savefig("ejercicio3.png", dpi=150)
plt.show()
print("Gráfico guardado: ejercicio3.png")

"""
CONCLUSION:

Sobre el Ejercicio 1:
PCA permite comprimir imágenes de 784 dimensiones a muchas menos conservando la mayor parte de la información. Con K=50 ya se recupera una imagen visualmente aceptable, mientras que con K=2 la reconstrucción es muy pobre. Esto se refleja en las curvas: la varianza explicada sube rápido al principio y se estabiliza, mientras el MSE cae rápido y se achata.
Sobre el Ejercicio 2:
Reducir dimensionalidad no solo no perjudica la clasificación sino que con un K intermedio la mejora, porque PCA elimina ruido y deja solo la información relevante. Con K muy chico el clasificador no tiene suficiente información, y con K grande se iguala al caso sin PCA.
Sobre el Ejercicio 3:
En el espacio original dos píxeles arbitrarios no separan las clases. En el espacio PCA con K=2 ya se observa separación entre clases, lo que confirma que PCA encuentra las direcciones realmente informativas para distinguirlas.
Idea general que une todo:
PCA es una herramienta poderosa para reducir dimensionalidad manteniendo la información relevante, lo que facilita tanto la visualización como la clasificación.
Agarrá estos puntos, ponelos en tus palabras y agregá los números concretos que te dieron tus gráficos (el accuracy exacto, con qué K fue el pico, etc.). Eso le da mucho más valor al informe.

"""