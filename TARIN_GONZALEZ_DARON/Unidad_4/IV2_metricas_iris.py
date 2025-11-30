# iv2_metricas_iris.py
# Complemento de código para el reporte:
# "IV.2. Métricas de evaluación de modelos"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def main():
    # ============================
    # 1. Cargar dataset Iris
    # ============================
    iris = load_iris()
    X = iris.data           # 4 atributos numéricos
    y = iris.target         # etiquetas reales (no se usan para las métricas)
    feature_names = iris.feature_names

    print("Shape de X:", X.shape)
    print("Atributos:", feature_names)
    print()

    # ============================
    # 2. Estandarizar datos
    # ============================
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # ============================
    # 3. Clustering con K-Means (k = 3)
    # ============================
    k = 3
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(X_std)

    # ============================
    # 4. Métricas de clustering
    # ============================
    sil = silhouette_score(X_std, labels)
    db = davies_bouldin_score(X_std, labels)
    ch = calinski_harabasz_score(X_std, labels)

    print("=== Métricas de Clustering (K-means, k=3) ===")
    print(f"Índice de Silueta:        {sil:.4f}")
    print(f"Índice Davies–Bouldin:    {db:.4f}")
    print(f"Índice Calinski–Harabasz: {ch:.4f}")
    print()

    # ============================
    # 5. PCA para reducción de dimensionalidad
    #    (sobre datos estandarizados)
    # ============================
    pca_full = PCA(n_components=4)
    pca_full.fit(X_std)
    var_ratio = pca_full.explained_variance_ratio_
    var_acumulada = np.cumsum(var_ratio)

    print("=== Varianza explicada por PCA ===")
    for i, (var, var_cum) in enumerate(zip(var_ratio, var_acumulada), start=1):
        print(f"PC{i}: {var*100:.2f}%  |  Acumulada: {var_cum*100:.2f}%")
    print()

    # ============================
    # 6. Error de reconstrucción usando 2 componentes
    # ============================
    pca_2 = PCA(n_components=2)
    X_2 = pca_2.fit_transform(X_std)
    X_recon = pca_2.inverse_transform(X_2)

    # Error medio cuadrático entre X_std y su reconstrucción X_recon
    reconstruction_error = np.mean((X_std - X_recon) ** 2)

    print("=== Error de reconstrucción (PCA con 2 componentes) ===")
    print(f"E = {reconstruction_error:.6f}")
    print()

    # ============================
    # 7. Figura 1: Clusters proyectados a 2D con PCA
    # ============================
    plt.figure()
    scatter = plt.scatter(X_2[:, 0], X_2[:, 1], c=labels)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters proyectados a 2D con PCA (Iris, K-means k=3)")
    plt.grid(True)
    plt.savefig("clusters_pca2d.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ============================
    # 8. Figura 2: Varianza explicada por componente principal
    # ============================
    componentes = np.arange(1, len(var_ratio) + 1)

    plt.figure()
    plt.plot(componentes, var_ratio, marker="o")
    plt.xticks(componentes)
    plt.xlabel("Componente principal")
    plt.ylabel("Varianza explicada")
    plt.title("Varianza explicada por componente principal (PCA)")
    plt.grid(True)
    plt.savefig("pca_variance.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Figuras guardadas como:")
    print(" - clusters_pca2d.png")
    print(" - pca_variance.png")


if __name__ == "__main__":
    main()
