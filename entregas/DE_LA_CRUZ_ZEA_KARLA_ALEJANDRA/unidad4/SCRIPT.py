from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd

# Cargar dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Métricas
sil = silhouette_score(X, labels)
db = davies_bouldin_score(X, labels)
ch = calinski_harabasz_score(X, labels)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

var = pca.explained_variance_ratio_
