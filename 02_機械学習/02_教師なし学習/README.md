# 教師なし学習

## 概要

教師なし学習（Unsupervised Learning）は、正解ラベルのないデータから隠れたパターンや構造を発見する手法です。データの内在する特徴を自動的に抽出し、新しい洞察を得ることができます。

## 主要な手法

### 1. クラスタリング

データを類似性に基づいてグループ分けする手法です。

#### K-meansクラスタリング

最も基本的なクラスタリング手法。データをk個のクラスタに分割します。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# サンプルデータの生成
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                       cluster_std=0.5, random_state=42)

# K-meansクラスタリング
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# 結果の可視化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
plt.title('真のクラスタ')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', s=200, marker='*', edgecolor='black')
plt.title('K-meansの結果')
plt.show()

# エルボー法による最適なクラスタ数の決定
inertias = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K, inertias, 'bo-')
plt.xlabel('クラスタ数 k')
plt.ylabel('慣性（Inertia）')
plt.title('エルボー法')
plt.show()
```

#### 階層的クラスタリング

データを階層的にグループ化し、デンドログラム（樹形図）で表現します。

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch

# 階層的クラスタリング
linkage_matrix = linkage(X, method='ward')

# デンドログラムの描画
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix)
plt.title('階層的クラスタリング - デンドログラム')
plt.xlabel('サンプルインデックス')
plt.ylabel('距離')
plt.show()

# Agglomerative Clusteringの実行
agg_clustering = AgglomerativeClustering(n_clusters=4)
y_agg = agg_clustering.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_agg, cmap='viridis', alpha=0.6)
plt.title('階層的クラスタリングの結果')
plt.show()
```

#### DBSCAN

密度ベースのクラスタリング手法。ノイズに強く、任意の形状のクラスタを発見できます。

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)

# 結果の可視化（-1はノイズを表す）
plt.figure(figsize=(8, 6))
unique_labels = set(y_dbscan)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'black'  # ノイズは黒で表示
    
    class_member_mask = (y_dbscan == k)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.6, 
               label='ノイズ' if k == -1 else f'クラスタ {k}')

plt.title('DBSCANの結果')
plt.legend()
plt.show()
```

### 2. 次元削減

高次元データを低次元に変換し、可視化や計算効率の向上を図ります。

#### 主成分分析（PCA）

データの分散を最大化する方向を見つけて次元削減を行います。

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# 手書き数字データセットの読み込み
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# PCAによる次元削減
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_digits)

# 可視化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, 
                     cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel('第1主成分')
plt.ylabel('第2主成分')
plt.title('PCAによる手書き数字データの2次元可視化')
plt.show()

# 寄与率の確認
pca_full = PCA()
pca_full.fit(X_digits)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('主成分の数')
plt.ylabel('累積寄与率')
plt.title('累積寄与率')

plt.subplot(1, 2, 2)
plt.plot(pca_full.explained_variance_ratio_[:20])
plt.xlabel('主成分')
plt.ylabel('寄与率')
plt.title('各主成分の寄与率')
plt.show()
```

#### t-SNE

非線形な次元削減手法。局所的な構造を保持しながら可視化に適しています。

```python
from sklearn.manifold import TSNE

# t-SNEによる次元削減（計算時間がかかるため、サンプル数を制限）
n_samples = 1000
X_sample = X_digits[:n_samples]
y_sample = y_digits[:n_samples]

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_sample)

# 可視化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, 
                     cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNEによる手書き数字データの2次元可視化')
plt.show()
```

#### オートエンコーダ

ニューラルネットワークを使った次元削減手法。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# データの準備
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# オートエンコーダの構築
encoding_dim = 32  # 圧縮後の次元数

# エンコーダ
input_img = keras.Input(shape=(784,))
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# デコーダ
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

# モデルの作成
autoencoder = keras.Model(input_img, decoded)
encoder = keras.Model(input_img, encoded)

# コンパイルと学習
autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(x_train, x_train,
                         epochs=50,
                         batch_size=256,
                         shuffle=True,
                         validation_data=(x_test, x_test),
                         verbose=1)

# 圧縮された表現の取得
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

# 結果の可視化
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # 元の画像
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 再構成された画像
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

### 3. 異常検知

正常なデータのパターンを学習し、異常なデータを検出します。

#### Isolation Forest

```python
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_classification

# 正常データと異常データの生成
X_normal, _ = make_classification(n_samples=500, n_features=2, 
                                  n_redundant=0, n_clusters_per_class=1,
                                  random_state=42)

# 異常データを追加
rng = np.random.RandomState(42)
X_outliers = rng.uniform(low=-6, high=6, size=(50, 2))
X_anomaly = np.vstack([X_normal, X_outliers])

# Isolation Forestによる異常検知
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred = iso_forest.fit_predict(X_anomaly)

# 可視化
plt.figure(figsize=(10, 8))
colors = np.array(['#377eb8', '#ff7f00'])
plt.scatter(X_anomaly[:, 0], X_anomaly[:, 1], 
           c=colors[(y_pred == -1).astype(int)], alpha=0.6)
plt.title('Isolation Forestによる異常検知')
plt.legend(['正常', '異常'])
plt.show()
```

#### One-Class SVM

```python
from sklearn.svm import OneClassSVM

# One-Class SVMによる異常検知
oc_svm = OneClassSVM(gamma='auto', nu=0.1)
oc_svm.fit(X_normal)
y_pred_svm = oc_svm.predict(X_anomaly)

# 決定境界の可視化
xx, yy = np.meshgrid(np.linspace(-7, 7, 100),
                     np.linspace(-7, 7, 100))
Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), 
             cmap=plt.cm.PuBu_r, alpha=0.3)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
plt.scatter(X_anomaly[:, 0], X_anomaly[:, 1], 
           c=colors[(y_pred_svm == -1).astype(int)], alpha=0.6)
plt.title('One-Class SVMによる異常検知')
plt.legend(['決定境界', '正常', '異常'])
plt.show()
```

## 教師なし学習の評価

教師なし学習では正解ラベルがないため、評価が難しいですが、以下の指標があります：

### クラスタリングの評価

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# シルエットスコア（-1から1の間、高いほど良い）
silhouette_avg = silhouette_score(X, y_pred)
print(f"シルエットスコア: {silhouette_avg:.3f}")

# Calinski-Harabaszスコア（高いほど良い）
ch_score = calinski_harabasz_score(X, y_pred)
print(f"Calinski-Harabaszスコア: {ch_score:.3f}")
```

## 実践的な応用例

### 顧客セグメンテーション

```python
# 仮想的な顧客データの生成
np.random.seed(42)
n_customers = 1000

# 特徴量：購買頻度、平均購買額、最終購買からの日数
purchase_frequency = np.random.exponential(10, n_customers)
avg_purchase_amount = np.random.lognormal(4, 1, n_customers)
days_since_last_purchase = np.random.exponential(30, n_customers)

customer_data = np.column_stack([purchase_frequency, 
                                avg_purchase_amount, 
                                days_since_last_purchase])

# データの標準化
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)

# K-meansでセグメント化
kmeans = KMeans(n_clusters=4, random_state=42)
segments = kmeans.fit_predict(customer_data_scaled)

# セグメントごとの特徴を分析
for i in range(4):
    segment_data = customer_data[segments == i]
    print(f"\nセグメント {i+1}:")
    print(f"  顧客数: {len(segment_data)}")
    print(f"  平均購買頻度: {segment_data[:, 0].mean():.2f}")
    print(f"  平均購買額: ¥{segment_data[:, 1].mean():.0f}")
    print(f"  平均最終購買日数: {segment_data[:, 2].mean():.1f}日前")
```

## まとめ

教師なし学習は、ラベルのないデータから価値ある洞察を得るための強力な手法です。クラスタリング、次元削減、異常検知など、様々な用途に応用できます。

## 次へ

[強化学習](../03_強化学習/README.md)へ進む