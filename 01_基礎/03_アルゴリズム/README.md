# 主要なアルゴリズム

## 概要

AIの基礎となるアルゴリズムを理解することは、より高度な機械学習やディープラーニングを学ぶ上で重要です。ここでは、古典的なAIアルゴリズムから基本的な機械学習アルゴリズムまでを紹介します。

## 1. 探索アルゴリズム

### 深さ優先探索（DFS）
木構造やグラフを探索する基本的なアルゴリズム。できるだけ深く探索してから戻る。

**特徴**：
- メモリ効率が良い
- 最適解を保証しない
- 無限ループに陥る可能性

### 幅優先探索（BFS）
同じレベルのノードを全て探索してから次のレベルへ進む。

**特徴**：
- 最短経路を保証
- メモリを多く使用
- 完全性がある

### A*（エースター）アルゴリズム
ヒューリスティック関数を使用して効率的に最短経路を見つける。

```python
# A*アルゴリズムの基本的な考え方
f(n) = g(n) + h(n)
# g(n): スタートからnまでの実際のコスト
# h(n): nからゴールまでの推定コスト（ヒューリスティック）
```

## 2. 最適化アルゴリズム

### 勾配降下法
関数の最小値を見つけるための反復的アルゴリズム。機械学習の基礎。

```python
# 基本的な更新式
θ = θ - α * ∇f(θ)
# θ: パラメータ
# α: 学習率
# ∇f(θ): 勾配
```

### 遺伝的アルゴリズム
生物の進化を模倣した最適化手法。

**基本的な流れ**：
1. 初期集団の生成
2. 適応度の評価
3. 選択
4. 交叉
5. 突然変異
6. 次世代の生成

## 3. 基本的な機械学習アルゴリズム

### k近傍法（k-NN）
最も単純な分類アルゴリズムの一つ。

**原理**：
- 新しいデータ点の周りのk個の最近傍点を見る
- 多数決で分類を決定

**特徴**：
- 実装が簡単
- 計算コストが高い
- 次元の呪いに弱い

### 決定木
データを条件分岐で分類する木構造のアルゴリズム。

**利点**：
- 解釈しやすい
- 前処理が少ない
- カテゴリカルデータも扱える

**欠点**：
- 過学習しやすい
- 不安定

### k-meansクラスタリング
教師なし学習の代表的なアルゴリズム。

**アルゴリズム**：
1. k個の初期中心点を選択
2. 各データ点を最も近い中心に割り当て
3. 各クラスタの中心を再計算
4. 収束するまで2-3を繰り返し

## 4. 評価指標

### 分類問題
- **精度（Accuracy）**：全体の正解率
- **適合率（Precision）**：陽性と予測したものの正解率
- **再現率（Recall）**：実際の陽性を正しく予測できた割合
- **F1スコア**：適合率と再現率の調和平均

### 回帰問題
- **平均二乗誤差（MSE）**
- **平均絶対誤差（MAE）**
- **決定係数（R²）**

## Pythonでの実装例

### k-NNの簡単な実装

```python
import numpy as np
from collections import Counter

class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = []
        for x in X:
            # 各訓練データとの距離を計算
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            # k個の最近傍のインデックスを取得
            k_indices = np.argsort(distances)[:self.k]
            # 最近傍のラベルを取得
            k_nearest_labels = self.y_train[k_indices]
            # 多数決で予測
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

# 使用例
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# データの準備
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# モデルの訓練と予測
knn = SimpleKNN(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# 精度の計算
accuracy = np.mean(predictions == y_test)
print(f"精度: {accuracy:.2f}")
```

## まとめ

これらの基本的なアルゴリズムは、現代のAI技術の基礎となっています。探索アルゴリズムはゲームAIやパス検索に、最適化アルゴリズムは機械学習の学習過程に、そして基本的な機械学習アルゴリズムは、より複雑な手法を理解する土台となります。

## 次のステップ

基礎を理解したら、[機械学習](../../02_機械学習/README.md)のセクションでより高度な内容を学びましょう。