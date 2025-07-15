# 教師あり学習

## 概要

教師あり学習（Supervised Learning）は、入力データとそれに対応する正解ラベル（教師データ）を使って学習する手法です。新しいデータに対して、学習したパターンを基に予測を行います。

## 教師あり学習の分類

### 1. 回帰（Regression）
連続的な数値を予測する問題

**例**：
- 住宅価格の予測
- 株価の予測
- 気温の予測

### 2. 分類（Classification）
離散的なカテゴリを予測する問題

**例**：
- メールのスパム判定
- 画像の物体認識
- 病気の診断

## 主要なアルゴリズム

### 1. 線形回帰（Linear Regression）

最も基本的な回帰アルゴリズム。データと直線の関係を学習します。

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# サンプルデータの生成
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 2

# モデルの学習
model = LinearRegression()
model.fit(X, y)

# 予測
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# 可視化
plt.scatter(X, y, alpha=0.5, label='データ')
plt.plot(X_test, y_pred, 'r-', label='予測')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('線形回帰の例')
plt.show()

print(f"係数: {model.coef_[0][0]:.2f}")
print(f"切片: {model.intercept_[0]:.2f}")
```

### 2. ロジスティック回帰（Logistic Regression）

分類問題に使用される手法。確率を出力します。

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# データの準備
iris = load_iris()
X = iris.data[:, :2]  # 最初の2つの特徴量のみ使用
y = iris.target

# 2クラス分類に簡略化
X = X[y != 2]
y = y[y != 2]

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# モデルの学習
model = LogisticRegression()
model.fit(X_train, y_train)

# 予測と評価
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"精度: {accuracy:.2f}")
print("\n分類レポート:")
print(classification_report(y_test, y_pred))
```

### 3. サポートベクターマシン（SVM）

マージン最大化の原理に基づく強力な分類器。

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# データの標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVMモデルの学習
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_scaled, y_train)

# 予測と評価
y_pred_svm = svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM精度: {accuracy_svm:.2f}")
```

### 4. 決定木（Decision Tree）

if-then ルールの木構造で分類・回帰を行います。

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 決定木モデルの学習
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# 可視化
plt.figure(figsize=(10, 8))
plot_tree(tree_model, feature_names=['特徴1', '特徴2'], 
          class_names=['クラス0', 'クラス1'], filled=True)
plt.title('決定木の可視化')
plt.show()

# 予測と評価
y_pred_tree = tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"決定木精度: {accuracy_tree:.2f}")
```

### 5. ランダムフォレスト（Random Forest）

複数の決定木を組み合わせたアンサンブル学習。

```python
from sklearn.ensemble import RandomForestClassifier

# ランダムフォレストモデルの学習
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 特徴量の重要度
feature_importance = rf_model.feature_importances_
print("特徴量の重要度:")
for i, importance in enumerate(feature_importance):
    print(f"特徴{i+1}: {importance:.3f}")

# 予測と評価
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nランダムフォレスト精度: {accuracy_rf:.2f}")
```

## 過学習と正則化

### 過学習（Overfitting）
訓練データに過度に適合し、汎化性能が低下する現象。

### 対策
1. **正則化**：L1（Lasso）、L2（Ridge）正則化
2. **交差検証**：k-fold cross-validation
3. **早期停止**：Early stopping
4. **ドロップアウト**：ニューラルネットワークで使用

### 正則化の例

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score

# Ridge回帰（L2正則化）
ridge_model = Ridge(alpha=1.0)
ridge_scores = cross_val_score(ridge_model, X_train, y_train, cv=5)
print(f"Ridge回帰の交差検証スコア: {ridge_scores.mean():.3f} (+/- {ridge_scores.std():.3f})")

# Lasso回帰（L1正則化）
lasso_model = Lasso(alpha=0.1)
lasso_scores = cross_val_score(lasso_model, X_train, y_train, cv=5)
print(f"Lasso回帰の交差検証スコア: {lasso_scores.mean():.3f} (+/- {lasso_scores.std():.3f})")
```

## ハイパーパラメータチューニング

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

# パラメータグリッドの定義
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Grid Searchの実行
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"最適なパラメータ: {grid_search.best_params_}")
print(f"最高スコア: {grid_search.best_score_:.3f}")
```

## 評価指標

### 分類問題の評価指標

```python
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

# 混同行列
cm = confusion_matrix(y_test, y_pred)
print("混同行列:")
print(cm)

# ROC曲線とAUC
if len(np.unique(y)) == 2:  # 2クラス分類の場合
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC曲線 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('偽陽性率')
    plt.ylabel('真陽性率')
    plt.title('ROC曲線')
    plt.legend()
    plt.show()
```

## まとめ

教師あり学習は、正解データが利用できる場合に非常に強力な手法です。適切なアルゴリズムの選択、過学習の防止、ハイパーパラメータの調整が成功の鍵となります。

## 次へ

[教師なし学習](../02_教師なし学習/README.md)へ進む