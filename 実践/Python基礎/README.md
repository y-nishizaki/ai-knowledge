# Python基礎

## 概要

AI開発において、Pythonは最も広く使われているプログラミング言語です。豊富なライブラリ、簡潔な文法、活発なコミュニティがその理由です。ここでは、AI開発に必要なPythonの基礎知識を学びます。

## Python環境のセットアップ

### 1. Pythonのインストール

```bash
# バージョン確認
python --version  # 3.8以上を推奨

# pipのアップグレード
python -m pip install --upgrade pip
```

### 2. 仮想環境の作成

```bash
# venvを使った仮想環境
python -m venv .venv

# アクティベート
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 仮想環境の無効化
deactivate
```

### 3. Jupyter Notebookの設定

```python
# Jupyterのインストール
pip install jupyter notebook ipykernel

# 仮想環境をJupyterカーネルに追加
python -m ipykernel install --user --name=.venv --display-name="Python (AI)"

# Jupyter Notebookの起動
jupyter notebook
```

## NumPy - 数値計算の基礎

### 配列の作成と操作

```python
import numpy as np

# 配列の作成
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# 特殊な配列
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
identity = np.eye(4)
random_arr = np.random.randn(3, 3)

# 配列の属性
print(f"形状: {arr2.shape}")
print(f"次元数: {arr2.ndim}")
print(f"データ型: {arr2.dtype}")
print(f"要素数: {arr2.size}")
```

### 配列の演算

```python
# 要素ごとの演算
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print("加算:", a + b)
print("乗算:", a * b)
print("べき乗:", a ** 2)

# 行列演算
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("行列積:", np.dot(A, B))
print("要素ごとの積:", A * B)

# ブロードキャスティング
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])

print("ブロードキャスティング:")
print(matrix + vector)
```

### インデックスとスライシング

```python
# 1次元配列
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print("スライス:", arr[2:7])
print("ステップ指定:", arr[::2])
print("逆順:", arr[::-1])

# 2次元配列
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print("行の選択:", matrix[1])
print("列の選択:", matrix[:, 2])
print("部分行列:", matrix[1:, 2:])

# ブールインデックス
data = np.array([1, -2, 3, -4, 5])
mask = data > 0
print("正の値のみ:", data[mask])
```

### 統計関数

```python
# 基本的な統計量
data = np.random.randn(1000)

print(f"平均: {np.mean(data):.4f}")
print(f"標準偏差: {np.std(data):.4f}")
print(f"最小値: {np.min(data):.4f}")
print(f"最大値: {np.max(data):.4f}")
print(f"中央値: {np.median(data):.4f}")

# 軸に沿った計算
matrix = np.random.randn(4, 5)
print("列ごとの平均:", np.mean(matrix, axis=0))
print("行ごとの合計:", np.sum(matrix, axis=1))
```

## Pandas - データ処理の基礎

### DataFrameの作成と基本操作

```python
import pandas as pd

# DataFrameの作成
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'Score': [85, 92, 78, 91],
    'City': ['Tokyo', 'Osaka', 'Tokyo', 'Kyoto']
}
df = pd.DataFrame(data)

print("データフレーム:")
print(df)
print("\n基本情報:")
print(df.info())
print("\n統計情報:")
print(df.describe())
```

### データの読み込みと書き出し

```python
# CSVファイルの読み込み
# df = pd.read_csv('data.csv')

# Excelファイルの読み込み
# df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSONファイルの読み込み
# df = pd.read_json('data.json')

# データの保存
# df.to_csv('output.csv', index=False)
# df.to_excel('output.xlsx', index=False)
# df.to_json('output.json', orient='records')

# サンプルデータの作成と保存
sample_df = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=10),
    'Sales': np.random.randint(100, 1000, 10),
    'Customers': np.random.randint(10, 100, 10)
})

print("サンプルデータ:")
print(sample_df.head())
```

### データの選択とフィルタリング

```python
# 列の選択
print("年齢列:", df['Age'])
print("複数列:", df[['Name', 'Score']])

# 行の選択
print("インデックス0の行:", df.iloc[0])
print("条件による選択:", df[df['Age'] > 28])

# locとilocの使い方
print("loc（ラベルベース）:", df.loc[df['City'] == 'Tokyo', ['Name', 'Score']])
print("iloc（位置ベース）:", df.iloc[0:2, 1:3])

# 複雑な条件
condition = (df['Age'] > 25) & (df['Score'] > 80)
print("複数条件:", df[condition])
```

### データの集計とグループ化

```python
# グループ化
grouped = df.groupby('City')
print("都市ごとの平均スコア:")
print(grouped['Score'].mean())

# 複数の集計
agg_result = grouped.agg({
    'Age': ['mean', 'min', 'max'],
    'Score': ['mean', 'std']
})
print("\n複数の集計:")
print(agg_result)

# ピボットテーブル
pivot_data = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=20),
    'Product': ['A', 'B'] * 10,
    'Sales': np.random.randint(100, 500, 20),
    'Region': ['East', 'West'] * 10
})

pivot_table = pivot_data.pivot_table(
    values='Sales',
    index='Product',
    columns='Region',
    aggfunc='mean'
)
print("\nピボットテーブル:")
print(pivot_table)
```

### 欠損値の処理

```python
# 欠損値を含むデータの作成
df_with_nan = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})

print("欠損値を含むデータ:")
print(df_with_nan)
print("\n欠損値の確認:")
print(df_with_nan.isnull().sum())

# 欠損値の処理
# 削除
df_dropped = df_with_nan.dropna()
print("\n欠損値を含む行を削除:")
print(df_dropped)

# 補完
df_filled = df_with_nan.fillna(df_with_nan.mean())
print("\n平均値で補完:")
print(df_filled)

# 前方補完
df_ffill = df_with_nan.fillna(method='ffill')
print("\n前方補完:")
print(df_ffill)
```

## Matplotlib - データの可視化

### 基本的なプロット

```python
import matplotlib.pyplot as plt

# 線グラフ
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2, linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('三角関数のプロット')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 散布図
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6, c=x, cmap='viridis')
plt.colorbar(label='x value')
plt.xlabel('x')
plt.ylabel('y')
plt.title('散布図の例')
plt.show()

# ヒストグラム
data = np.random.normal(100, 15, 1000)

plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=2, label=f'平均: {data.mean():.2f}')
plt.xlabel('値')
plt.ylabel('頻度')
plt.title('正規分布のヒストグラム')
plt.legend()
plt.show()
```

### サブプロット

```python
# 複数のグラフを並べる
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 各サブプロットにグラフを描画
x = np.linspace(0, 2*np.pi, 100)

axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sin')

axes[0, 1].plot(x, np.cos(x), 'r-')
axes[0, 1].set_title('Cos')

axes[1, 0].plot(x, np.tan(x))
axes[1, 0].set_ylim(-5, 5)
axes[1, 0].set_title('Tan')

axes[1, 1].plot(x, x**2, 'g-')
axes[1, 1].set_title('x²')

plt.tight_layout()
plt.show()
```

## Seaborn - 統計的可視化

```python
import seaborn as sns

# Seabornのスタイル設定
sns.set_style("whitegrid")

# サンプルデータの作成
tips = sns.load_dataset("tips")

# ペアプロット
plt.figure(figsize=(10, 8))
sns.pairplot(tips[['total_bill', 'tip', 'size']], hue='time')
plt.show()

# ヒートマップ
correlation = tips[['total_bill', 'tip', 'size']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('相関行列のヒートマップ')
plt.show()

# ボックスプロット
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='total_bill', hue='sex', data=tips)
plt.title('曜日別・性別の会計金額分布')
plt.show()

# バイオリンプロット
plt.figure(figsize=(10, 6))
sns.violinplot(x='day', y='total_bill', data=tips, inner='quartile')
plt.title('曜日別の会計金額分布（バイオリンプロット）')
plt.show()
```

## データの前処理

### 特徴量スケーリング

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# サンプルデータ
data = pd.DataFrame({
    'feature1': np.random.normal(100, 50, 1000),
    'feature2': np.random.exponential(10, 1000),
    'feature3': np.random.uniform(0, 1, 1000)
})

# 標準化（平均0、標準偏差1）
scaler = StandardScaler()
data_standard = scaler.fit_transform(data)

# 正規化（0-1の範囲）
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)

# ロバストスケーリング（外れ値に強い）
robust_scaler = RobustScaler()
data_robust = robust_scaler.fit_transform(data)

# 可視化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(data['feature1'], bins=30, alpha=0.7)
axes[0, 0].set_title('元データ')

axes[0, 1].hist(data_standard[:, 0], bins=30, alpha=0.7)
axes[0, 1].set_title('標準化後')

axes[1, 0].hist(data_minmax[:, 0], bins=30, alpha=0.7)
axes[1, 0].set_title('正規化後')

axes[1, 1].hist(data_robust[:, 0], bins=30, alpha=0.7)
axes[1, 1].set_title('ロバストスケーリング後')

plt.tight_layout()
plt.show()
```

### カテゴリ変数のエンコーディング

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# サンプルデータ
categorical_data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'M', 'S'],
    'price': [100, 200, 150, 100, 200]
})

# Label Encoding
le = LabelEncoder()
categorical_data['color_encoded'] = le.fit_transform(categorical_data['color'])

# One-Hot Encoding
one_hot = pd.get_dummies(categorical_data[['color', 'size']], prefix=['color', 'size'])

print("元データ:")
print(categorical_data)
print("\nOne-Hot Encoding後:")
print(one_hot)
```

## ファイル操作とデータ管理

### pickleによるオブジェクトの保存

```python
import pickle

# モデルやデータの保存
model_data = {
    'weights': np.random.randn(10, 5),
    'bias': np.random.randn(5),
    'config': {'learning_rate': 0.01, 'epochs': 100}
}

# 保存
with open('model_data.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# 読み込み
with open('model_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

print("保存したデータ:")
print(loaded_data['config'])
```

### JSONでの設定管理

```python
import json

# 設定ファイルの例
config = {
    'model': {
        'type': 'CNN',
        'layers': [
            {'type': 'Conv2D', 'filters': 32, 'kernel_size': 3},
            {'type': 'MaxPooling2D', 'pool_size': 2},
            {'type': 'Dense', 'units': 128}
        ]
    },
    'training': {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001
    }
}

# JSONファイルに保存
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

# JSONファイルから読み込み
with open('config.json', 'r') as f:
    loaded_config = json.load(f)

print("設定内容:")
print(json.dumps(loaded_config, indent=2))
```

## エラーハンドリングとデバッグ

```python
# 基本的なエラーハンドリング
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("エラー: ゼロで除算しようとしました")
        return None
    except TypeError:
        print("エラー: 数値以外が入力されました")
        return None
    finally:
        print("計算を終了します")

# テスト
print(safe_divide(10, 2))
print(safe_divide(10, 0))
print(safe_divide(10, "a"))

# アサーションによるデバッグ
def process_data(data):
    assert isinstance(data, np.ndarray), "データはNumPy配列である必要があります"
    assert data.shape[1] == 10, "データは10個の特徴量を持つ必要があります"
    
    # データ処理
    return data.mean(axis=0)

# ログ出力
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(epochs):
    logger.info(f"学習を開始します。エポック数: {epochs}")
    
    for epoch in range(epochs):
        # 学習処理（ダミー）
        loss = np.random.random()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    logger.info("学習が完了しました")

train_model(30)
```

## パフォーマンスの最適化

```python
import time

# 処理時間の計測
def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}の実行時間: {end - start:.4f}秒")
        return result
    return wrapper

@measure_time
def slow_function(n):
    """リスト内包表記を使わない例"""
    result = []
    for i in range(n):
        if i % 2 == 0:
            result.append(i ** 2)
    return result

@measure_time
def fast_function(n):
    """リスト内包表記を使った例"""
    return [i ** 2 for i in range(n) if i % 2 == 0]

# NumPyを使った高速化
@measure_time
def numpy_function(n):
    """NumPyを使った例"""
    arr = np.arange(n)
    mask = arr % 2 == 0
    return arr[mask] ** 2

# 比較
n = 1000000
result1 = slow_function(n)
result2 = fast_function(n)
result3 = numpy_function(n)
```

## まとめ

AI開発に必要なPythonの基礎知識：

1. **NumPy**：数値計算の基礎、配列操作
2. **Pandas**：データ処理、前処理
3. **Matplotlib/Seaborn**：データの可視化
4. **データの前処理**：スケーリング、エンコーディング
5. **ファイル操作**：モデルや設定の保存・読み込み

これらのスキルは、機械学習やディープラーニングのプロジェクトで必須となります。

## 次へ

[主要ライブラリ](../ライブラリ/README.md)へ進む