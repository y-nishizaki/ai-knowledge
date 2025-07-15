# AI開発の実践

AI開発を実際に行うための実践的な知識とツールについて学びます。環境構築から、ライブラリの使い方、モデルの評価、そして本番環境へのデプロイまでをカバーします。

## 学習内容

### 1. [Python基礎](./Python基礎/README.md)
- AI開発に必要なPythonの知識
- NumPy、Pandasの使い方
- データの前処理

### 2. [主要ライブラリ](./ライブラリ/README.md)
- Scikit-learn
- TensorFlow/Keras
- PyTorch
- その他の便利なライブラリ

### 3. [ディープラーニングフレームワーク](./フレームワーク/README.md)
- フレームワークの選び方
- モデルの構築と学習
- カスタムレイヤーの作成

### 4. [MLOps入門](./MLOps/README.md)
- モデルのバージョン管理
- 実験管理
- モデルのデプロイ
- モニタリング

## 開発環境の準備

```bash
# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 基本的なパッケージのインストール
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install tensorflow  # または pip install torch
pip install jupyter notebook
```

## プロジェクト構造の例

```
my_ai_project/
├── data/
│   ├── raw/            # 生データ
│   ├── processed/      # 前処理済みデータ
│   └── external/       # 外部データ
├── notebooks/          # Jupyter notebooks
├── src/
│   ├── data/          # データ処理スクリプト
│   ├── features/      # 特徴量エンジニアリング
│   ├── models/        # モデル定義
│   └── visualization/ # 可視化
├── models/            # 学習済みモデル
├── reports/           # レポートと図
├── requirements.txt   # 依存関係
└── README.md
```

## 次のステップ

実践的なスキルを身につけたら、[応用分野](../05_応用/README.md)で具体的な応用例を学びましょう。