# ディープラーニング

ディープラーニング（深層学習）は、多層のニューラルネットワークを使用して複雑なパターンを学習する機械学習の一分野です。画像認識、自然言語処理、音声認識など、様々な分野で革命的な成果を上げています。

## 学習内容

### 1. [ニューラルネットワークの基礎](./ニューラルネットワーク/README.md)
- パーセプトロンから多層パーセプトロンへ
- 活性化関数
- 誤差逆伝播法
- 最適化アルゴリズム

### 2. [CNN（畳み込みニューラルネットワーク）](./CNN/README.md)
- 畳み込み層とプーリング層
- 画像認識への応用
- 有名なCNNアーキテクチャ
- 転移学習

### 3. [RNN（再帰型ニューラルネットワーク）](./RNN/README.md)
- 時系列データの処理
- LSTM と GRU
- 自然言語処理への応用
- 系列変換モデル

### 4. [Transformer](./Transformer/README.md)
- Attention機構
- Self-Attention
- BERT、GPTなどの言語モデル
- Vision Transformerへの発展

## ディープラーニングの特徴

### 利点
- **特徴量の自動抽出**：手動での特徴量設計が不要
- **高い表現力**：複雑なパターンを学習可能
- **転移学習**：学習済みモデルの活用

### 課題
- **大量のデータが必要**
- **計算コストが高い**
- **ブラックボックス性**

## 前提知識

- [機械学習](../機械学習/README.md)の基礎を理解していること
- 線形代数（行列演算）
- 微分（連鎖律）
- Pythonプログラミング

## 必要な環境

```python
# 主要なディープラーニングフレームワーク
import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt

# GPU確認
print("TensorFlow GPU:", tf.config.list_physical_devices('GPU'))
print("PyTorch GPU:", torch.cuda.is_available())
```

## 次のステップ

ディープラーニングを学んだら、[AI開発の実践](../実践/README.md)で実際のプロジェクトに取り組みましょう。