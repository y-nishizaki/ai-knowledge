# 主要ライブラリ

## 概要

AI開発では、様々なライブラリを活用することで効率的に開発を進めることができます。ここでは、機械学習とディープラーニングの主要なライブラリについて、実践的な使い方を学びます。

## Scikit-learn - 機械学習の標準ライブラリ

### 基本的なワークフロー

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# データの準備（例：Irisデータセット）
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# データの標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# モデルの学習
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 予測と評価
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"精度: {accuracy:.3f}")
print("\n分類レポート:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### パイプラインの構築

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 複雑なデータの例
data = pd.DataFrame({
    'numeric_feature': np.random.randn(100),
    'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
    'target': np.random.choice([0, 1], 100)
})

X = data[['numeric_feature', 'categorical_feature']]
y = data['target']

# 前処理パイプライン
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['numeric_feature']),
        ('cat', OneHotEncoder(), ['categorical_feature'])
    ]
)

# 完全なパイプライン
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 学習と予測
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### ハイパーパラメータチューニング

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
import time

# グリッドサーチ
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

start_time = time.time()
grid_search.fit(X_train_scaled, y_train)
print(f"グリッドサーチ時間: {time.time() - start_time:.2f}秒")
print(f"最適パラメータ: {grid_search.best_params_}")
print(f"最高スコア: {grid_search.best_score_:.3f}")

# ランダムサーチ（より効率的）
from scipy.stats import uniform, randint

param_dist = {
    'C': uniform(0.1, 10),
    'gamma': uniform(0.001, 0.1),
    'kernel': ['rbf', 'linear', 'poly']
}

random_search = RandomizedSearchCV(
    SVC(), param_dist, n_iter=20, cv=5, 
    scoring='accuracy', n_jobs=-1, random_state=42
)

start_time = time.time()
random_search.fit(X_train_scaled, y_train)
print(f"\nランダムサーチ時間: {time.time() - start_time:.2f}秒")
print(f"最適パラメータ: {random_search.best_params_}")
print(f"最高スコア: {random_search.best_score_:.3f}")
```

### 特徴量選択

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.feature_selection import mutual_info_classif

# 相関に基づく特徴量選択
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X_train, y_train)

# 選択された特徴量
selected_features = iris.feature_names[selector.get_support()]
print("選択された特徴量:", selected_features)

# 再帰的特徴除去（RFE）
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression()
rfe = RFE(estimator, n_features_to_select=2)
rfe.fit(X_train_scaled, y_train)

print("\nRFEランキング:", rfe.ranking_)
print("選択された特徴量:", [iris.feature_names[i] for i in range(len(iris.feature_names)) if rfe.support_[i]])

# 特徴量の重要度（ツリーベースモデル）
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特徴量の重要度:")
print(feature_importance)
```

## TensorFlow/Keras - ディープラーニング

### 基本的なモデル構築

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# シーケンシャルモデル
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(4,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])

# モデルのコンパイル
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# モデルの概要
model.summary()

# 学習
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 学習曲線の可視化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

### Functional API

```python
# より複雑なモデル構造
inputs = keras.Input(shape=(100,))

# 分岐構造
x = layers.Dense(64, activation='relu')(inputs)

# ブランチ1
branch1 = layers.Dense(32, activation='relu')(x)
branch1 = layers.Dense(16)(branch1)

# ブランチ2
branch2 = layers.Dense(32, activation='relu')(x)
branch2 = layers.Dense(16)(branch2)

# 結合
concatenated = layers.concatenate([branch1, branch2])
outputs = layers.Dense(10, activation='softmax')(concatenated)

# モデルの作成
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# プロット（Graphvizが必要）
keras.utils.plot_model(model, "multi_branch_model.png", show_shapes=True)
```

### カスタムレイヤーとコールバック

```python
# カスタムレイヤー
class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="kernel"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# カスタムコールバック
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss') < 0.1:
            print(f"\nエポック {epoch}: 検証損失が0.1未満になりました。学習を停止します。")
            self.model.stop_training = True

# 使用例
model = keras.Sequential([
    CustomDense(64),
    layers.Activation('relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# コールバックの使用
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
    CustomCallback()
]

# history = model.fit(X_train, y_train, validation_split=0.2, 
#                    epochs=100, callbacks=callbacks)
```

### データジェネレータ

```python
# 画像データのジェネレータ
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# データ拡張付きジェネレータ
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2
)

# カスタムジェネレータ
def custom_generator(X, y, batch_size=32):
    """カスタムデータジェネレータの例"""
    num_samples = len(X)
    
    while True:
        indices = np.random.permutation(num_samples)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            # バッチデータの準備
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # データ拡張やその他の処理
            # ...
            
            yield batch_X, batch_y

# tf.data APIの使用
def create_dataset(X, y, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# 使用例
train_dataset = create_dataset(X_train_scaled, y_train, batch_size=32)
val_dataset = create_dataset(X_test_scaled, y_test, batch_size=32, shuffle=False)
```

## PyTorch - 研究向けディープラーニング

### 基本的なモデル構築

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

# PyTorchモデルの定義
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# モデルのインスタンス化
model = SimpleNet(input_size=4, hidden_size=128, num_classes=3).to(device)
print(model)

# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# データローダーの準備
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 学習ループ
num_epochs = 50
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 順伝播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

# 評価
model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'\nテスト精度: {accuracy:.4f}')
```

### カスタムデータセット

```python
from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image as Image

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 画像の読み込み
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # 変換の適用
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 画像変換の定義
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# データセットとローダーの作成
# dataset = CustomImageDataset(image_paths, labels, transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### 動的計算グラフの活用

```python
# 動的な処理の例
class DynamicNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DynamicNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, use_dropout=True):
        # 条件に応じて処理を変更
        out = torch.relu(self.fc1(x))
        
        if use_dropout and self.training:
            out = torch.dropout(out, p=0.5, train=True)
        
        # ランダムな数の層を追加（例）
        num_extra_layers = torch.randint(1, 4, (1,)).item()
        for i in range(num_extra_layers):
            layer = nn.Linear(self.hidden_size, self.hidden_size).to(x.device)
            out = torch.relu(layer(out))
        
        out = self.fc2(out)
        return out

# 使用例
dynamic_model = DynamicNet(4, 64, 3)
output = dynamic_model(X_train_tensor[:10])
print(f"出力形状: {output.shape}")
```

## その他の便利なライブラリ

### XGBoost - 勾配ブースティング

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# XGBoostモデル
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='multi:softprob',
    random_state=42
)

# 学習
xgb_model.fit(X_train, y_train)

# 交差検証
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"交差検証スコア: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# 特徴量の重要度
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特徴量の重要度:")
print(feature_importance)
```

### LightGBM - 高速勾配ブースティング

```python
import lightgbm as lgb

# LightGBMデータセット
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# パラメータ
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'random_state': 42
}

# 学習
lgb_model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_eval],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(20)]
)

# 予測
y_pred_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
y_pred_lgb = np.argmax(y_pred_lgb, axis=1)
```

### Optuna - ハイパーパラメータ最適化

```python
import optuna

def objective(trial):
    # ハイパーパラメータの提案
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    
    # モデルの作成と学習
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42
    )
    
    # 交差検証
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    return cv_scores.mean()

# 最適化の実行
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"最適パラメータ: {study.best_params}")
print(f"最高スコア: {study.best_value:.3f}")

# 最適化履歴の可視化
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

### MLflow - 実験管理

```python
import mlflow
import mlflow.sklearn

# MLflowの実験開始
mlflow.set_experiment("iris_classification")

with mlflow.start_run():
    # パラメータの記録
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    # モデルの学習
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # メトリクスの記録
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    
    mlflow.log_metric("train_accuracy", train_score)
    mlflow.log_metric("test_accuracy", test_score)
    
    # モデルの保存
    mlflow.sklearn.log_model(rf_model, "model")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

## ライブラリ選択のガイドライン

### 機械学習
- **Scikit-learn**: 標準的な機械学習タスク、プロトタイピング
- **XGBoost/LightGBM**: 構造化データの予測タスク、Kaggleコンペ
- **CatBoost**: カテゴリ変数が多いデータ

### ディープラーニング
- **TensorFlow/Keras**: プロダクション環境、モバイル展開
- **PyTorch**: 研究開発、カスタム実装が必要な場合
- **JAX**: 高速な数値計算、関数型プログラミング

### その他
- **Optuna**: ハイパーパラメータ最適化
- **MLflow**: 実験管理、モデル管理
- **DVC**: データバージョン管理
- **Weights & Biases**: 実験トラッキング、可視化

## まとめ

主要なAIライブラリの特徴：

1. **Scikit-learn**: 機械学習の基本、豊富なアルゴリズム
2. **TensorFlow/Keras**: 本番環境向け、エコシステムが充実
3. **PyTorch**: 研究向け、動的計算グラフ
4. **XGBoost/LightGBM**: 高性能な勾配ブースティング
5. **実験管理ツール**: 再現性とトラッキング

適切なライブラリを選択し、組み合わせることで、効率的なAI開発が可能になります。

## 次へ

[ディープラーニングフレームワーク](../フレームワーク/README.md)へ進む