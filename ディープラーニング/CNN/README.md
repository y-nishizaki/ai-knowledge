# CNN（畳み込みニューラルネットワーク）

## 概要

畳み込みニューラルネットワーク（Convolutional Neural Network, CNN）は、画像認識に特化したニューラルネットワークです。局所的な特徴を階層的に抽出することで、高精度な画像分類を実現します。

## CNNの基本構造

### 主要な層

1. **畳み込み層（Convolutional Layer）**：特徴マップの抽出
2. **プーリング層（Pooling Layer）**：特徴マップの縮小
3. **全結合層（Fully Connected Layer）**：最終的な分類

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 簡単なCNNの構造を可視化
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()
```

## 畳み込み層の仕組み

### 畳み込み演算の実装

```python
def convolution2d(image, kernel, stride=1, padding=0):
    """
    2次元畳み込み演算の実装
    """
    # パディングの追加
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    
    # 出力サイズの計算
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    
    # 畳み込み演算
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image[i*stride:i*stride+kh, j*stride:j*stride+kw] * kernel)
    
    return output

# 畳み込みの例
# サンプル画像（エッジ検出）
image = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
], dtype=float)

# エッジ検出カーネル
kernel_edge = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# 畳み込み演算
result = convolution2d(image, kernel_edge, padding=1)

# 可視化
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('元画像')
axes[0].axis('off')

axes[1].imshow(kernel_edge, cmap='RdBu')
axes[1].set_title('カーネル（エッジ検出）')
axes[1].axis('off')

axes[2].imshow(result, cmap='gray')
axes[2].set_title('畳み込み結果')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

### 様々なフィルタ

```python
# 各種フィルタの効果を確認
filters = {
    'エッジ検出（水平）': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    'エッジ検出（垂直）': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    'ぼかし': np.ones((3, 3)) / 9,
    'シャープ化': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'エンボス': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
}

# より複雑な画像を作成
complex_image = np.zeros((50, 50))
complex_image[10:20, 10:40] = 1
complex_image[30:40, 20:30] = 1
complex_image[15:35, 15:20] = 0.5

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

axes[0].imshow(complex_image, cmap='gray')
axes[0].set_title('元画像')
axes[0].axis('off')

for idx, (name, filt) in enumerate(filters.items(), 1):
    filtered = convolution2d(complex_image, filt, padding=1)
    axes[idx].imshow(filtered, cmap='gray')
    axes[idx].set_title(name)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
```

## プーリング層

### MaxPoolingとAveragePooling

```python
def max_pooling2d(image, pool_size=2, stride=2):
    """MaxPoolingの実装"""
    h, w = image.shape
    out_h = h // stride
    out_w = w // stride
    
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.max(image[i*stride:i*stride+pool_size, 
                                       j*stride:j*stride+pool_size])
    return output

def average_pooling2d(image, pool_size=2, stride=2):
    """AveragePoolingの実装"""
    h, w = image.shape
    out_h = h // stride
    out_w = w // stride
    
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.mean(image[i*stride:i*stride+pool_size, 
                                        j*stride:j*stride+pool_size])
    return output

# プーリングの効果を確認
sample_feature_map = np.random.rand(8, 8)

max_pooled = max_pooling2d(sample_feature_map)
avg_pooled = average_pooling2d(sample_feature_map)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(sample_feature_map, cmap='viridis')
axes[0].set_title('元の特徴マップ (8x8)')
axes[0].grid(True, alpha=0.3)

axes[1].imshow(max_pooled, cmap='viridis')
axes[1].set_title('Max Pooling後 (4x4)')
axes[1].grid(True, alpha=0.3)

axes[2].imshow(avg_pooled, cmap='viridis')
axes[2].set_title('Average Pooling後 (4x4)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## MNISTでの実践

### データの準備と前処理

```python
# MNISTデータセットの読み込み
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# データの前処理
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# サンプル画像の表示
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {np.argmax(y_train[i])}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### CNNモデルの構築と学習

```python
# CNNモデルの構築
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 学習
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    validation_split=0.1,
                    verbose=1)

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

# テストデータでの評価
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')
```

### 特徴マップの可視化

```python
# 中間層の出力を取得するモデルを作成
layer_outputs = [layer.output for layer in model.layers[:6]]  # 最初の6層
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

# テスト画像での活性化を取得
img = x_test[0:1]
activations = activation_model.predict(img)

# 各層の特徴マップを可視化
layer_names = ['Conv2D_1', 'MaxPooling2D_1', 'Conv2D_2', 'MaxPooling2D_2', 'Conv2D_3']

for layer_name, layer_activation in zip(layer_names[:5], activations[:5]):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    
    # 最大16個の特徴マップを表示
    n_cols = 8
    n_features_to_show = min(n_features, 16)
    n_rows = n_features_to_show // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 1.5))
    
    for i in range(n_features_to_show):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
            
        ax.imshow(layer_activation[0, :, :, i], cmap='viridis')
        ax.axis('off')
    
    fig.suptitle(f'{layer_name} - {n_features} features (showing {n_features_to_show})')
    plt.tight_layout()
    plt.show()
```

## 有名なCNNアーキテクチャ

### LeNet-5 (1998)

```python
def build_lenet5():
    model = keras.Sequential([
        keras.layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 1)),
        keras.layers.AveragePooling2D((2, 2)),
        keras.layers.Conv2D(16, (5, 5), activation='tanh'),
        keras.layers.AveragePooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation='tanh'),
        keras.layers.Dense(84, activation='tanh'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

lenet = build_lenet5()
lenet.summary()
```

### AlexNet (2012)

```python
def build_alexnet():
    model = keras.Sequential([
        keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu', 
                           input_shape=(227, 227, 3)),
        keras.layers.MaxPooling2D((3, 3), strides=2),
        keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((3, 3), strides=2),
        keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((3, 3), strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1000, activation='softmax')
    ])
    return model
```

### VGG16 (2014)

```python
def build_vgg_block(filters, n_conv):
    """VGGスタイルのブロック"""
    layers = []
    for _ in range(n_conv):
        layers.append(keras.layers.Conv2D(filters, (3, 3), 
                                         activation='relu', padding='same'))
    layers.append(keras.layers.MaxPooling2D((2, 2)))
    return layers

def build_vgg16():
    model = keras.Sequential([
        # 入力層
        keras.layers.Input(shape=(224, 224, 3)),
        
        # Block 1
        *build_vgg_block(64, 2),
        
        # Block 2
        *build_vgg_block(128, 2),
        
        # Block 3
        *build_vgg_block(256, 3),
        
        # Block 4
        *build_vgg_block(512, 3),
        
        # Block 5
        *build_vgg_block(512, 3),
        
        # 全結合層
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1000, activation='softmax')
    ])
    return model
```

## 転移学習

### 事前学習済みモデルの活用

```python
# 転移学習の例（VGG16を使用）
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 事前学習済みVGG16モデルの読み込み（最上位層を除く）
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# ベースモデルの重みを固定
base_model.trainable = False

# 新しいモデルの構築
transfer_model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')  # 10クラス分類の例
])

transfer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

transfer_model.summary()
```

### ファインチューニング

```python
# ファインチューニングの例
def fine_tune_model(base_model, n_classes):
    # ベースモデルの一部の層を解凍
    base_model.trainable = True
    
    # 最後の数層のみ学習可能にする
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # モデルの構築
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    # 学習率を下げてコンパイル
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

## データ拡張

### ImageDataGeneratorの使用

```python
# データ拡張の設定
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# サンプル画像での拡張例を表示
sample_image = x_train[0].reshape(1, 28, 28, 1)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    augmented = datagen.flow(sample_image, batch_size=1)[0]
    ax.imshow(augmented[0].reshape(28, 28), cmap='gray')
    ax.axis('off')

plt.suptitle('データ拡張の例')
plt.tight_layout()
plt.show()
```

## 実践的なTips

### 1. 受容野の計算

```python
def calculate_receptive_field(layers):
    """CNNの受容野を計算"""
    rf = 1  # 初期受容野
    stride = 1  # 累積ストライド
    
    for layer in layers:
        if 'conv' in layer['type']:
            rf = rf + (layer['kernel_size'] - 1) * stride
            stride *= layer.get('stride', 1)
        elif 'pool' in layer['type']:
            rf = rf + (layer['pool_size'] - 1) * stride
            stride *= layer.get('stride', layer['pool_size'])
    
    return rf

# 例：シンプルなCNNの受容野
layers = [
    {'type': 'conv', 'kernel_size': 3, 'stride': 1},
    {'type': 'pool', 'pool_size': 2, 'stride': 2},
    {'type': 'conv', 'kernel_size': 3, 'stride': 1},
    {'type': 'pool', 'pool_size': 2, 'stride': 2},
    {'type': 'conv', 'kernel_size': 3, 'stride': 1}
]

print(f"受容野: {calculate_receptive_field(layers)}ピクセル")
```

### 2. Grad-CAMによる可視化

```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Grad-CAMヒートマップの生成"""
    # 勾配を計算するモデルを作成
    grad_model = keras.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # 勾配を計算
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # 正規化
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```

## まとめ

CNNの主要な概念：

1. **畳み込み層**：局所的な特徴を抽出
2. **プーリング層**：特徴マップのダウンサンプリング
3. **階層的な特徴抽出**：低次から高次の特徴へ
4. **転移学習**：事前学習済みモデルの活用
5. **データ拡張**：過学習の防止

CNNは画像認識において最も成功したアーキテクチャであり、コンピュータビジョンの基礎となっています。

## 次へ

[RNN（再帰型ニューラルネットワーク）](../RNN/README.md)へ進む