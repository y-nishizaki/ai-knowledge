# ニューラルネットワークの基礎

## 概要

ニューラルネットワーク（Neural Network）は、人間の脳の神経細胞（ニューロン）の仕組みを模倣した計算モデルです。入力層、隠れ層、出力層から構成され、各層のニューロンが重み付けされた結合で繋がっています。

## パーセプトロン

### 単純パーセプトロン

最も基本的なニューラルネットワークの単位です。

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                
                # 重みの更新
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                
    def activation(self, x):
        return np.where(x >= 0, 1, 0)
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

# AND演算の学習
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND演算の出力

perceptron = Perceptron()
perceptron.fit(X, y)

# 結果の確認
print("学習した重み:", perceptron.weights)
print("バイアス:", perceptron.bias)
print("予測結果:")
for i in range(len(X)):
    print(f"{X[i]} -> {perceptron.predict(X[i].reshape(1, -1))[0]}")
```

### 多層パーセプトロン（MLP）

複数の層を持つニューラルネットワークです。

```python
import tensorflow as tf
from tensorflow import keras

# XOR問題（単純パーセプトロンでは解けない）
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR演算の出力

# MLPモデルの構築
model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 学習
history = model.fit(X, y, epochs=1000, verbose=0)

# 予測結果
predictions = model.predict(X)
print("XOR問題の予測結果:")
for i in range(len(X)):
    print(f"{X[i]} -> {predictions[i][0]:.3f} (実際: {y[i]})")

# 決定境界の可視化
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), 
                     np.linspace(-0.5, 1.5, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='RdBu', edgecolor='black')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('XOR問題の決定境界')
plt.show()
```

## 活性化関数

### 主要な活性化関数

```python
def plot_activation_functions():
    x = np.linspace(-5, 5, 100)
    
    # 各活性化関数の定義
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(x):
        return np.tanh(x)
    
    def relu(x):
        return np.maximum(0, x)
    
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def swish(x):
        return x * sigmoid(x)
    
    # プロット
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    functions = [
        (sigmoid, 'Sigmoid'),
        (tanh, 'Tanh'),
        (relu, 'ReLU'),
        (lambda x: leaky_relu(x), 'Leaky ReLU'),
        (swish, 'Swish'),
        (lambda x: np.where(x > 0, x, np.exp(x) - 1), 'ELU')
    ]
    
    for ax, (func, name) in zip(axes.flat, functions):
        y = func(x)
        ax.plot(x, y, 'b-', linewidth=2)
        ax.grid(True, alpha=0.3)
        ax.set_title(name, fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

plot_activation_functions()
```

### 活性化関数の選び方

```python
# 活性化関数の比較実験
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# データの生成
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 異なる活性化関数でのモデル比較
activation_functions = ['sigmoid', 'tanh', 'relu', 'elu', 'swish']
results = {}

for activation in activation_functions:
    model = keras.Sequential([
        keras.layers.Dense(64, activation=activation, input_shape=(20,)),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                       validation_split=0.2, verbose=0)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    results[activation] = test_acc

# 結果の表示
for activation, accuracy in results.items():
    print(f"{activation}: {accuracy:.3f}")
```

## 誤差逆伝播法（Backpropagation）

### 手動実装による理解

```python
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 重みの初期化（Xavier初期化）
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1/hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[0]
        
        # 出力層の誤差
        self.dz2 = self.a2 - y
        self.dW2 = (1/m) * np.dot(self.a1.T, self.dz2)
        self.db2 = (1/m) * np.sum(self.dz2, axis=0, keepdims=True)
        
        # 隠れ層の誤差
        da1 = np.dot(self.dz2, self.W2.T)
        self.dz1 = da1 * self.sigmoid_derivative(self.z1)
        self.dW1 = (1/m) * np.dot(X.T, self.dz1)
        self.db1 = (1/m) * np.sum(self.dz1, axis=0, keepdims=True)
        
        # 重みの更新
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
    
    def train(self, X, y, epochs=1000, learning_rate=0.1):
        losses = []
        for epoch in range(epochs):
            # 順伝播
            output = self.forward(X)
            
            # 損失計算（二乗誤差）
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            
            # 逆伝播
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses

# XOR問題で検証
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
losses = nn.train(X, y, epochs=1000, learning_rate=0.5)

# 損失の推移
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('学習の進捗')
plt.show()

# 予測結果
predictions = nn.forward(X)
print("\n予測結果:")
for i in range(len(X)):
    print(f"{X[i]} -> {predictions[i][0]:.3f}")
```

## 最適化アルゴリズム

### 各種最適化手法の比較

```python
# 最適化アルゴリズムの可視化
def visualize_optimizers():
    # Rosenbrock関数（最適化のベンチマーク）
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    # 勾配の計算
    def gradient(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])
    
    # 各最適化アルゴリズムの実装
    class SGD:
        def __init__(self, lr=0.001):
            self.lr = lr
            
        def update(self, params, grads):
            return params - self.lr * grads
    
    class Momentum:
        def __init__(self, lr=0.001, momentum=0.9):
            self.lr = lr
            self.momentum = momentum
            self.v = 0
            
        def update(self, params, grads):
            self.v = self.momentum * self.v - self.lr * grads
            return params + self.v
    
    class Adam:
        def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.m = 0
            self.v = 0
            self.t = 0
            
        def update(self, params, grads):
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * grads
            self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)
            return params - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    # 最適化の実行と可視化
    optimizers = {
        'SGD': SGD(lr=0.001),
        'Momentum': Momentum(lr=0.001),
        'Adam': Adam(lr=0.01)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 等高線図の準備
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    for ax, (name, optimizer) in zip(axes, optimizers.items()):
        # 等高線を描画
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
        
        # 最適化の軌跡
        position = np.array([-1.5, 2.5])
        trajectory = [position.copy()]
        
        for _ in range(100):
            grad = gradient(position[0], position[1])
            position = optimizer.update(position, grad)
            trajectory.append(position.copy())
        
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', markersize=3)
        ax.plot(1, 1, 'g*', markersize=15)  # 最適解
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.show()

visualize_optimizers()
```

## 重みの初期化

### 初期化手法の比較

```python
def compare_initializations():
    # 各初期化手法
    def zeros_init(shape):
        return np.zeros(shape)
    
    def random_init(shape):
        return np.random.randn(*shape) * 0.01
    
    def xavier_init(shape):
        return np.random.randn(*shape) * np.sqrt(1 / shape[0])
    
    def he_init(shape):
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])
    
    # 深いネットワークでの活性化値の分布を確認
    initializations = {
        'Zeros': zeros_init,
        'Random': random_init,
        'Xavier': xavier_init,
        'He': he_init
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, (name, init_func) in zip(axes, initializations.items()):
        # 10層のネットワークをシミュレーション
        layer_sizes = [100] * 11
        activations = []
        
        x = np.random.randn(1000, layer_sizes[0])
        
        for i in range(10):
            w = init_func((layer_sizes[i], layer_sizes[i+1]))
            x = np.maximum(0, np.dot(x, w))  # ReLU活性化
            activations.append(x.flatten())
        
        # 各層の活性化値の分布をプロット
        for i, activation in enumerate(activations):
            ax.hist(activation, bins=50, alpha=0.5, density=True, label=f'Layer {i+1}')
        
        ax.set_title(f'{name} Initialization')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Density')
        ax.set_xlim(-2, 2)
        if name == 'Zeros':
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()

compare_initializations()
```

## 正則化手法

### ドロップアウト

```python
class DropoutLayer:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        
    def forward(self, x, training=True):
        if training:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask / (1 - self.dropout_rate)
        else:
            return x
    
    def backward(self, dout):
        return dout * self.mask / (1 - self.dropout_rate)

# ドロップアウトの効果を確認
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ドロップアウトなし
model_no_dropout = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(2,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# ドロップアウトあり
model_with_dropout = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(2,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

# 両モデルを学習
for model, name in [(model_no_dropout, 'Without Dropout'), 
                    (model_with_dropout, 'With Dropout')]:
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                       validation_split=0.2, verbose=0)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
```

### バッチ正規化

```python
# バッチ正規化の実装
class BatchNormalization:
    def __init__(self, gamma=1.0, beta=0.0, momentum=0.9, epsilon=1e-5):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None
        
    def forward(self, x, training=True):
        if self.running_mean is None:
            self.running_mean = np.zeros(x.shape[1])
            self.running_var = np.ones(x.shape[1])
        
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * x_normalized + self.beta

# バッチ正規化の効果
model_with_bn = keras.Sequential([
    keras.layers.Dense(128, input_shape=(2,)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(128),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_with_bn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_bn = model_with_bn.fit(X_train, y_train, epochs=50, batch_size=32, 
                              validation_split=0.2, verbose=0)

print("バッチ正規化モデルの最終精度:", history_bn.history['val_accuracy'][-1])
```

## まとめ

ニューラルネットワークの基礎として、以下を学びました：

1. **パーセプトロン**：最も基本的な構成要素
2. **活性化関数**：非線形性を導入し、表現力を向上
3. **誤差逆伝播法**：効率的な学習アルゴリズム
4. **最適化手法**：SGD、Momentum、Adamなど
5. **正則化**：過学習を防ぐ手法

これらの概念は、より高度なディープラーニングモデルの基礎となります。

## 次へ

[CNN（畳み込みニューラルネットワーク）](../CNN/README.md)へ進む