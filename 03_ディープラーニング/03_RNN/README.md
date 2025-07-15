# RNN（再帰型ニューラルネットワーク）

## 概要

再帰型ニューラルネットワーク（Recurrent Neural Network, RNN）は、時系列データや系列データを扱うために設計されたニューラルネットワークです。過去の情報を記憶し、現在の入力と組み合わせて処理することで、文脈を考慮した予測が可能になります。

## RNNの基本構造

### 基本的なRNNセル

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        # 重みの初期化
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        
    def forward(self, x, h_prev):
        """
        x: 現在の入力
        h_prev: 前の時刻の隠れ状態
        """
        h = np.tanh(np.dot(x, self.Wxh) + np.dot(h_prev, self.Whh) + self.bh)
        return h

# RNNセルの動作確認
rnn_cell = SimpleRNNCell(input_size=10, hidden_size=20)
x = np.random.randn(1, 10)  # バッチサイズ1、入力次元10
h = np.zeros((1, 20))  # 初期隠れ状態

# 5ステップの系列処理
for t in range(5):
    h = rnn_cell.forward(x, h)
    print(f"Time step {t}: hidden state shape = {h.shape}")
```

### RNNの展開

```python
def visualize_rnn_unrolling():
    """RNNの時間展開を可視化"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 時間ステップ
    time_steps = 5
    
    # ノードの位置
    x_positions = np.arange(time_steps) * 2
    
    # 入力層、隠れ層、出力層のy座標
    y_input = 0
    y_hidden = 1
    y_output = 2
    
    # ノードを描画
    for t in range(time_steps):
        # 入力ノード
        ax.scatter(x_positions[t], y_input, s=500, c='lightblue', 
                  edgecolors='black', linewidth=2)
        ax.text(x_positions[t], y_input, f'x{t}', ha='center', va='center')
        
        # 隠れ層ノード
        ax.scatter(x_positions[t], y_hidden, s=500, c='lightgreen', 
                  edgecolors='black', linewidth=2)
        ax.text(x_positions[t], y_hidden, f'h{t}', ha='center', va='center')
        
        # 出力ノード
        ax.scatter(x_positions[t], y_output, s=500, c='lightcoral', 
                  edgecolors='black', linewidth=2)
        ax.text(x_positions[t], y_output, f'y{t}', ha='center', va='center')
        
        # 接続を描画
        # 入力から隠れ層
        ax.arrow(x_positions[t], y_input + 0.1, 0, 0.7, 
                head_width=0.1, head_length=0.05, fc='black')
        
        # 隠れ層から出力
        ax.arrow(x_positions[t], y_hidden + 0.1, 0, 0.7, 
                head_width=0.1, head_length=0.05, fc='black')
        
        # 隠れ層間の接続
        if t < time_steps - 1:
            ax.arrow(x_positions[t] + 0.1, y_hidden, 1.7, 0, 
                    head_width=0.1, head_length=0.1, fc='red', linewidth=2)
    
    ax.set_xlim(-1, x_positions[-1] + 1)
    ax.set_ylim(-0.5, 2.5)
    ax.set_title('RNNの時間展開', fontsize=16)
    ax.axis('off')
    
    # 凡例
    ax.text(-0.5, y_input, '入力層', ha='right', va='center', fontsize=12)
    ax.text(-0.5, y_hidden, '隠れ層', ha='right', va='center', fontsize=12)
    ax.text(-0.5, y_output, '出力層', ha='right', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()

visualize_rnn_unrolling()
```

## 勾配消失・爆発問題

### 問題の可視化

```python
def demonstrate_gradient_problem():
    """勾配消失問題のデモンストレーション"""
    # 長い系列での勾配の変化
    sequence_length = 50
    
    # tanh関数の微分の最大値は1
    gradient_tanh = []
    gradient_value = 1.0
    
    for t in range(sequence_length):
        gradient_value *= 0.5  # 勾配が0.5ずつ減衰すると仮定
        gradient_tanh.append(gradient_value)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(gradient_tanh, label='勾配の大きさ')
    plt.xlabel('時間ステップ')
    plt.ylabel('勾配の大きさ（対数スケール）')
    plt.title('RNNにおける勾配消失問題')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print(f"初期勾配: {gradient_tanh[0]:.6f}")
    print(f"50ステップ後の勾配: {gradient_tanh[-1]:.2e}")

demonstrate_gradient_problem()
```

## LSTM（Long Short-Term Memory）

### LSTMの構造

```python
class SimpleLSTMCell:
    def __init__(self, input_size, hidden_size):
        # 4つのゲート用の重み（入力、忘却、出力、候補）
        self.Wxi = np.random.randn(input_size, hidden_size) * 0.01
        self.Whi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bi = np.zeros((1, hidden_size))
        
        self.Wxf = np.random.randn(input_size, hidden_size) * 0.01
        self.Whf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((1, hidden_size))
        
        self.Wxo = np.random.randn(input_size, hidden_size) * 0.01
        self.Who = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bo = np.zeros((1, hidden_size))
        
        self.Wxc = np.random.randn(input_size, hidden_size) * 0.01
        self.Whc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bc = np.zeros((1, hidden_size))
        
    def forward(self, x, h_prev, c_prev):
        # 入力ゲート
        i = self.sigmoid(np.dot(x, self.Wxi) + np.dot(h_prev, self.Whi) + self.bi)
        
        # 忘却ゲート
        f = self.sigmoid(np.dot(x, self.Wxf) + np.dot(h_prev, self.Whf) + self.bf)
        
        # 出力ゲート
        o = self.sigmoid(np.dot(x, self.Wxo) + np.dot(h_prev, self.Who) + self.bo)
        
        # 候補値
        c_tilde = np.tanh(np.dot(x, self.Wxc) + np.dot(h_prev, self.Whc) + self.bc)
        
        # セル状態の更新
        c = f * c_prev + i * c_tilde
        
        # 隠れ状態の更新
        h = o * np.tanh(c)
        
        return h, c, {'i': i, 'f': f, 'o': o, 'c_tilde': c_tilde}
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# LSTMの動作確認
lstm_cell = SimpleLSTMCell(input_size=10, hidden_size=20)
x = np.random.randn(1, 10)
h = np.zeros((1, 20))
c = np.zeros((1, 20))

# ゲートの値を可視化
gates_history = []
for t in range(10):
    h, c, gates = lstm_cell.forward(x, h, c)
    gates_history.append(gates)

# ゲートの動作を可視化
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
gate_names = ['入力ゲート (i)', '忘却ゲート (f)', '出力ゲート (o)', '候補値 (c̃)']
gate_keys = ['i', 'f', 'o', 'c_tilde']

for ax, name, key in zip(axes.flat, gate_names, gate_keys):
    values = [g[key][0, :5] for g in gates_history]  # 最初の5次元のみ表示
    ax.imshow(np.array(values).T, aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('時間ステップ')
    ax.set_ylabel('隠れ次元')
    ax.set_title(name)
    
plt.tight_layout()
plt.show()
```

### Kerasを使ったLSTMの実装

```python
# 簡単な時系列予測タスク
def generate_sine_wave(n_samples, n_timesteps, n_features=1):
    """サイン波の時系列データを生成"""
    X = np.zeros((n_samples, n_timesteps, n_features))
    y = np.zeros((n_samples, 1))
    
    for i in range(n_samples):
        start = np.random.uniform(0, 2*np.pi)
        for t in range(n_timesteps):
            X[i, t, 0] = np.sin(start + t * 0.1)
        y[i, 0] = np.sin(start + n_timesteps * 0.1)
    
    return X, y

# データの生成
X_train, y_train = generate_sine_wave(1000, 50)
X_test, y_test = generate_sine_wave(200, 50)

# LSTMモデルの構築
model = keras.Sequential([
    keras.layers.LSTM(50, activation='tanh', input_shape=(50, 1)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 学習
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

# 予測結果の可視化
predictions = model.predict(X_test[:10])

plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(X_test[i, :, 0], 'b-', label='入力系列')
    plt.axvline(x=49, color='k', linestyle='--', alpha=0.5)
    plt.plot(50, y_test[i, 0], 'go', markersize=10, label='正解')
    plt.plot(50, predictions[i, 0], 'ro', markersize=10, label='予測')
    plt.xlabel('時間ステップ')
    plt.ylabel('値')
    plt.title(f'サンプル {i+1}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## GRU（Gated Recurrent Unit）

### GRUの実装

```python
class SimpleGRUCell:
    def __init__(self, input_size, hidden_size):
        # リセットゲート
        self.Wxr = np.random.randn(input_size, hidden_size) * 0.01
        self.Whr = np.random.randn(hidden_size, hidden_size) * 0.01
        self.br = np.zeros((1, hidden_size))
        
        # 更新ゲート
        self.Wxz = np.random.randn(input_size, hidden_size) * 0.01
        self.Whz = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bz = np.zeros((1, hidden_size))
        
        # 候補隠れ状態
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        
    def forward(self, x, h_prev):
        # リセットゲート
        r = self.sigmoid(np.dot(x, self.Wxr) + np.dot(h_prev, self.Whr) + self.br)
        
        # 更新ゲート
        z = self.sigmoid(np.dot(x, self.Wxz) + np.dot(h_prev, self.Whz) + self.bz)
        
        # 候補隠れ状態
        h_tilde = np.tanh(np.dot(x, self.Wxh) + np.dot(r * h_prev, self.Whh) + self.bh)
        
        # 隠れ状態の更新
        h = (1 - z) * h_prev + z * h_tilde
        
        return h
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# LSTMとGRUの比較
models = {
    'LSTM': keras.Sequential([
        keras.layers.LSTM(50, input_shape=(50, 1)),
        keras.layers.Dense(1)
    ]),
    'GRU': keras.Sequential([
        keras.layers.GRU(50, input_shape=(50, 1)),
        keras.layers.Dense(1)
    ]),
    'Simple RNN': keras.Sequential([
        keras.layers.SimpleRNN(50, input_shape=(50, 1)),
        keras.layers.Dense(1)
    ])
}

# 各モデルを学習
results = {}
for name, model in models.items():
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, 
                       epochs=30, 
                       batch_size=32,
                       validation_split=0.2,
                       verbose=0)
    
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    results[name] = {
        'history': history.history,
        'test_loss': test_loss
    }
    print(f"{name} - Test Loss: {test_loss:.4f}")

# 学習曲線の比較
plt.figure(figsize=(10, 6))
for name, result in results.items():
    plt.plot(result['history']['val_loss'], label=name)

plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('RNN vs LSTM vs GRU')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 双方向RNN

### Bidirectional LSTMの実装

```python
# テキスト分類のための双方向LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# サンプルデータの準備
texts = [
    "This movie is great",
    "I love this film",
    "Terrible movie",
    "Worst film ever",
    "Amazing story and acting",
    "Boring and predictable"
]
labels = [1, 1, 0, 0, 1, 0]  # 1: ポジティブ, 0: ネガティブ

# テキストの前処理
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=10)
y = np.array(labels)

# 双方向LSTMモデル
model = keras.Sequential([
    keras.layers.Embedding(len(tokenizer.word_index) + 1, 16, input_length=10),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 双方向の効果を可視化
def visualize_bidirectional():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 系列の長さ
    seq_length = 5
    
    # 前方向LSTM
    for i in range(seq_length):
        ax.arrow(i, 1, 0.8, 0, head_width=0.1, head_length=0.1, 
                fc='blue', ec='blue', linewidth=2)
        ax.scatter(i, 1, s=300, c='lightblue', edgecolors='blue', linewidth=2)
        ax.text(i, 1, f't{i}', ha='center', va='center')
    
    # 後方向LSTM
    for i in range(seq_length):
        ax.arrow(seq_length-i-1, -1, -0.8, 0, head_width=0.1, head_length=0.1, 
                fc='red', ec='red', linewidth=2)
        ax.scatter(i, -1, s=300, c='lightcoral', edgecolors='red', linewidth=2)
        ax.text(i, -1, f't{i}', ha='center', va='center')
    
    # 結合層
    for i in range(seq_length):
        ax.plot([i, i], [-0.7, 0.7], 'k--', alpha=0.5)
        ax.scatter(i, 0, s=400, c='lightgreen', edgecolors='black', linewidth=2)
        ax.text(i, 0, 'concat', ha='center', va='center', fontsize=8)
    
    ax.set_xlim(-1, seq_length)
    ax.set_ylim(-2, 2)
    ax.set_title('双方向LSTM', fontsize=16)
    ax.text(-0.5, 1, '前方向', ha='right', va='center', color='blue', fontsize=12)
    ax.text(-0.5, -1, '後方向', ha='right', va='center', color='red', fontsize=12)
    ax.axis('off')
    plt.show()

visualize_bidirectional()
```

## Sequence-to-Sequence（Seq2Seq）

### エンコーダ・デコーダモデル

```python
# 簡単な翻訳タスクの例（数字を英語に変換）
def create_seq2seq_data():
    # 入力: 数字の列、出力: 英語の数字
    input_texts = []
    target_texts = []
    
    num_to_word = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    
    for _ in range(1000):
        num_str = ''.join([str(np.random.randint(0, 10)) for _ in range(3)])
        word_str = ' '.join([num_to_word[n] for n in num_str])
        
        input_texts.append(num_str)
        target_texts.append('\t' + word_str + '\n')
    
    return input_texts, target_texts

# データの準備
input_texts, target_texts = create_seq2seq_data()

# 文字レベルのトークン化
input_characters = set(''.join(input_texts))
target_characters = set(''.join(target_texts))

input_token_index = {char: i for i, char in enumerate(sorted(input_characters))}
target_token_index = {char: i for i, char in enumerate(sorted(target_characters))}

# Seq2Seqモデルの構築
latent_dim = 256

# エンコーダ
encoder_inputs = keras.Input(shape=(None, len(input_characters)))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# デコーダ
decoder_inputs = keras.Input(shape=(None, len(target_characters)))
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(len(target_characters), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# モデルの定義
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print("Seq2Seqモデルの構造:")
model.summary()
```

## 実践的なRNNの応用

### 1. テキスト生成

```python
# 文字レベルのテキスト生成
def prepare_text_generation_data(text, seq_length=40):
    # ユニークな文字の抽出
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # 訓練データの作成
    X = []
    y = []
    
    for i in range(0, len(text) - seq_length):
        X.append([char_to_idx[ch] for ch in text[i:i + seq_length]])
        y.append(char_to_idx[text[i + seq_length]])
    
    return np.array(X), np.array(y), char_to_idx, idx_to_char

# サンプルテキスト
sample_text = """
人工知能は人間の知的能力をコンピュータ上で実現する技術です。
機械学習は経験からパターンを学習します。
ディープラーニングは多層のニューラルネットワークを使用します。
"""

X, y, char_to_idx, idx_to_char = prepare_text_generation_data(sample_text, seq_length=20)

# モデルの構築
vocab_size = len(char_to_idx)
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 50, input_length=20),
    keras.layers.LSTM(128),
    keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# テキスト生成関数
def generate_text(model, start_text, length=100, temperature=1.0):
    generated = start_text
    
    for _ in range(length):
        # 入力の準備
        x = np.zeros((1, 20))
        for t, char in enumerate(generated[-20:]):
            if char in char_to_idx:
                x[0, t] = char_to_idx[char]
        
        # 予測
        predictions = model.predict(x, verbose=0)[0]
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        
        # サンプリング
        predicted_idx = np.random.choice(len(predictions), p=predictions)
        generated += idx_to_char[predicted_idx]
    
    return generated
```

### 2. 時系列異常検知

```python
# LSTMオートエンコーダによる異常検知
def create_lstm_autoencoder(sequence_length, n_features):
    model = keras.Sequential([
        # エンコーダ
        keras.layers.LSTM(32, activation='relu', input_shape=(sequence_length, n_features), 
                         return_sequences=True),
        keras.layers.LSTM(16, activation='relu', return_sequences=False),
        
        # デコーダ
        keras.layers.RepeatVector(sequence_length),
        keras.layers.LSTM(16, activation='relu', return_sequences=True),
        keras.layers.LSTM(32, activation='relu', return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(n_features))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# 正常データの生成
normal_data = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000)

# 異常データの挿入
anomaly_data = normal_data.copy()
anomaly_data[400:420] += 2  # 異常値

# 系列データの作成
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

sequences = create_sequences(normal_data, 50)
X_train = sequences[:800].reshape(-1, 50, 1)
X_test = create_sequences(anomaly_data, 50).reshape(-1, 50, 1)

# モデルの学習
autoencoder = create_lstm_autoencoder(50, 1)
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)

# 異常検知
reconstructed = autoencoder.predict(X_test)
mse = np.mean((X_test - reconstructed) ** 2, axis=(1, 2))

# 結果の可視化
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(anomaly_data, label='データ')
plt.axvspan(400, 420, alpha=0.3, color='red', label='異常区間')
plt.legend()
plt.title('時系列データ')

plt.subplot(2, 1, 2)
plt.plot(mse)
threshold = np.percentile(mse, 95)
plt.axhline(y=threshold, color='r', linestyle='--', label='閾値')
plt.fill_between(range(len(mse)), 0, mse, where=mse > threshold, 
                 alpha=0.3, color='red', label='異常検知')
plt.legend()
plt.title('再構成誤差')
plt.tight_layout()
plt.show()
```

## Attention機構

### 簡単なAttentionの実装

```python
class SimpleAttention(keras.layers.Layer):
    def __init__(self, units):
        super(SimpleAttention, self).__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)
        
    def call(self, encoder_output, decoder_hidden):
        # encoder_output: (batch_size, seq_len, hidden_dim)
        # decoder_hidden: (batch_size, hidden_dim)
        
        # decoder_hiddenを拡張
        decoder_hidden_expanded = tf.expand_dims(decoder_hidden, 1)
        
        # スコアの計算
        score = self.V(tf.nn.tanh(
            self.W1(encoder_output) + self.W2(decoder_hidden_expanded)
        ))
        
        # Attention重みの計算
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vectorの計算
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

# Attention付きモデルの例
encoder_input = keras.Input(shape=(20, 100))
encoder_lstm = keras.layers.LSTM(256, return_sequences=True, return_state=True)
encoder_output, state_h, state_c = encoder_lstm(encoder_input)

decoder_input = keras.Input(shape=(None, 100))
decoder_lstm = keras.layers.LSTM(256, return_sequences=True)
decoder_output = decoder_lstm(decoder_input, initial_state=[state_h, state_c])

# Attentionレイヤー
attention = SimpleAttention(256)
context_vector, attention_weights = attention(encoder_output, state_h)

print("Attention実装完了")
```

## まとめ

RNNとその発展形について学びました：

1. **基本的なRNN**：系列データの処理が可能だが、長期依存関係の学習が困難
2. **LSTM**：ゲート機構により長期記憶が可能
3. **GRU**：LSTMの簡略版で、パラメータ数が少ない
4. **双方向RNN**：過去と未来の両方の文脈を考慮
5. **Seq2Seq**：系列から系列への変換
6. **Attention機構**：重要な部分に注目する仕組み

これらの技術は自然言語処理、時系列予測、音声認識など幅広い分野で使用されています。

## 次へ

[Transformer](../04_Transformer/README.md)へ進む