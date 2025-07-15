# Transformer

## 概要

Transformerは2017年に発表された「Attention Is All You Need」論文で提案されたアーキテクチャです。RNNやCNNを使わず、Attention機構のみで構成され、並列処理が可能で高速な学習を実現しました。現在のNLP分野の主流となっており、BERT、GPT、T5などの基盤技術です。

## Self-Attention機構

### Scaled Dot-Product Attention

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Scaled Dot-Product Attentionの実装
    
    Args:
        query: (batch_size, seq_len_q, d_k)
        key: (batch_size, seq_len_k, d_k)
        value: (batch_size, seq_len_v, d_v)
        mask: (batch_size, seq_len_q, seq_len_k)
    
    Returns:
        output: (batch_size, seq_len_q, d_v)
        attention_weights: (batch_size, seq_len_q, seq_len_k)
    """
    
    # QとKの内積を計算
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    
    # スケーリング
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # マスクの適用
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # Softmaxで正規化
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    # 重み付き和の計算
    output = tf.matmul(attention_weights, value)
    
    return output, attention_weights

# 動作確認
batch_size = 1
seq_len = 4
d_model = 8

# ランダムなQ, K, Vを生成
query = tf.random.normal((batch_size, seq_len, d_model))
key = tf.random.normal((batch_size, seq_len, d_model))
value = tf.random.normal((batch_size, seq_len, d_model))

output, attention_weights = scaled_dot_product_attention(query, key, value)

# Attention重みの可視化
plt.figure(figsize=(6, 6))
plt.imshow(attention_weights[0], cmap='Blues')
plt.colorbar()
plt.xlabel('Key positions')
plt.ylabel('Query positions')
plt.title('Attention Weights')
for i in range(seq_len):
    for j in range(seq_len):
        plt.text(j, i, f'{attention_weights[0, i, j]:.2f}', 
                ha='center', va='center')
plt.show()
```

### Multi-Head Attention

```python
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)
        
        self.dense = keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """入力を複数のヘッドに分割"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]
        
        # 線形変換
        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)
        
        # ヘッドに分割
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Scaled dot-product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(
            query, key, value, mask)
        
        # ヘッドを結合
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, 
                                     (batch_size, -1, self.d_model))
        
        # 最終的な線形変換
        output = self.dense(concat_attention)
        
        return output, attention_weights

# Multi-Head Attentionの動作確認
mha = MultiHeadAttention(d_model=512, num_heads=8)
temp_input = tf.random.normal((1, 60, 512))
output, attn = mha(temp_input, temp_input, temp_input, mask=None)
print(f"Multi-Head Attention出力形状: {output.shape}")
print(f"Attention重み形状: {attn.shape}")
```

## Positional Encoding

```python
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)
    
    # 偶数インデックスにはsin、奇数インデックスにはcosを適用
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

# Positional Encodingの可視化
pos_encoding = positional_encoding(50, 512)

plt.figure(figsize=(12, 8))
plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Embedding Dimensions')
plt.ylabel('Position')
plt.colorbar()
plt.title('Positional Encoding')
plt.show()

# 特定の次元での位置エンコーディングを可視化
plt.figure(figsize=(12, 6))
positions = np.arange(50)
for i in [0, 1, 2, 3, 100, 101, 102, 103]:
    plt.plot(positions, pos_encoding[0, :, i], label=f'dim {i}')
plt.legend()
plt.xlabel('Position')
plt.ylabel('Positional Encoding Value')
plt.title('Positional Encoding for Different Dimensions')
plt.grid(True, alpha=0.3)
plt.show()
```

## Transformerブロックの実装

### Encoder層

```python
class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dense(d_model)
        ])
        
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
    
    def call(self, x, training, mask=None):
        # Multi-head attention (Self-attention)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # 残差接続とLayer Norm
        
        # Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # 残差接続とLayer Norm
        
        return out2

# Encoder層の動作確認
encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)
sample_input = tf.random.normal((64, 50, 512))
sample_output = encoder_layer(sample_input, training=False)
print(f"Encoder層出力形状: {sample_output.shape}")
```

### Decoder層

```python
class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dense(d_model)
        ])
        
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        # Masked self-attention
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        # Encoder-decoder attention
        attn2, attn_weights_block2 = self.mha2(
            out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        # Feed forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3, attn_weights_block1, attn_weights_block2
```

## 完全なTransformerモデル

```python
class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        
        # Encoder
        self.encoder_embedding = keras.layers.Embedding(input_vocab_size, d_model)
        self.encoder_pos_encoding = positional_encoding(pe_input, d_model)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                              for _ in range(num_layers)]
        self.encoder_dropout = keras.layers.Dropout(rate)
        
        # Decoder
        self.decoder_embedding = keras.layers.Embedding(target_vocab_size, d_model)
        self.decoder_pos_encoding = positional_encoding(pe_target, d_model)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                              for _ in range(num_layers)]
        self.decoder_dropout = keras.layers.Dropout(rate)
        
        self.final_layer = keras.layers.Dense(target_vocab_size)
    
    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]
    
    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    
    def call(self, inputs, training):
        inp, tar = inputs
        
        # マスクの作成
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        # Encoder
        seq_len = tf.shape(inp)[1]
        x = self.encoder_embedding(inp)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.encoder_pos_encoding[:, :seq_len, :]
        x = self.encoder_dropout(x, training=training)
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training, enc_padding_mask)
        
        enc_output = x
        
        # Decoder
        seq_len = tf.shape(tar)[1]
        x = self.decoder_embedding(tar)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.decoder_pos_encoding[:, :seq_len, :]
        x = self.decoder_dropout(x, training=training)
        
        for decoder_layer in self.decoder_layers:
            x, _, _ = decoder_layer(x, enc_output, training, 
                                   combined_mask, dec_padding_mask)
        
        final_output = self.final_layer(x)
        
        return final_output
```

## BERTの基本実装

```python
class BERTModel(keras.Model):
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072,
                 max_position_embeddings=512):
        super(BERTModel, self).__init__()
        
        # Embeddings
        self.word_embeddings = keras.layers.Embedding(
            vocab_size, hidden_size, name="word_embeddings"
        )
        self.position_embeddings = keras.layers.Embedding(
            max_position_embeddings, hidden_size, name="position_embeddings"
        )
        self.token_type_embeddings = keras.layers.Embedding(
            2, hidden_size, name="token_type_embeddings"
        )
        
        self.embedding_norm = keras.layers.LayerNormalization(
            epsilon=1e-12, name="embeddings/LayerNorm"
        )
        self.embedding_dropout = keras.layers.Dropout(0.1)
        
        # Encoder layers
        self.encoder_layers = []
        for i in range(num_hidden_layers):
            self.encoder_layers.append(
                EncoderLayer(hidden_size, num_attention_heads, 
                           intermediate_size, rate=0.1)
            )
        
        # Pooler
        self.pooler = keras.layers.Dense(
            hidden_size,
            activation="tanh",
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="pooler/dense"
        )
    
    def call(self, inputs, training=False):
        input_ids, token_type_ids = inputs
        
        seq_length = tf.shape(input_ids)[1]
        position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        
        # Embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings, training=training)
        
        # Encoder
        hidden_states = embeddings
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states, training=training)
        
        # Pooling
        pooled_output = self.pooler(hidden_states[:, 0])
        
        return hidden_states, pooled_output

# BERTの使用例
bert = BERTModel(vocab_size=30000)
input_ids = tf.constant([[101, 2023, 2003, 1037, 3231, 102]])
token_type_ids = tf.constant([[0, 0, 0, 0, 0, 0]])

sequence_output, pooled_output = bert((input_ids, token_type_ids))
print(f"Sequence output shape: {sequence_output.shape}")
print(f"Pooled output shape: {pooled_output.shape}")
```

## GPTスタイルのモデル

```python
class GPTModel(keras.Model):
    def __init__(self, vocab_size, n_positions=1024, n_embd=768, 
                 n_layer=12, n_head=12):
        super(GPTModel, self).__init__()
        
        self.wte = keras.layers.Embedding(vocab_size, n_embd)
        self.wpe = keras.layers.Embedding(n_positions, n_embd)
        self.drop = keras.layers.Dropout(0.1)
        
        self.blocks = [self._build_block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = keras.layers.LayerNormalization(epsilon=1e-5)
        
    def _build_block(self, n_embd, n_head):
        return {
            'ln_1': keras.layers.LayerNormalization(epsilon=1e-5),
            'attn': MultiHeadAttention(n_embd, n_head),
            'ln_2': keras.layers.LayerNormalization(epsilon=1e-5),
            'mlp': keras.Sequential([
                keras.layers.Dense(4 * n_embd, activation='gelu'),
                keras.layers.Dense(n_embd)
            ])
        }
    
    def call(self, inputs, training=False):
        input_ids = inputs
        batch_size, sequence_length = tf.shape(input_ids)[0], tf.shape(input_ids)[1]
        
        # Embeddings
        position_ids = tf.range(sequence_length)[tf.newaxis, :]
        
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states, training=training)
        
        # 因果的マスクの作成
        attention_mask = tf.linalg.band_part(
            tf.ones((sequence_length, sequence_length)), -1, 0
        )
        attention_mask = tf.cast(attention_mask, tf.float32)
        attention_mask = (1.0 - attention_mask) * -1e9
        
        # Transformer blocks
        for block in self.blocks:
            residual = hidden_states
            hidden_states = block['ln_1'](hidden_states)
            attn_output, _ = block['attn'](
                hidden_states, hidden_states, hidden_states, attention_mask
            )
            hidden_states = residual + attn_output
            
            residual = hidden_states
            hidden_states = block['ln_2'](hidden_states)
            feed_forward_output = block['mlp'](hidden_states)
            hidden_states = residual + feed_forward_output
        
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states
```

## Vision Transformer (ViT)

```python
class VisionTransformer(keras.Model):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 dim=768, depth=12, heads=12, mlp_dim=3072):
        super(VisionTransformer, self).__init__()
        
        assert image_size % patch_size == 0
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = self.add_weight(
            "pos_embedding", 
            shape=(1, num_patches + 1, dim),
            initializer="random_normal"
        )
        self.cls_token = self.add_weight(
            "cls_token",
            shape=(1, 1, dim),
            initializer="random_normal"
        )
        
        self.patch_projection = keras.layers.Dense(dim)
        self.dropout = keras.layers.Dropout(0.1)
        
        self.transformer_blocks = [
            EncoderLayer(dim, heads, mlp_dim, rate=0.1) 
            for _ in range(depth)
        ]
        
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp_head = keras.Sequential([
            keras.layers.Dense(mlp_dim, activation='gelu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(num_classes)
        ])
    
    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, patches.shape[-1]])
        return patches
    
    def call(self, images, training=False):
        batch_size = tf.shape(images)[0]
        
        # パッチに分割
        x = self.extract_patches(images)
        x = self.patch_projection(x)
        
        # CLSトークンを追加
        cls_tokens = tf.broadcast_to(
            self.cls_token, [batch_size, 1, self.cls_token.shape[-1]]
        )
        x = tf.concat([cls_tokens, x], axis=1)
        
        # 位置エンコーディングを追加
        x += self.pos_embedding
        x = self.dropout(x, training=training)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        x = self.layer_norm(x)
        
        # 分類ヘッド（CLSトークンの出力を使用）
        x = self.mlp_head(x[:, 0])
        
        return x

# ViTの使用例
vit = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_classes=10,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072
)

# ダミー画像
dummy_images = tf.random.normal((4, 224, 224, 3))
output = vit(dummy_images)
print(f"ViT output shape: {output.shape}")
```

## 実践的な応用

### 1. 機械翻訳

```python
# 簡単な翻訳タスクの例
class TranslationTransformer(keras.Model):
    def __init__(self, input_vocab_size, target_vocab_size, 
                 d_model=128, num_heads=8, num_layers=4):
        super(TranslationTransformer, self).__init__()
        
        self.transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=512,
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            pe_input=1000,
            pe_target=1000
        )
    
    def call(self, inputs, training=False):
        return self.transformer(inputs, training)

# カスタム学習率スケジュール
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
        arg2 = tf.cast(step, tf.float32) * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(tf.cast(self.d_model, tf.float32)) * tf.math.minimum(arg1, arg2)

# 学習率スケジュールの可視化
d_model = 128
learning_rate = CustomSchedule(d_model)

plt.figure(figsize=(10, 6))
plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.title("Custom Learning Rate Schedule")
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. テキスト生成

```python
def generate_text_transformer(model, tokenizer, seed_text, max_length=50, temperature=1.0):
    """Transformerモデルを使ったテキスト生成"""
    
    for _ in range(max_length):
        # 入力をトークン化
        tokenized = tokenizer.encode(seed_text)
        tokenized = tf.expand_dims(tokenized, 0)
        
        # 予測
        predictions = model(tokenized, training=False)
        predictions = predictions[0, -1, :] / temperature
        
        # サンプリング
        predicted_id = tf.random.categorical(predictions[tf.newaxis, :], num_samples=1)
        predicted_id = tf.squeeze(predicted_id, axis=[0, 1]).numpy()
        
        # 生成されたトークンを追加
        seed_text += tokenizer.decode([predicted_id])
        
    return seed_text
```

## Transformerの利点と課題

### 利点
1. **並列処理が可能**：RNNと異なり、系列全体を同時に処理
2. **長距離依存関係の学習**：Self-Attentionにより任意の位置間の関係を直接学習
3. **転移学習に適している**：事前学習済みモデルの活用が容易

### 課題
1. **計算量**：系列長の2乗に比例する計算量
2. **メモリ使用量**：長い系列では大量のメモリが必要
3. **位置情報**：明示的な位置エンコーディングが必要

### 最新の発展

```python
# Efficient Transformerの例（Linformer風の実装）
class EfficientAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads, k=256):
        super(EfficientAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.k = k  # 圧縮後の系列長
        
        self.depth = d_model // num_heads
        
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)
        
        # 投影行列
        self.E = keras.layers.Dense(k)
        self.F = keras.layers.Dense(k)
        
        self.dense = keras.layers.Dense(d_model)
    
    def call(self, query, key, value):
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]
        
        # Q, K, Vの計算
        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)
        
        # KとVを圧縮
        K = self.E(tf.transpose(K, [0, 2, 1]))
        K = tf.transpose(K, [0, 2, 1])
        
        V = self.F(tf.transpose(V, [0, 2, 1]))
        V = tf.transpose(V, [0, 2, 1])
        
        # Multi-head分割
        Q = tf.reshape(Q, (batch_size, seq_len, self.num_heads, self.depth))
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        
        K = tf.reshape(K, (batch_size, self.k, self.num_heads, self.depth))
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        
        V = tf.reshape(V, (batch_size, self.k, self.num_heads, self.depth))
        V = tf.transpose(V, perm=[0, 2, 1, 3])
        
        # Attention計算
        attention_scores = tf.matmul(Q, K, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(
            tf.cast(self.depth, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        attention_output = tf.matmul(attention_weights, V)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, 
                                     (batch_size, seq_len, self.d_model))
        
        output = self.dense(attention_output)
        return output
```

## まとめ

Transformerアーキテクチャの重要な概念：

1. **Self-Attention**：系列内の任意の位置間の関係を学習
2. **Multi-Head Attention**：複数の表現部分空間で並列にAttentionを計算
3. **Positional Encoding**：位置情報の埋め込み
4. **Layer Normalization**：層正規化による安定した学習
5. **Residual Connection**：勾配消失の防止

Transformerは現在のNLPの基盤技術であり、BERT、GPT、T5などの強力なモデルの基礎となっています。また、Vision TransformerのようにNLP以外の分野にも応用されています。

## 次へ

[AI開発の実践](../../実践/README.md)へ進む