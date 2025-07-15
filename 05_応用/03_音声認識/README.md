# 音声認識

## 概要

音声認識は、人間の音声をテキストに変換する技術です。音声信号処理の基礎から、最新のEnd-to-Endモデルまで、幅広い技術を扱います。音声合成（TTS）や音声変換も含めて解説します。

## 音声信号処理の基礎

### 音声データの前処理

```python
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path):
        """音声ファイルの読み込み"""
        # librosaを使用（自動的にモノラル、指定サンプルレートに変換）
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr
    
    def normalize_audio(self, audio):
        """音声の正規化"""
        # 最大振幅で正規化
        return audio / np.max(np.abs(audio))
    
    def trim_silence(self, audio, top_db=20):
        """無音部分の除去"""
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    
    def add_noise(self, audio, noise_level=0.005):
        """ノイズ追加（データ拡張）"""
        noise = np.random.randn(len(audio))
        return audio + noise_level * noise
    
    def time_stretch(self, audio, rate=1.0):
        """時間伸縮（データ拡張）"""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, n_steps=0):
        """ピッチシフト（データ拡張）"""
        return librosa.effects.pitch_shift(
            audio, sr=self.sample_rate, n_steps=n_steps
        )

# 音声の可視化
def visualize_audio(audio, sr):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 波形
    axes[0].plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    
    # スペクトログラム
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='time', 
                                   sr=sr, ax=axes[1])
    axes[1].set_title('Spectrogram')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    # メルスペクトログラム
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time',
                                   sr=sr, ax=axes[2])
    axes[2].set_title('Mel-Spectrogram')
    fig.colorbar(img, ax=axes[2], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()
```

### 特徴量抽出

```python
class FeatureExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def extract_mfcc(self, audio, n_mfcc=13):
        """MFCC（メル周波数ケプストラム係数）の抽出"""
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=n_mfcc
        )
        
        # デルタとデルタデルタの計算
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 結合
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        return features.T
    
    def extract_mel_spectrogram(self, audio, n_mels=128, n_fft=2048, 
                               hop_length=512):
        """メルスペクトログラムの抽出"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # デシベルスケールに変換
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_spectral_features(self, audio):
        """スペクトル特徴量の抽出"""
        features = {}
        
        # スペクトル重心
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate
        )
        
        # スペクトルロールオフ
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate
        )
        
        # ゼロ交差率
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)
        
        # スペクトル帯域幅
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate
        )
        
        return features
    
    def extract_chroma_features(self, audio):
        """クロマ特徴量の抽出"""
        chroma = librosa.feature.chroma_stft(
            y=audio, 
            sr=self.sample_rate
        )
        return chroma
```

## 従来の音声認識手法

### HMM-GMM

```python
import hmmlearn.hmm
from sklearn.mixture import GaussianMixture

class HMMGMMRecognizer:
    def __init__(self, n_components=5, n_mix=8):
        self.n_components = n_components
        self.n_mix = n_mix
        self.models = {}
    
    def train_phone_model(self, features_dict, phone_label):
        """音素モデルの学習"""
        # 特徴量の結合
        all_features = []
        lengths = []
        
        for features in features_dict[phone_label]:
            all_features.append(features)
            lengths.append(len(features))
        
        X = np.vstack(all_features)
        
        # GMMで各状態の出力確率をモデル化
        model = hmmlearn.hmm.GMMHMM(
            n_components=self.n_components,
            n_mix=self.n_mix,
            covariance_type='diag'
        )
        
        model.fit(X, lengths)
        self.models[phone_label] = model
        
        return model
    
    def recognize(self, features):
        """音声認識"""
        scores = {}
        
        for phone, model in self.models.items():
            score = model.score(features)
            scores[phone] = score
        
        # 最も尤度の高い音素を返す
        return max(scores, key=scores.get)
```

## ディープラーニングベースの音声認識

### CNN-RNN-CTC

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class DeepSpeech2:
    def __init__(self, input_dim, output_dim, rnn_units=800):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_units = rnn_units
        self.model = self.build_model()
    
    def build_model(self):
        # 入力層
        inputs = layers.Input(shape=(None, self.input_dim), name='audio_input')
        
        # CNN層
        x = layers.Reshape((-1, self.input_dim, 1))(inputs)
        
        # 1D Convolution
        for filters in [32, 32]:
            x = layers.Conv2D(filters, (41, 11), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D((2, 2))(x)
        
        # RNNのための形状変換
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        
        # Bidirectional RNN層
        for _ in range(5):
            x = layers.Bidirectional(
                layers.GRU(self.rnn_units, return_sequences=True, 
                          dropout=0.1)
            )(x)
        
        # 出力層
        outputs = layers.Dense(self.output_dim + 1, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile_with_ctc_loss(self):
        """CTC損失での コンパイル"""
        def ctc_loss(y_true, y_pred):
            # バッチ内の各サンプルの長さを取得
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
            
            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            
            loss = tf.nn.ctc_loss(
                labels=y_true,
                logits=y_pred,
                label_length=label_length,
                logit_length=input_length,
                blank_index=self.output_dim
            )
            
            return loss
        
        self.model.compile(
            optimizer='adam',
            loss=ctc_loss
        )

# CTCデコーダ
def ctc_decode(y_pred, input_length, greedy=True):
    """CTC出力のデコード"""
    if greedy:
        decoded = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, [1, 0, 2]),
            sequence_length=input_length,
            merge_repeated=True
        )
    else:
        decoded = tf.nn.ctc_beam_search_decoder(
            inputs=tf.transpose(y_pred, [1, 0, 2]),
            sequence_length=input_length,
            beam_width=100,
            merge_repeated=True
        )
    
    return decoded
```

### Transformer音声認識

```python
class ConformerBlock(layers.Layer):
    """Conformer: CNN + Transformer の組み合わせ"""
    def __init__(self, d_model, n_heads, kernel_size=32):
        super(ConformerBlock, self).__init__()
        self.d_model = d_model
        
        # Feed Forward Module
        self.ffn1 = self._build_ffn()
        
        # Multi-Head Self Attention
        self.mhsa = layers.MultiHeadAttention(n_heads, d_model)
        self.norm_mhsa = layers.LayerNormalization()
        
        # Convolution Module
        self.conv = self._build_conv_module(kernel_size)
        self.norm_conv = layers.LayerNormalization()
        
        # Feed Forward Module
        self.ffn2 = self._build_ffn()
        
        # Layer Normalization
        self.norm_out = layers.LayerNormalization()
    
    def _build_ffn(self):
        return tf.keras.Sequential([
            layers.Dense(self.d_model * 4, activation='swish'),
            layers.Dropout(0.1),
            layers.Dense(self.d_model),
            layers.Dropout(0.1)
        ])
    
    def _build_conv_module(self, kernel_size):
        return tf.keras.Sequential([
            layers.Conv1D(self.d_model * 2, 1),
            layers.Lambda(lambda x: x[:, :, :self.d_model] * 
                         tf.nn.sigmoid(x[:, :, self.d_model:])),  # GLU
            layers.Conv1D(self.d_model, kernel_size, padding='same', 
                         groups=self.d_model),  # Depthwise
            layers.BatchNormalization(),
            layers.Activation('swish'),
            layers.Conv1D(self.d_model, 1),
            layers.Dropout(0.1)
        ])
    
    def call(self, x, training=False):
        # First Feed Forward
        x = x + 0.5 * self.ffn1(x)
        
        # Multi-Head Self Attention
        attn_out = self.mhsa(x, x, training=training)
        x = self.norm_mhsa(x + attn_out)
        
        # Convolution Module
        conv_out = self.conv(x)
        x = self.norm_conv(x + conv_out)
        
        # Second Feed Forward
        x = x + 0.5 * self.ffn2(x)
        
        # Output normalization
        x = self.norm_out(x)
        
        return x

class Conformer(Model):
    def __init__(self, input_dim, output_dim, d_model=512, n_heads=8, 
                 n_layers=12):
        super(Conformer, self).__init__()
        
        # 入力層
        self.input_proj = layers.Dense(d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Conformer Blocks
        self.conformer_blocks = [
            ConformerBlock(d_model, n_heads) for _ in range(n_layers)
        ]
        
        # 出力層
        self.output_proj = layers.Dense(output_dim)
    
    def call(self, inputs, training=False):
        # 入力投影
        x = self.input_proj(inputs)
        
        # Positional Encoding
        x = self.pos_encoding(x)
        
        # Conformer Blocks
        for block in self.conformer_blocks:
            x = block(x, training=training)
        
        # 出力投影
        outputs = self.output_proj(x)
        
        return outputs

class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * 
                          (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]
```

### Whisper風モデル

```python
class WhisperModel(Model):
    """OpenAI Whisper風の音声認識モデル"""
    def __init__(self, d_model=512, n_heads=8, n_encoder_layers=6,
                 n_decoder_layers=6, vocab_size=51865):
        super(WhisperModel, self).__init__()
        
        # Audio Encoder
        self.audio_encoder = self.build_audio_encoder(d_model)
        
        # Text Decoder (Transformer)
        self.text_decoder = self.build_text_decoder(
            d_model, n_heads, n_decoder_layers, vocab_size
        )
    
    def build_audio_encoder(self, d_model):
        """音声エンコーダ（CNN + Transformer）"""
        return tf.keras.Sequential([
            # CNN特徴抽出
            layers.Conv1D(d_model, 3, padding='same'),
            layers.ReLU(),
            layers.Conv1D(d_model, 3, strides=2, padding='same'),
            layers.ReLU(),
            
            # Positional Encoding
            PositionalEncoding(d_model),
            
            # Transformer Encoder
            *[TransformerEncoderLayer(d_model, 8) for _ in range(6)]
        ])
    
    def build_text_decoder(self, d_model, n_heads, n_layers, vocab_size):
        """テキストデコーダ"""
        decoder_layers = []
        
        # 埋め込み層
        decoder_layers.append(layers.Embedding(vocab_size, d_model))
        decoder_layers.append(PositionalEncoding(d_model))
        
        # Transformer Decoder層
        for _ in range(n_layers):
            decoder_layers.append(
                TransformerDecoderLayer(d_model, n_heads)
            )
        
        # 出力層
        decoder_layers.append(layers.Dense(vocab_size))
        
        return tf.keras.Sequential(decoder_layers)
    
    def call(self, audio_input, text_input=None, training=False):
        # 音声エンコード
        audio_features = self.audio_encoder(audio_input, training=training)
        
        if text_input is not None:
            # テキストデコード（訓練時）
            text_output = self.text_decoder(
                text_input, 
                encoder_output=audio_features,
                training=training
            )
            return text_output
        else:
            # 推論時（自己回帰的生成）
            return self.generate(audio_features)
    
    def generate(self, audio_features, max_length=448):
        """テキスト生成"""
        batch_size = tf.shape(audio_features)[0]
        
        # 開始トークン
        decoder_input = tf.ones((batch_size, 1), dtype=tf.int32) * START_TOKEN
        
        for _ in range(max_length):
            # デコード
            predictions = self.text_decoder(
                decoder_input, 
                encoder_output=audio_features,
                training=False
            )
            
            # 最後のトークンの予測
            predicted_id = tf.argmax(predictions[:, -1, :], axis=-1)
            
            # 終了トークンのチェック
            if tf.reduce_all(predicted_id == END_TOKEN):
                break
            
            # 次の入力に追加
            decoder_input = tf.concat([
                decoder_input, 
                predicted_id[:, tf.newaxis]
            ], axis=-1)
        
        return decoder_input
```

## 音声合成（Text-to-Speech）

### Tacotron 2

```python
class Tacotron2:
    def __init__(self, vocab_size, embedding_dim=512):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.postnet = self.build_postnet()
    
    def build_encoder(self):
        """テキストエンコーダ"""
        inputs = layers.Input(shape=(None,), dtype='int32')
        
        # 文字埋め込み
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        
        # 3層の畳み込み
        for filters in [512, 512, 512]:
            x = layers.Conv1D(filters, 5, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.5)(x)
        
        # Bidirectional LSTM
        x = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True)
        )(x)
        
        return Model(inputs, x)
    
    def build_decoder(self):
        """メルスペクトログラムデコーダ"""
        # 入力
        encoder_output = layers.Input(shape=(None, 512))
        mel_input = layers.Input(shape=(None, 80))  # 80次元メルスペクトログラム
        
        # Pre-Net
        prenet = layers.Dense(256, activation='relu')(mel_input)
        prenet = layers.Dropout(0.5)(prenet)
        prenet = layers.Dense(256, activation='relu')(prenet)
        prenet = layers.Dropout(0.5)(prenet)
        
        # Attention RNN
        attention_rnn = layers.LSTM(1024, return_sequences=True)(prenet)
        
        # Attention
        attention = BahdanauAttention(512)
        context, attention_weights = attention(
            attention_rnn, encoder_output
        )
        
        # Decoder RNN
        decoder_input = layers.Concatenate()([attention_rnn, context])
        decoder_rnn = layers.LSTM(1024, return_sequences=True)(decoder_input)
        
        # 出力投影
        mel_output = layers.Dense(80)(decoder_rnn)
        stop_output = layers.Dense(1, activation='sigmoid')(decoder_rnn)
        
        return Model(
            [encoder_output, mel_input], 
            [mel_output, stop_output, attention_weights]
        )
    
    def build_postnet(self):
        """Post-Net: メルスペクトログラムの改善"""
        mel_input = layers.Input(shape=(None, 80))
        
        x = mel_input
        for i in range(5):
            filters = 512 if i < 4 else 80
            x = layers.Conv1D(filters, 5, padding='same')(x)
            if i < 4:
                x = layers.BatchNormalization()(x)
                x = layers.Activation('tanh')(x)
                x = layers.Dropout(0.5)(x)
        
        # 残差接続
        refined_mel = layers.Add()([mel_input, x])
        
        return Model(mel_input, refined_mel)

class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, query, values):
        # query: (batch_size, time_steps, hidden_dim)
        # values: (batch_size, time_steps, hidden_dim)
        
        # スコア計算
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        ))
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
```

### WaveNet

```python
class WaveNet:
    def __init__(self, n_filters=32, n_layers=30, n_stacks=3):
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_stacks = n_stacks
        self.model = self.build_model()
    
    def causal_conv1d(self, x, filters, kernel_size, dilation_rate):
        """因果的畳み込み"""
        # パディングを追加して因果性を保証
        pad_size = (kernel_size - 1) * dilation_rate
        x = tf.pad(x, [[0, 0], [pad_size, 0], [0, 0]])
        
        return layers.Conv1D(
            filters, 
            kernel_size, 
            dilation_rate=dilation_rate,
            padding='valid'
        )(x)
    
    def wavenet_block(self, x, dilation_rate, filters):
        """WaveNetの基本ブロック"""
        # Gated activation unit
        tanh_out = self.causal_conv1d(
            x, filters, 2, dilation_rate
        )
        tanh_out = layers.Activation('tanh')(tanh_out)
        
        sigmoid_out = self.causal_conv1d(
            x, filters, 2, dilation_rate
        )
        sigmoid_out = layers.Activation('sigmoid')(sigmoid_out)
        
        # Element-wise multiplication
        gated = layers.Multiply()([tanh_out, sigmoid_out])
        
        # 1x1 convolution
        skip = layers.Conv1D(filters, 1)(gated)
        residual = layers.Conv1D(filters, 1)(gated)
        
        # Residual connection
        x = layers.Add()([x, residual])
        
        return x, skip
    
    def build_model(self):
        inputs = layers.Input(shape=(None, 1))
        
        # 初期畳み込み
        x = self.causal_conv1d(inputs, self.n_filters, 2, 1)
        
        skip_connections = []
        
        # WaveNetスタック
        for stack in range(self.n_stacks):
            for layer in range(self.n_layers):
                dilation_rate = 2 ** (layer % 10)
                x, skip = self.wavenet_block(
                    x, dilation_rate, self.n_filters
                )
                skip_connections.append(skip)
        
        # Skip connectionsの合計
        out = layers.Add()(skip_connections)
        out = layers.Activation('relu')(out)
        
        # 出力層
        out = layers.Conv1D(256, 1, activation='relu')(out)
        out = layers.Conv1D(256, 1)(out)
        
        # 256値のμ-law量子化
        outputs = layers.Activation('softmax')(out)
        
        return Model(inputs, outputs)
    
    def mu_law_encode(self, audio, mu=255):
        """μ-law圧縮"""
        mu = float(mu)
        audio = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
        return ((audio + 1) / 2 * mu).astype(np.int32)
    
    def mu_law_decode(self, encoded, mu=255):
        """μ-law展開"""
        mu = float(mu)
        encoded = encoded.astype(np.float32)
        audio = 2 * (encoded / mu) - 1
        audio = np.sign(audio) * (1 / mu) * ((1 + mu) ** np.abs(audio) - 1)
        return audio
```

## 音声変換

### Voice Conversion

```python
class CycleGANVC:
    """CycleGANベースの音声変換"""
    def __init__(self, n_features=36):
        self.n_features = n_features
        self.generator_A2B = self.build_generator()
        self.generator_B2A = self.build_generator()
        self.discriminator_A = self.build_discriminator()
        self.discriminator_B = self.build_discriminator()
    
    def build_generator(self):
        """ジェネレータ（1D CNN）"""
        inputs = layers.Input(shape=(None, self.n_features))
        
        # ダウンサンプリング
        x = layers.Conv1D(128, 15, strides=1, padding='same')(inputs)
        x = layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv1D(256, 5, strides=2, padding='same')(x)
        x = layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv1D(512, 5, strides=2, padding='same')(x)
        x = layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        
        # Residual blocks
        for _ in range(6):
            residual = x
            x = layers.Conv1D(512, 5, padding='same')(x)
            x = layers.InstanceNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv1D(512, 5, padding='same')(x)
            x = layers.InstanceNormalization()(x)
            x = layers.Add()([x, residual])
        
        # アップサンプリング
        x = layers.Conv1DTranspose(256, 5, strides=2, padding='same')(x)
        x = layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv1DTranspose(128, 5, strides=2, padding='same')(x)
        x = layers.InstanceNormalization()(x)
        x = layers.ReLU()(x)
        
        outputs = layers.Conv1D(self.n_features, 15, padding='same')(x)
        
        return Model(inputs, outputs)
    
    def build_discriminator(self):
        """ディスクリミネータ（PatchGAN）"""
        inputs = layers.Input(shape=(None, self.n_features))
        
        x = layers.Conv1D(128, 3, strides=1, padding='same')(inputs)
        x = layers.LeakyReLU(0.01)(x)
        
        x = layers.Conv1D(256, 3, strides=2, padding='same')(x)
        x = layers.InstanceNormalization()(x)
        x = layers.LeakyReLU(0.01)(x)
        
        x = layers.Conv1D(512, 3, strides=2, padding='same')(x)
        x = layers.InstanceNormalization()(x)
        x = layers.LeakyReLU(0.01)(x)
        
        x = layers.Conv1D(1024, 3, strides=2, padding='same')(x)
        x = layers.InstanceNormalization()(x)
        x = layers.LeakyReLU(0.01)(x)
        
        outputs = layers.Conv1D(1, 3, padding='same')(x)
        
        return Model(inputs, outputs)

class InstanceNormalization(layers.Layer):
    """インスタンス正規化層"""
    def __init__(self):
        super(InstanceNormalization, self).__init__()
    
    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1], keepdims=True)
        return self.gamma * (x - mean) / tf.sqrt(var + 1e-5) + self.beta
```

## 実践的な応用

### リアルタイム音声認識

```python
import pyaudio
import threading
import queue

class RealtimeASR:
    def __init__(self, model, sample_rate=16000, chunk_duration=0.5):
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # PyAudio設定
        self.p = pyaudio.PyAudio()
        self.stream = None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """音声入力コールバック"""
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self):
        """録音開始"""
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
    
    def process_audio(self):
        """音声処理スレッド"""
        buffer = np.array([])
        
        while True:
            if not self.audio_queue.empty():
                # 音声データの取得
                audio_chunk = self.audio_queue.get()
                audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
                
                # バッファに追加
                buffer = np.concatenate([buffer, audio_array])
                
                # 一定長になったら処理
                if len(buffer) >= self.sample_rate * 2:  # 2秒分
                    # 特徴量抽出
                    features = self.extract_features(buffer)
                    
                    # 音声認識
                    text = self.recognize(features)
                    self.result_queue.put(text)
                    
                    # バッファをスライド
                    buffer = buffer[self.sample_rate:]
    
    def extract_features(self, audio):
        """特徴量抽出"""
        # メルスペクトログラム
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_mels=80
        )
        return librosa.power_to_db(mel_spec, ref=np.max)
    
    def recognize(self, features):
        """音声認識"""
        # モデルによる推論
        features = np.expand_dims(features, axis=0)
        predictions = self.model.predict(features)
        
        # デコード
        text = self.decode_predictions(predictions)
        return text

# 使用例
# asr = RealtimeASR(model)
# asr.start_recording()
# processing_thread = threading.Thread(target=asr.process_audio)
# processing_thread.start()
```

### 話者認識

```python
class SpeakerVerification:
    """話者認識システム"""
    def __init__(self, embedding_size=256):
        self.embedding_size = embedding_size
        self.model = self.build_model()
        self.speaker_embeddings = {}
    
    def build_model(self):
        """話者埋め込みモデル"""
        inputs = layers.Input(shape=(None, 40))  # 40次元MFCC
        
        # LSTM層
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.LSTM(128)(x)
        
        # 埋め込み層
        embeddings = layers.Dense(self.embedding_size)(x)
        embeddings = layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=-1)
        )(embeddings)
        
        return Model(inputs, embeddings)
    
    def extract_embedding(self, audio):
        """話者埋め込みの抽出"""
        # MFCC特徴量
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
        mfcc = mfcc.T[np.newaxis, :]
        
        # 埋め込み抽出
        embedding = self.model.predict(mfcc)[0]
        return embedding
    
    def enroll_speaker(self, speaker_id, audio_samples):
        """話者の登録"""
        embeddings = []
        
        for audio in audio_samples:
            embedding = self.extract_embedding(audio)
            embeddings.append(embedding)
        
        # 平均埋め込み
        mean_embedding = np.mean(embeddings, axis=0)
        self.speaker_embeddings[speaker_id] = mean_embedding
    
    def verify(self, audio, claimed_speaker_id, threshold=0.7):
        """話者照合"""
        if claimed_speaker_id not in self.speaker_embeddings:
            return False, 0.0
        
        # 入力音声の埋め込み
        test_embedding = self.extract_embedding(audio)
        
        # 登録済み埋め込みとの類似度
        enrolled_embedding = self.speaker_embeddings[claimed_speaker_id]
        similarity = np.dot(test_embedding, enrolled_embedding)
        
        return similarity > threshold, similarity
    
    def identify(self, audio, threshold=0.7):
        """話者識別"""
        test_embedding = self.extract_embedding(audio)
        
        best_speaker = None
        best_similarity = -1
        
        for speaker_id, enrolled_embedding in self.speaker_embeddings.items():
            similarity = np.dot(test_embedding, enrolled_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_id
        
        if best_similarity > threshold:
            return best_speaker, best_similarity
        else:
            return None, best_similarity
```

## まとめ

音声認識・合成の主要技術：

1. **音声処理基礎**：信号処理、特徴量抽出
2. **音声認識**：HMM-GMM、DNN-HMM、End-to-End
3. **音声合成**：Tacotron、WaveNet、FastSpeech
4. **音声変換**：Voice Conversion、話者適応
5. **応用技術**：リアルタイム処理、話者認識

音声技術は急速に進化しており、Transformerベースのモデルが主流になりつつあります。

## 次へ

[推薦システム](../推薦システム/README.md)へ進む