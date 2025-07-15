# 画像生成

## 概要

画像生成は、AIが新しい画像を作り出す技術です。GAN（敵対的生成ネットワーク）から始まり、VAE（変分オートエンコーダ）、そして最新の拡散モデル（Diffusion Models）まで、様々な手法が開発されています。

## GAN（Generative Adversarial Networks）

### 基本的なGANの実装

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

class SimpleGAN:
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
        
    def build_generator(self):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.latent_dim),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(784, activation='tanh')
        ])
        return model
    
    def build_discriminator(self):
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(784,)),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def build_gan(self):
        self.discriminator.trainable = False
        
        gan_input = keras.Input(shape=(self.latent_dim,))
        fake_image = self.generator(gan_input)
        gan_output = self.discriminator(fake_image)
        
        gan = keras.Model(gan_input, gan_output)
        gan.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002),
            loss='binary_crossentropy'
        )
        return gan
    
    def train(self, X_train, epochs=10000, batch_size=128):
        half_batch = batch_size // 2
        
        for epoch in range(epochs):
            # Discriminatorの学習
            # 本物の画像
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            real_images = X_train[idx]
            
            # 偽物の画像
            noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
            fake_images = self.generator.predict(noise)
            
            # ラベル
            real_labels = np.ones((half_batch, 1))
            fake_labels = np.zeros((half_batch, 1))
            
            # 学習
            d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Generatorの学習
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            valid_labels = np.ones((batch_size, 1))
            
            g_loss = self.gan.train_on_batch(noise, valid_labels)
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch} - D Loss: {d_loss[0]:.4f}, "
                      f"D Acc: {d_loss[1]:.4f}, G Loss: {g_loss:.4f}")
                self.sample_images(epoch)
    
    def sample_images(self, epoch, samples=16):
        noise = np.random.normal(0, 1, (samples, self.latent_dim))
        generated_images = self.generator.predict(noise)
        generated_images = generated_images.reshape(-1, 28, 28)
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated_images[i], cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'gan_samples_epoch_{epoch}.png')
        plt.close()
```

### DCGAN（Deep Convolutional GAN）

```python
class DCGAN:
    def __init__(self, img_shape=(64, 64, 3), latent_dim=100):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
    
    def build_generator(self):
        model = keras.Sequential()
        
        # 開始: 4x4x1024
        model.add(layers.Dense(4 * 4 * 1024, input_dim=self.latent_dim))
        model.add(layers.Reshape((4, 4, 1024)))
        
        # 8x8x512
        model.add(layers.Conv2DTranspose(512, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        
        # 16x16x256
        model.add(layers.Conv2DTranspose(256, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        
        # 32x32x128
        model.add(layers.Conv2DTranspose(128, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        
        # 64x64x3
        model.add(layers.Conv2DTranspose(3, 4, strides=2, padding='same'))
        model.add(layers.Activation('tanh'))
        
        return model
    
    def build_discriminator(self):
        model = keras.Sequential()
        
        # 64x64x3 -> 32x32x64
        model.add(layers.Conv2D(64, 4, strides=2, padding='same', 
                               input_shape=self.img_shape))
        model.add(layers.LeakyReLU(0.2))
        
        # 32x32x64 -> 16x16x128
        model.add(layers.Conv2D(128, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        
        # 16x16x128 -> 8x8x256
        model.add(layers.Conv2D(256, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        
        # 8x8x256 -> 4x4x512
        model.add(layers.Conv2D(512, 4, strides=2, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        
        # 分類
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_gan(self):
        self.discriminator.trainable = False
        
        gan_input = keras.Input(shape=(self.latent_dim,))
        fake_image = self.generator(gan_input)
        gan_output = self.discriminator(fake_image)
        
        gan = keras.Model(gan_input, gan_output)
        gan.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
        
        return gan
```

### StyleGAN

```python
class StyleGAN:
    def __init__(self, img_size=256, latent_dim=512, style_dim=512):
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        
    def mapping_network(self):
        """潜在コードをスタイルコードに変換"""
        z_input = layers.Input(shape=(self.latent_dim,))
        
        w = layers.Dense(self.style_dim)(z_input)
        w = layers.LeakyReLU(0.2)(w)
        
        for _ in range(7):
            w = layers.Dense(self.style_dim)(w)
            w = layers.LeakyReLU(0.2)(w)
        
        return keras.Model(z_input, w, name='mapping_network')
    
    def synthesis_block(self, x, w, filters):
        """StyleGANの合成ブロック"""
        # Adaptive Instance Normalization
        x = AdaIN()([x, w])
        
        # 畳み込み
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # ノイズ追加
        noise = layers.Input(shape=(x.shape[1], x.shape[2], 1))
        noise_weight = self.add_weight(shape=(filters,))
        x = x + noise * noise_weight
        
        return x

class AdaIN(layers.Layer):
    """Adaptive Instance Normalization"""
    def __init__(self):
        super(AdaIN, self).__init__()
    
    def build(self, input_shape):
        self.epsilon = 1e-5
    
    def call(self, inputs):
        x, w = inputs
        
        # インスタンス正規化
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2], keepdims=True)
        normalized = (x - mean) / (std + self.epsilon)
        
        # スタイル変調
        style_shape = [1, 1, 1, x.shape[-1]]
        gamma = layers.Dense(x.shape[-1])(w)
        beta = layers.Dense(x.shape[-1])(w)
        
        gamma = tf.reshape(gamma, style_shape)
        beta = tf.reshape(beta, style_shape)
        
        return normalized * gamma + beta
```

## VAE（Variational Autoencoder）

### 基本的なVAE

```python
class VAE(keras.Model):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
    
    def build_encoder(self):
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        
        x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
        x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # サンプリング層
        z = Sampling()([z_mean, z_log_var])
        
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder
    
    def build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        
        x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
        x = layers.Reshape((7, 7, 64))(x)
        
        x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
        
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
        
        decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
        return decoder
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        # KLダイバージェンスの計算
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)
        
        return reconstructed

class Sampling(layers.Layer):
    """再パラメータ化トリック"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# VAEの学習
vae = VAE(latent_dim=32)
vae.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy()
)

# 潜在空間の可視化
def plot_latent_space(vae, n=30, digit_size=28):
    scale = 2.0
    figure = np.zeros((digit_size * n, digit_size * n))
    
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit
    
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.axis('off')
    plt.show()
```

### β-VAE（Disentangled Representations）

```python
class BetaVAE(VAE):
    def __init__(self, latent_dim=32, beta=4.0):
        super(BetaVAE, self).__init__(latent_dim)
        self.beta = beta
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        
        # β-VAEのKL損失
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(self.beta * kl_loss)
        
        return reconstructed
    
    def interpolate(self, x1, x2, n_steps=10):
        """2つの画像間の補間"""
        # エンコード
        z1 = self.encoder(x1)[2]
        z2 = self.encoder(x2)[2]
        
        # 線形補間
        interpolated = []
        for alpha in np.linspace(0, 1, n_steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            x_interp = self.decoder(z_interp)
            interpolated.append(x_interp)
        
        return interpolated
```

## 拡散モデル（Diffusion Models）

### DDPM（Denoising Diffusion Probabilistic Models）

```python
class DiffusionModel:
    def __init__(self, img_size=32, channels=3, timesteps=1000):
        self.img_size = img_size
        self.channels = channels
        self.timesteps = timesteps
        
        # ノイズスケジュール
        self.betas = self.linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        
        # U-Netモデル
        self.model = self.build_unet()
    
    def linear_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return np.linspace(beta_start, beta_end, timesteps)
    
    def build_unet(self):
        """時間埋め込み付きU-Net"""
        # 画像入力
        img_input = layers.Input(shape=(self.img_size, self.img_size, self.channels))
        # 時間ステップ入力
        time_input = layers.Input(shape=(1,))
        
        # 時間埋め込み
        time_emb = layers.Dense(128)(time_input)
        time_emb = layers.Activation('swish')(time_emb)
        time_emb = layers.Dense(128)(time_emb)
        
        # エンコーダ
        x1 = layers.Conv2D(64, 3, padding='same')(img_input)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('swish')(x1)
        
        # 時間埋め込みの追加
        time_emb_1 = layers.Dense(64)(time_emb)
        time_emb_1 = layers.Reshape((1, 1, 64))(time_emb_1)
        x1 = x1 + time_emb_1
        
        x2 = layers.Conv2D(128, 3, strides=2, padding='same')(x1)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('swish')(x2)
        
        x3 = layers.Conv2D(256, 3, strides=2, padding='same')(x2)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Activation('swish')(x3)
        
        # ボトルネック
        x = layers.Conv2D(512, 3, padding='same')(x3)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        
        # デコーダ
        x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, x3])
        x = layers.Conv2D(256, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        
        x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, x2])
        x = layers.Conv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        
        # 出力
        output = layers.Conv2D(self.channels, 3, padding='same')(x)
        
        return keras.Model([img_input, time_input], output)
    
    def forward_diffusion(self, x0, t):
        """前方拡散プロセス"""
        noise = tf.random.normal(shape=tf.shape(x0))
        
        sqrt_alphas_cumprod_t = tf.gather(tf.sqrt(self.alphas_cumprod), t)
        sqrt_one_minus_alphas_cumprod_t = tf.gather(
            tf.sqrt(1. - self.alphas_cumprod), t
        )
        
        # x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        return (sqrt_alphas_cumprod_t[:, None, None, None] * x0 + 
                sqrt_one_minus_alphas_cumprod_t[:, None, None, None] * noise), noise
    
    def train_step(self, x0, optimizer):
        """学習ステップ"""
        batch_size = tf.shape(x0)[0]
        
        # ランダムな時間ステップ
        t = tf.random.uniform((batch_size,), 0, self.timesteps, dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            # 前方拡散
            x_t, noise = self.forward_diffusion(x0, t)
            
            # ノイズ予測
            noise_pred = self.model([x_t, tf.cast(t, tf.float32)])
            
            # 損失計算
            loss = tf.reduce_mean(tf.square(noise - noise_pred))
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
    
    @tf.function
    def sample(self, batch_size=16):
        """画像生成（逆拡散プロセス）"""
        # ランダムノイズから開始
        x = tf.random.normal((batch_size, self.img_size, self.img_size, self.channels))
        
        for t in reversed(range(self.timesteps)):
            t_batch = tf.fill([batch_size], t)
            
            # ノイズ予測
            predicted_noise = self.model([x, tf.cast(t_batch, tf.float32)])
            
            # 逆拡散ステップ
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = tf.random.normal(shape=tf.shape(x))
            else:
                noise = tf.zeros_like(x)
            
            x = (1 / tf.sqrt(alpha)) * (
                x - ((1 - alpha) / tf.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + tf.sqrt(beta) * noise
        
        return x
```

### Stable Diffusion風の実装

```python
class LatentDiffusion:
    def __init__(self, img_size=512, latent_size=64):
        self.img_size = img_size
        self.latent_size = latent_size
        
        # VAEエンコーダ/デコーダ
        self.vae_encoder = self.build_vae_encoder()
        self.vae_decoder = self.build_vae_decoder()
        
        # U-Net（潜在空間で動作）
        self.unet = self.build_latent_unet()
        
        # テキストエンコーダ（CLIP風）
        self.text_encoder = self.build_text_encoder()
    
    def build_vae_encoder(self):
        """画像を潜在表現に変換"""
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        x = layers.Conv2D(128, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        
        # さらにダウンサンプリング
        for filters in [256, 512, 512]:
            x = layers.Conv2D(filters, 3, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('swish')(x)
        
        # 潜在表現
        latent = layers.Conv2D(8, 1)(x)
        
        return keras.Model(inputs, latent)
    
    def build_vae_decoder(self):
        """潜在表現から画像を再構成"""
        latent_inputs = layers.Input(shape=(self.latent_size, self.latent_size, 8))
        
        x = layers.Conv2D(512, 1)(latent_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        
        # アップサンプリング
        for filters in [512, 256, 128]:
            x = layers.Conv2DTranspose(filters, 3, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('swish')(x)
        
        # 最終出力
        outputs = layers.Conv2DTranspose(3, 3, strides=2, padding='same', 
                                        activation='tanh')(x)
        
        return keras.Model(latent_inputs, outputs)
    
    def build_text_encoder(self):
        """テキストを条件ベクトルに変換"""
        text_input = layers.Input(shape=(77,), dtype=tf.int32)  # CLIP風のトークン長
        
        # 埋め込み
        x = layers.Embedding(49408, 768)(text_input)  # CLIP風の語彙数
        
        # Transformer層
        for _ in range(12):
            attn = layers.MultiHeadAttention(num_heads=12, key_dim=64)(x, x)
            x = layers.LayerNormalization()(x + attn)
            
            ffn = layers.Dense(3072, activation='gelu')(x)
            ffn = layers.Dense(768)(ffn)
            x = layers.LayerNormalization()(x + ffn)
        
        # 最終出力
        text_features = layers.GlobalAveragePooling1D()(x)
        
        return keras.Model(text_input, text_features)
    
    def text_to_image(self, text_prompt, num_inference_steps=50):
        """テキストから画像を生成"""
        # テキストエンコード
        text_features = self.text_encoder(text_prompt)
        
        # 潜在空間でのノイズ
        latent = tf.random.normal((1, self.latent_size, self.latent_size, 8))
        
        # 逆拡散プロセス（簡略化）
        for t in range(num_inference_steps):
            # U-Netでノイズ予測（テキスト条件付き）
            noise_pred = self.unet([latent, text_features, t])
            
            # ノイズ除去
            latent = latent - noise_pred * 0.1
        
        # 画像にデコード
        image = self.vae_decoder(latent)
        
        return image
```

## スタイル変換

### Neural Style Transfer

```python
def style_transfer_loss(style_features, content_features, generated_features):
    """スタイル変換の損失関数"""
    style_weight = 1e-2
    content_weight = 1e4
    
    # コンテンツ損失
    content_loss = tf.reduce_mean(
        tf.square(content_features['block5_conv2'] - 
                 generated_features['block5_conv2'])
    )
    
    # スタイル損失
    style_loss = 0
    for layer in style_features.keys():
        style_gram = gram_matrix(style_features[layer])
        generated_gram = gram_matrix(generated_features[layer])
        
        layer_loss = tf.reduce_mean(tf.square(style_gram - generated_gram))
        style_loss += layer_loss
    
    style_loss /= len(style_features)
    
    # 総損失
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    return total_loss, content_loss, style_loss

def gram_matrix(input_tensor):
    """グラム行列の計算"""
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

class StyleTransfer:
    def __init__(self):
        # VGG19の特徴抽出器
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        
        # 中間層の出力を取得
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                       'block4_conv1', 'block5_conv1']
        content_layers = ['block5_conv2']
        
        output_layers = style_layers + content_layers
        outputs = [vgg.get_layer(name).output for name in output_layers]
        
        self.feature_extractor = keras.Model([vgg.input], outputs)
    
    def transfer(self, content_image, style_image, iterations=1000):
        # 生成画像の初期化（コンテンツ画像のコピー）
        generated_image = tf.Variable(content_image, dtype=tf.float32)
        
        # オプティマイザ
        optimizer = tf.optimizers.Adam(learning_rate=0.02)
        
        # 特徴抽出
        style_features = self.extract_features(style_image)
        content_features = self.extract_features(content_image)
        
        for i in range(iterations):
            with tf.GradientTape() as tape:
                generated_features = self.extract_features(generated_image)
                
                loss, content_loss, style_loss = style_transfer_loss(
                    style_features, content_features, generated_features
                )
            
            gradients = tape.gradient(loss, generated_image)
            optimizer.apply_gradients([(gradients, generated_image)])
            
            # クリッピング
            generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))
            
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.2f}")
        
        return generated_image
```

## 実践的な応用

### 画像補完（Inpainting）

```python
class ImageInpainting:
    def __init__(self, model_type='conv_autoencoder'):
        self.model = self.build_model(model_type)
    
    def build_model(self, model_type):
        if model_type == 'conv_autoencoder':
            return self.build_conv_autoencoder()
        elif model_type == 'partial_conv':
            return self.build_partial_conv_model()
    
    def build_conv_autoencoder(self):
        # エンコーダ
        inputs = layers.Input(shape=(256, 256, 3))
        mask = layers.Input(shape=(256, 256, 1))
        
        # マスクされた画像
        masked_image = layers.Multiply()([inputs, mask])
        
        x = layers.Conv2D(64, 3, strides=2, padding='same')(masked_image)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # さらにエンコード
        for filters in [128, 256, 512]:
            x = layers.Conv2D(filters, 3, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        
        # デコーダ
        for filters in [256, 128, 64]:
            x = layers.Conv2DTranspose(filters, 3, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        
        outputs = layers.Conv2DTranspose(3, 3, strides=2, padding='same', 
                                        activation='sigmoid')(x)
        
        return keras.Model([inputs, mask], outputs)
    
    def create_random_mask(self, shape, mask_ratio=0.5):
        """ランダムなマスクを生成"""
        mask = np.ones(shape)
        h, w = shape[:2]
        
        # ランダムな矩形マスク
        num_masks = int(10 * mask_ratio)
        for _ in range(num_masks):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            w_mask = np.random.randint(10, 50)
            h_mask = np.random.randint(10, 50)
            
            mask[y:y+h_mask, x:x+w_mask] = 0
        
        return mask

# Partial Convolution
class PartialConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(PartialConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kwargs = kwargs
    
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            self.filters, self.kernel_size, **self.kwargs
        )
        
        # マスク用の畳み込み
        self.mask_conv = layers.Conv2D(
            1, self.kernel_size,
            use_bias=False,
            kernel_initializer='ones',
            trainable=False,
            **self.kwargs
        )
    
    def call(self, inputs):
        image, mask = inputs
        
        # マスクされた画像の畳み込み
        masked_image = image * mask
        output = self.conv(masked_image)
        
        # マスクの更新
        updated_mask = self.mask_conv(mask)
        mask_ratio = self.kernel_size ** 2 / (updated_mask + 1e-8)
        
        # 正規化
        output = output * mask_ratio
        
        # 新しいマスク（0より大きい部分を1に）
        updated_mask = tf.cast(updated_mask > 0, tf.float32)
        
        return output, updated_mask
```

### 超解像（Super Resolution）

```python
class ESRGAN:
    """Enhanced Super-Resolution GAN"""
    def __init__(self, scale_factor=4):
        self.scale_factor = scale_factor
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
    
    def residual_dense_block(self, x, filters=64, growth_channels=32):
        """Residual Dense Block"""
        convs = []
        for i in range(5):
            conv = layers.Conv2D(growth_channels, 3, padding='same')(x)
            conv = layers.LeakyReLU(0.2)(conv)
            x = layers.Concatenate()([x, conv])
            convs.append(conv)
        
        # Local Feature Fusion
        x_lff = layers.Conv2D(filters, 1)(x)
        
        return x_lff
    
    def build_generator(self):
        inputs = layers.Input(shape=(None, None, 3))
        
        # 最初の畳み込み
        x = layers.Conv2D(64, 3, padding='same')(inputs)
        x_skip = x
        
        # Residual Dense Blocks
        for _ in range(16):
            x = self.residual_dense_block(x)
        
        # Global Residual Learning
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.Add()([x, x_skip])
        
        # アップサンプリング
        for _ in range(self.scale_factor // 2):
            x = layers.Conv2D(256, 3, padding='same')(x)
            x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
            x = layers.LeakyReLU(0.2)(x)
        
        outputs = layers.Conv2D(3, 3, padding='same')(x)
        
        return keras.Model(inputs, outputs)
    
    def build_discriminator(self):
        inputs = layers.Input(shape=(None, None, 3))
        
        x = layers.Conv2D(64, 3, padding='same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        # ダウンサンプリング
        for filters in [128, 256, 512]:
            x = layers.Conv2D(filters, 3, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(0.2)(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(1)(x)
        
        return keras.Model(inputs, outputs)
```

## まとめ

画像生成の主要技術：

1. **GAN**：敵対的学習による高品質な画像生成
2. **VAE**：確率的な潜在表現による生成
3. **拡散モデル**：ノイズ除去による高品質生成
4. **スタイル変換**：画像の芸術的変換
5. **応用技術**：補完、超解像、編集

各手法には長所と短所があり、用途に応じて選択することが重要です。

## 次へ

[音声認識](../音声認識/README.md)へ進む