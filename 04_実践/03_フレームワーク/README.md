# ディープラーニングフレームワーク

## 概要

ディープラーニングフレームワークは、ニューラルネットワークの構築、学習、デプロイを効率的に行うためのツールです。ここでは、主要なフレームワークの詳細な使い方と、実践的なテクニックを学びます。

## フレームワークの選び方

### 主要フレームワークの比較

| フレームワーク | 特徴 | 適用場面 |
|--------------|------|---------|
| TensorFlow | 本番環境向け、エコシステム充実 | 大規模システム、モバイル展開 |
| PyTorch | 研究向け、動的グラフ | 研究開発、プロトタイピング |
| JAX | 関数型、高速 | 科学計算、最適化 |
| MXNet | スケーラブル | 分散学習 |

## TensorFlow詳細

### カスタムトレーニングループ

```python
import tensorflow as tf
import numpy as np
import time

# カスタムモデル
class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10)
    
    def call(self, x, training=False):
        x = self.dense1(x)
        if training:
            x = self.dropout(x)
        x = self.dense2(x)
        return self.dense3(x)

# データの準備
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# データセットの作成
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(60000).batch(128)

# モデルとオプティマイザ
model = CustomModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# メトリクス
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# トレーニングステップ
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)

# トレーニングループ
EPOCHS = 5

for epoch in range(EPOCHS):
    # メトリクスのリセット
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    start_time = time.time()
    
    for images, labels in train_dataset:
        train_step(images, labels)
    
    end_time = time.time()
    
    print(f'Epoch {epoch + 1}, '
          f'Loss: {train_loss.result():.4f}, '
          f'Accuracy: {train_accuracy.result():.4f}, '
          f'Time: {end_time - start_time:.2f}s')
```

### カスタムレイヤーの実装

```python
# カスタムレイヤー
class CustomAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CustomAttentionLayer, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='attention_weight'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='attention_bias'
        )
        
    def call(self, inputs):
        # Attention計算
        attention_scores = tf.matmul(inputs, self.W) + self.b
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # 重み付き和
        weighted_sum = tf.reduce_sum(
            tf.expand_dims(attention_weights, -1) * tf.expand_dims(inputs, -2), 
            axis=-2
        )
        
        return weighted_sum, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units
        })
        return config

# カスタムレイヤーを使ったモデル
def create_model_with_attention():
    inputs = tf.keras.Input(shape=(10, 32))
    
    # カスタムAttentionレイヤー
    attended, weights = CustomAttentionLayer(64)(inputs)
    
    # 後続の処理
    x = tf.keras.layers.Dense(32, activation='relu')(attended)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

attention_model = create_model_with_attention()
attention_model.summary()
```

### TensorFlow Serving用のモデル保存

```python
# モデルの保存（SavedModel形式）
import os

# バージョン管理されたモデルパス
model_version = "1"
export_path = os.path.join("./saved_models/my_model", model_version)

# モデルの保存
model.save(export_path)

# 保存されたモデルの確認
!saved_model_cli show --dir {export_path} --all

# 推論用の署名を指定して保存
class ExportModel(tf.Module):
    def __init__(self, model):
        self.model = model
    
    @tf.function
    def serving_fn(self, input_data):
        return {
            'predictions': self.model(input_data),
            'confidence': tf.nn.softmax(self.model(input_data))
        }

export_model = ExportModel(model)
tf.saved_model.save(
    export_model, 
    export_path,
    signatures={'serving_default': export_model.serving_fn}
)
```

### TensorFlow Lite変換

```python
# TFLiteへの変換
converter = tf.lite.TFLiteConverter.from_saved_model(export_path)

# 最適化オプション
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 量子化
converter.representative_dataset = lambda: representative_dataset_gen()
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# 変換
tflite_model = converter.convert()

# 保存
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# モデルサイズの確認
import os
print(f"TFLiteモデルサイズ: {os.path.getsize('model.tflite') / 1024:.2f} KB")
```

## PyTorch詳細

### 高度なモデル設計

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

# カスタムResNet
class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# モデルのインスタンス化
model = CustomResNet(num_classes=10)
print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Mixed Precision設定
scaler = GradScaler()

def train_with_amp(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Automatic Mixed Precision
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # スケーリングされた勾配の計算
        scaler.scale(loss).backward()
        
        # オプティマイザのステップ
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: {loss.item():.4f}')
    
    return total_loss / len(dataloader)
```

### 分散学習

```python
import torch.distributed as dist
import torch.nn.parallel
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # プロセスグループの初期化
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_distributed(rank, world_size):
    setup(rank, world_size)
    
    # モデルの作成
    model = CustomResNet().to(rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # データローダー（分散用）
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        transform=transforms.ToTensor()
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler
    )
    
    # 学習ループ
    optimizer = torch.optim.Adam(ddp_model.parameters())
    
    for epoch in range(10):
        sampler.set_epoch(epoch)
        
        for data, target in dataloader:
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    
    cleanup()

# 分散学習の実行
# world_size = torch.cuda.device_count()
# mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
```

### モデルの量子化

```python
import torch.quantization

# 量子化用のモデル準備
class QuantizableModel(nn.Module):
    def __init__(self):
        super(QuantizableModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 30 * 30, 10)
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

# 量子化の実行
model = QuantizableModel()
model.eval()

# 量子化設定
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# キャリブレーション（代表的なデータで実行）
with torch.no_grad():
    for _ in range(10):
        x = torch.randn(1, 3, 32, 32)
        model(x)

# 量子化の適用
torch.quantization.convert(model, inplace=True)

# モデルサイズの比較
def print_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    print(f"モデルサイズ: {size:.2f} MB")

print_model_size(model)
```

## JAX/Flax

### JAXの基礎

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import flax.linen as nn

# JAXの自動微分
def loss_fn(params, x, y):
    predictions = jnp.dot(x, params['w']) + params['b']
    return jnp.mean((predictions - y) ** 2)

# 勾配関数の作成
grad_fn = grad(loss_fn)

# JITコンパイル
@jit
def update(params, x, y, learning_rate):
    grads = grad_fn(params, x, y)
    return {
        'w': params['w'] - learning_rate * grads['w'],
        'b': params['b'] - learning_rate * grads['b']
    }

# ベクトル化
batched_predict = vmap(lambda x, params: jnp.dot(x, params['w']) + params['b'], 
                       in_axes=(0, None))
```

### Flaxでのモデル定義

```python
class CNN(nn.Module):
    num_classes: int
    
    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        x = nn.Dense(features=self.num_classes)(x)
        
        return x

# モデルの初期化
import flax
from flax.training import train_state
import optax

rng = jax.random.PRNGKey(0)
model = CNN(num_classes=10)

# ダミー入力で初期化
dummy_input = jnp.ones((1, 32, 32, 3))
params = model.init(rng, dummy_input)

# Optimizerとトレーニング状態
tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx
)

# トレーニングステップ
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply(params, batch['image'], training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss
```

## モデルの最適化テクニック

### プルーニング

```python
import torch.nn.utils.prune as prune

def apply_pruning(model, amount=0.3):
    """モデルの重みをプルーニング"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    return model

# プルーニングの適用
pruned_model = apply_pruning(model.cpu(), amount=0.3)

# プルーニングされたパラメータの確認
def count_parameters(model):
    total = 0
    pruned = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total += param.numel()
            if hasattr(param, '_mask'):
                pruned += (param._mask == 0).sum().item()
    
    return total, pruned

total, pruned = count_parameters(pruned_model)
print(f"総パラメータ数: {total:,}")
print(f"プルーニングされたパラメータ数: {pruned:,} ({pruned/total*100:.1f}%)")
```

### Knowledge Distillation

```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets loss
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = self.criterion(soft_prob, soft_targets) * (self.temperature ** 2)
        
        # Hard targets loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return loss

# 蒸留の実行
def train_with_distillation(student_model, teacher_model, dataloader, 
                           optimizer, device, temperature=3.0):
    teacher_model.eval()
    student_model.train()
    
    distill_loss = DistillationLoss(temperature=temperature)
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        
        # Teacher予測
        with torch.no_grad():
            teacher_output = teacher_model(data)
        
        # Student予測
        student_output = student_model(data)
        
        # Loss計算
        loss = distill_loss(student_output, teacher_output, target)
        
        # 最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## デプロイメント準備

### ONNX変換

```python
# PyTorchモデルのONNX変換
dummy_input = torch.randn(1, 3, 224, 224)
model.eval()

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# ONNXモデルの検証
import onnx
import onnxruntime

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# ONNXRuntimeでの推論
ort_session = onnxruntime.InferenceSession("model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 推論実行
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outputs = ort_session.run(None, ort_inputs)
```

### TorchScript

```python
# TorchScriptへの変換
# Method 1: Tracing
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("traced_model.pt")

# Method 2: Scripting
class ScriptableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 30 * 30, 10)
    
    @torch.jit.script_method
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

scripted_model = torch.jit.script(ScriptableModel())
scripted_model.save("scripted_model.pt")

# C++での読み込み用コード生成
print("""
// C++でのモデル読み込み
#include <torch/script.h>

torch::jit::script::Module module;
module = torch::jit::load("model.pt");

// 推論
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::ones({1, 3, 224, 224}));
at::Tensor output = module.forward(inputs).toTensor();
""")
```

## パフォーマンス最適化

### プロファイリング

```python
# PyTorchプロファイラー
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        for _ in range(100):
            model(dummy_input)

# 結果の表示
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# TensorBoardへの出力
prof.export_chrome_trace("trace.json")

# TensorFlowプロファイラー
import tensorflow as tf

# プロファイリング用のコールバック
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    profile_batch='10,20'
)

# モデルの学習（プロファイリング付き）
# model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])
```

### メモリ最適化

```python
# 勾配チェックポイント（PyTorch）
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1000, 1000) for _ in range(10)
        ])
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < 5:
                # 最初の5層はチェックポイント
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x

# メモリ使用量の監視
def print_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU メモリ使用量: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU メモリ予約量: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
```

## まとめ

ディープラーニングフレームワークの重要なポイント：

1. **フレームワーク選択**：用途に応じた適切な選択
2. **カスタム実装**：レイヤー、損失関数、学習ループ
3. **最適化技術**：量子化、プルーニング、蒸留
4. **分散学習**：大規模モデルの効率的な学習
5. **デプロイメント**：本番環境への展開準備

これらの技術を組み合わせることで、高性能で実用的なディープラーニングシステムを構築できます。

## 次へ

[MLOps入門](../04_MLOps/README.md)へ進む