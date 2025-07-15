# MLOps入門

## 概要

MLOps（Machine Learning Operations）は、機械学習モデルの開発から本番環境での運用まで、ライフサイクル全体を管理するための実践とツールの集合です。DevOpsの原則を機械学習に適用し、モデルの再現性、スケーラビリティ、信頼性を確保します。

## MLOpsの主要コンポーネント

### 1. バージョン管理

#### データバージョン管理（DVC）

```python
# DVCのセットアップ
# pip install dvc

import os
import pandas as pd

# データの準備
data = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})

# データの保存
os.makedirs('data', exist_ok=True)
data.to_csv('data/train.csv', index=False)

# DVCでトラッキング
# !dvc init
# !dvc add data/train.csv
# !git add data/train.csv.dvc .gitignore
# !git commit -m "Add training data"

# リモートストレージの設定
# !dvc remote add -d myremote s3://mybucket/path
# !dvc push
```

#### モデルバージョン管理

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# MLflowの設定
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("model_versioning_example")

# データの準備
X = data[['feature1', 'feature2']].values
y = data['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 実験の記録
with mlflow.start_run(run_name="rf_baseline"):
    # パラメータの記録
    n_estimators = 100
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # モデルの学習
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 予測と評価
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # メトリクスの記録
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # モデルの保存
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="RandomForestClassifier"
    )
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
```

### 2. 実験管理

#### Weights & Biases（wandb）

```python
import wandb
from sklearn.ensemble import GradientBoostingClassifier

# wandbの初期化
wandb.init(project="mlops-demo", name="gradient-boosting-experiment")

# ハイパーパラメータ設定
config = wandb.config
config.n_estimators = 200
config.learning_rate = 0.1
config.max_depth = 5

# モデルの学習
model = GradientBoostingClassifier(
    n_estimators=config.n_estimators,
    learning_rate=config.learning_rate,
    max_depth=config.max_depth,
    random_state=42
)

# 学習プロセスの記録
for i in range(10):
    # ミニバッチ学習のシミュレーション
    batch_size = len(X_train) // 10
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    
    X_batch = X_train[start_idx:end_idx]
    y_batch = y_train[start_idx:end_idx]
    
    if i == 0:
        model.fit(X_batch, y_batch)
    else:
        model.fit(X_batch, y_batch)  # 実際はwarm_start=Trueが必要
    
    # バッチごとの評価
    train_score = model.score(X_batch, y_batch)
    val_score = model.score(X_test, y_test)
    
    wandb.log({
        "batch": i,
        "train_accuracy": train_score,
        "val_accuracy": val_score
    })

# 最終評価
final_accuracy = model.score(X_test, y_test)
wandb.log({"final_accuracy": final_accuracy})

# アーティファクトの保存
import joblib
joblib.dump(model, "model.pkl")
wandb.save("model.pkl")

wandb.finish()
```

#### Neptune.ai

```python
import neptune.new as neptune
from neptune.new.types import File

# Neptuneの初期化
run = neptune.init_run(
    project="workspace/mlops-demo",
    api_token="YOUR_API_TOKEN"
)

# パラメータの記録
params = {
    "algorithm": "XGBoost",
    "max_depth": 6,
    "learning_rate": 0.3,
    "n_estimators": 100
}
run["parameters"] = params

# モデルの学習（XGBoost）
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 学習の記録
def log_evaluation(period, logger):
    def callback(env):
        if env.iteration % period == 0:
            logger["train/error"].log(env.evaluation_result_list[0][1])
            if len(env.evaluation_result_list) > 1:
                logger["valid/error"].log(env.evaluation_result_list[1][1])
    return callback

xgb_params = {
    'max_depth': params['max_depth'],
    'learning_rate': params['learning_rate'],
    'objective': 'binary:logistic',
    'eval_metric': 'error'
}

watchlist = [(dtrain, 'train'), (dtest, 'valid')]
model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=params['n_estimators'],
    evals=watchlist,
    callbacks=[log_evaluation(period=10, logger=run)],
    verbose_eval=False
)

# 特徴量の重要度
importance = model.get_score(importance_type='gain')
run["feature_importance"] = importance

# モデルの保存
model.save_model("xgb_model.json")
run["model"].upload(File("xgb_model.json"))

run.stop()
```

### 3. CI/CD パイプライン

#### GitHub Actions

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-and-train:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest flake8
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Run tests
      run: |
        pytest tests/
    
    - name: Train model
      run: |
        python scripts/train_model.py
    
    - name: Evaluate model
      run: |
        python scripts/evaluate_model.py
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v2
      with:
        name: model-artifacts
        path: models/
```

#### Pythonでのパイプライン実装

```python
# scripts/train_model.py
import os
import sys
import yaml
import joblib
from datetime import datetime

class MLPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_dir = self.config['model']['save_dir']
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_data(self):
        """データの読み込みと前処理"""
        print("Loading data...")
        # DVC経由でデータを取得
        os.system("dvc pull")
        
        data = pd.read_csv(self.config['data']['train_path'])
        return self.preprocess_data(data)
    
    def preprocess_data(self, data):
        """データの前処理"""
        print("Preprocessing data...")
        # 欠損値処理
        data = data.dropna()
        
        # 特徴量とターゲットの分離
        feature_cols = self.config['features']['columns']
        target_col = self.config['features']['target']
        
        X = data[feature_cols]
        y = data[target_col]
        
        # スケーリング
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # スケーラーの保存
        joblib.dump(scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        
        return X_scaled, y
    
    def train_model(self, X, y):
        """モデルの学習"""
        print("Training model...")
        
        # モデルの選択
        model_type = self.config['model']['type']
        model_params = self.config['model']['parameters']
        
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**model_params)
        elif model_type == 'xgboost':
            import xgboost as xgb
            model = xgb.XGBClassifier(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 学習
        model.fit(X, y)
        
        # モデルの保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f'model_{timestamp}.pkl')
        joblib.dump(model, model_path)
        
        # 最新モデルへのシンボリックリンク
        latest_path = os.path.join(self.model_dir, 'latest_model.pkl')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(model_path, latest_path)
        
        return model
    
    def validate_model(self, model, X, y):
        """モデルの検証"""
        print("Validating model...")
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        validation_results = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'cv_scores': scores.tolist()
        }
        
        # 検証結果の保存
        with open(os.path.join(self.model_dir, 'validation_results.yaml'), 'w') as f:
            yaml.dump(validation_results, f)
        
        # 閾値チェック
        min_accuracy = self.config['validation']['min_accuracy']
        if validation_results['mean_accuracy'] < min_accuracy:
            raise ValueError(f"Model accuracy {validation_results['mean_accuracy']:.3f} "
                           f"is below threshold {min_accuracy}")
        
        return validation_results
    
    def run(self):
        """パイプラインの実行"""
        try:
            # データ読み込み
            X, y = self.load_data()
            
            # モデル学習
            model = self.train_model(X, y)
            
            # モデル検証
            results = self.validate_model(model, X, y)
            
            print(f"Pipeline completed successfully!")
            print(f"Model accuracy: {results['mean_accuracy']:.3f}")
            
            return 0
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            return 1

if __name__ == "__main__":
    pipeline = MLPipeline('config/pipeline_config.yaml')
    sys.exit(pipeline.run())
```

### 4. モデルサービング

#### FastAPI によるモデルAPI

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPIアプリケーション
app = FastAPI(title="ML Model API", version="1.0.0")

# モデルとスケーラーの読み込み
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        model = joblib.load("models/latest_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        logger.info("Model and scaler loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

# リクエスト/レスポンスモデル
class PredictionRequest(BaseModel):
    features: List[float]
    
class PredictionResponse(BaseModel):
    prediction: int
    probability: List[float]
    model_version: str

# ヘルスチェック
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# 予測エンドポイント
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # 入力の検証
        features = np.array(request.features).reshape(1, -1)
        
        # スケーリング
        features_scaled = scaler.transform(features)
        
        # 予測
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0].tolist()
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# バッチ予測エンドポイント
class BatchPredictionRequest(BaseModel):
    instances: List[List[float]]

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    try:
        # バッチ処理
        features = np.array(request.instances)
        features_scaled = scaler.transform(features)
        
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "prediction": int(pred),
                "probability": prob.tolist()
            })
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Dockerfile
"""
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
```

#### Kubernetes デプロイメント

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-api
  labels:
    app: ml-model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model-api
  template:
    metadata:
      labels:
        app: ml-model-api
    spec:
      containers:
      - name: ml-model-api
        image: myregistry/ml-model-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-api-service
spec:
  selector:
    app: ml-model-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### 5. モニタリングとロギング

#### Prometheus + Grafana

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# メトリクスの定義
prediction_counter = Counter('model_predictions_total', 
                           'Total number of predictions')
prediction_latency = Histogram('model_prediction_duration_seconds',
                             'Prediction latency')
model_accuracy_gauge = Gauge('model_accuracy', 
                           'Current model accuracy')

# メトリクスの記録
def record_prediction_metrics(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            prediction_counter.inc()
            return result
        finally:
            prediction_latency.observe(time.time() - start_time)
    
    return wrapper

# 使用例
@record_prediction_metrics
def make_prediction(features):
    # 予測処理
    return model.predict(features)
```

#### カスタムロギング

```python
import logging
import json
from datetime import datetime
from pythonjsonlogger import jsonlogger

# 構造化ログの設定
def setup_logging():
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    logHandler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logHandler)
    
    return logger

logger = setup_logging()

class ModelMonitor:
    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version
        self.prediction_log = []
        
    def log_prediction(self, input_data, prediction, confidence, latency):
        """予測のログ記録"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "input_features": input_data.tolist() if hasattr(input_data, 'tolist') else input_data,
            "prediction": prediction,
            "confidence": confidence,
            "latency_ms": latency * 1000,
            "event_type": "prediction"
        }
        
        logger.info("Model prediction", extra=log_entry)
        self.prediction_log.append(log_entry)
        
        # データドリフトの検出
        self.check_data_drift(input_data)
    
    def check_data_drift(self, input_data):
        """データドリフトの検出"""
        # 簡易的なドリフト検出
        if len(self.prediction_log) > 100:
            recent_inputs = [log['input_features'] for log in self.prediction_log[-100:]]
            
            # 統計的な検定（例：Kolmogorov-Smirnov test）
            # ここでは簡略化
            mean_drift = np.abs(np.mean(recent_inputs) - self.expected_mean)
            
            if mean_drift > self.drift_threshold:
                logger.warning("Data drift detected", extra={
                    "model_name": self.model_name,
                    "drift_score": mean_drift,
                    "event_type": "drift_alert"
                })
```

### 6. A/Bテストとカナリアデプロイ

```python
# ab_testing.py
import random
from typing import Dict, Any

class ABTestManager:
    def __init__(self):
        self.experiments = {}
        
    def create_experiment(self, name: str, variants: Dict[str, Any], 
                         traffic_split: Dict[str, float]):
        """A/Bテストの作成"""
        self.experiments[name] = {
            'variants': variants,
            'traffic_split': traffic_split,
            'metrics': {variant: [] for variant in variants}
        }
    
    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """ユーザーに割り当てるバリアントを決定"""
        experiment = self.experiments[experiment_name]
        
        # ユーザーIDベースで一貫性のある割り当て
        random.seed(hash(f"{experiment_name}_{user_id}"))
        rand_val = random.random()
        
        cumulative_prob = 0
        for variant, prob in experiment['traffic_split'].items():
            cumulative_prob += prob
            if rand_val < cumulative_prob:
                return variant
        
        return list(experiment['variants'].keys())[-1]
    
    def record_metric(self, experiment_name: str, variant: str, 
                     metric_name: str, value: float):
        """メトリクスの記録"""
        self.experiments[experiment_name]['metrics'][variant].append({
            'metric_name': metric_name,
            'value': value,
            'timestamp': datetime.utcnow()
        })
    
    def get_results(self, experiment_name: str) -> Dict:
        """実験結果の取得"""
        experiment = self.experiments[experiment_name]
        results = {}
        
        for variant, metrics in experiment['metrics'].items():
            if metrics:
                values = [m['value'] for m in metrics if m['metric_name'] == 'accuracy']
                results[variant] = {
                    'mean_accuracy': np.mean(values),
                    'std_accuracy': np.std(values),
                    'sample_size': len(values)
                }
        
        return results

# 使用例
ab_manager = ABTestManager()

# 新旧モデルのA/Bテスト
ab_manager.create_experiment(
    name="model_v2_test",
    variants={
        'control': 'model_v1.pkl',
        'treatment': 'model_v2.pkl'
    },
    traffic_split={
        'control': 0.8,
        'treatment': 0.2
    }
)

# カナリアデプロイメント設定
class CanaryDeployment:
    def __init__(self, stable_model, canary_model, initial_canary_weight=0.1):
        self.stable_model = stable_model
        self.canary_model = canary_model
        self.canary_weight = initial_canary_weight
        self.metrics_buffer = []
        
    def predict(self, features):
        """カナリアデプロイメントでの予測"""
        if random.random() < self.canary_weight:
            model = self.canary_model
            model_version = "canary"
        else:
            model = self.stable_model
            model_version = "stable"
        
        start_time = time.time()
        prediction = model.predict(features)
        latency = time.time() - start_time
        
        # メトリクスの記録
        self.metrics_buffer.append({
            'model_version': model_version,
            'latency': latency,
            'timestamp': datetime.utcnow()
        })
        
        return prediction
    
    def adjust_traffic(self):
        """トラフィックの自動調整"""
        if len(self.metrics_buffer) < 100:
            return
        
        # 直近のメトリクスを分析
        recent_metrics = self.metrics_buffer[-100:]
        
        canary_metrics = [m for m in recent_metrics if m['model_version'] == 'canary']
        stable_metrics = [m for m in recent_metrics if m['model_version'] == 'stable']
        
        if canary_metrics and stable_metrics:
            canary_latency = np.mean([m['latency'] for m in canary_metrics])
            stable_latency = np.mean([m['latency'] for m in stable_metrics])
            
            # カナリーモデルのパフォーマンスが良好な場合、トラフィックを増やす
            if canary_latency < stable_latency * 1.1:  # 10%の許容範囲
                self.canary_weight = min(1.0, self.canary_weight + 0.1)
                logger.info(f"Increasing canary traffic to {self.canary_weight:.1%}")
            else:
                self.canary_weight = max(0.0, self.canary_weight - 0.1)
                logger.warning(f"Decreasing canary traffic to {self.canary_weight:.1%}")
```

## MLOpsのベストプラクティス

### 1. 再現性の確保

```python
# requirements.txt の自動生成
import pkg_resources

def generate_requirements():
    """現在の環境の requirements.txt を生成"""
    packages = []
    for dist in pkg_resources.working_set:
        packages.append(f"{dist.project_name}=={dist.version}")
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(sorted(packages)))

# Dockerfileでの再現性
"""
FROM python:3.8-slim

# 固定バージョンの指定
RUN pip install --no-cache-dir \
    numpy==1.21.0 \
    pandas==1.3.0 \
    scikit-learn==0.24.2 \
    mlflow==1.19.0

# シード値の固定
ENV PYTHONHASHSEED=42
"""

# コード内でのシード固定
import random
import numpy as np
import tensorflow as tf

def set_seeds(seed=42):
    """すべての乱数シードを固定"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
```

### 2. 自動テスト

```python
# tests/test_model.py
import pytest
import numpy as np
from sklearn.datasets import make_classification

class TestModel:
    @pytest.fixture
    def sample_data(self):
        """テスト用データの生成"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return X, y
    
    def test_model_prediction_shape(self, sample_data):
        """予測の形状をテスト"""
        X, y = sample_data
        model = load_model('models/latest_model.pkl')
        
        predictions = model.predict(X)
        assert predictions.shape == (100,)
    
    def test_model_prediction_range(self, sample_data):
        """予測値の範囲をテスト"""
        X, y = sample_data
        model = load_model('models/latest_model.pkl')
        
        predictions = model.predict_proba(X)
        assert np.all(predictions >= 0) and np.all(predictions <= 1)
    
    def test_model_performance(self, sample_data):
        """モデルの性能をテスト"""
        X, y = sample_data
        model = load_model('models/latest_model.pkl')
        
        accuracy = model.score(X, y)
        assert accuracy > 0.7  # 最低限の精度
```

### 3. 設定管理

```yaml
# config/config.yaml
model:
  type: "xgboost"
  parameters:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  save_dir: "models/"

data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  validation_split: 0.2

training:
  batch_size: 32
  epochs: 100
  early_stopping_patience: 10

deployment:
  api_port: 8000
  max_batch_size: 100
  timeout_seconds: 30

monitoring:
  log_predictions: true
  drift_detection: true
  alert_threshold: 0.1
```

## まとめ

MLOpsの重要な要素：

1. **バージョン管理**：コード、データ、モデルの追跡
2. **自動化**：CI/CDパイプラインによる自動化
3. **モニタリング**：本番環境での性能監視
4. **再現性**：実験の再現可能性の確保
5. **スケーラビリティ**：負荷に応じた拡張性

これらの実践により、機械学習モデルを安定的に本番環境で運用できます。

## 次へ

[応用分野](../../05_応用/README.md)へ進む