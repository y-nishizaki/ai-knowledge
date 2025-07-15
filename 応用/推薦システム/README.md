# 推薦システム

## 概要

推薦システムは、ユーザーの嗜好を学習し、関連性の高いアイテムを提案する技術です。ECサイト、動画配信、音楽ストリーミングなど、様々なサービスで活用されています。協調フィルタリング、コンテンツベースフィルタリング、そして深層学習を使った最新手法まで解説します。

## 協調フィルタリング

### メモリベース協調フィルタリング

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

class UserBasedCF:
    """ユーザーベース協調フィルタリング"""
    def __init__(self, k_neighbors=10):
        self.k_neighbors = k_neighbors
        self.user_similarity = None
        self.ratings_matrix = None
    
    def fit(self, ratings_df):
        """評価データから類似度行列を構築"""
        # ユーザー×アイテムの評価行列を作成
        self.ratings_matrix = ratings_df.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # ユーザー間の類似度計算
        self.user_similarity = cosine_similarity(self.ratings_matrix)
        np.fill_diagonal(self.user_similarity, 0)
        
        return self
    
    def predict(self, user_id, item_id):
        """特定のユーザー・アイテムペアの評価を予測"""
        if user_id not in self.ratings_matrix.index:
            return self.ratings_matrix.mean().mean()
        
        user_idx = self.ratings_matrix.index.get_loc(user_id)
        
        # k近傍ユーザーを取得
        similarities = self.user_similarity[user_idx]
        k_nearest_indices = np.argsort(similarities)[-self.k_neighbors:]
        
        # 加重平均で予測
        numerator = 0
        denominator = 0
        
        for neighbor_idx in k_nearest_indices:
            if item_id in self.ratings_matrix.columns:
                item_idx = self.ratings_matrix.columns.get_loc(item_id)
                neighbor_rating = self.ratings_matrix.iloc[neighbor_idx, item_idx]
                
                if neighbor_rating > 0:
                    similarity = similarities[neighbor_idx]
                    numerator += similarity * neighbor_rating
                    denominator += abs(similarity)
        
        if denominator > 0:
            return numerator / denominator
        else:
            return self.ratings_matrix.iloc[user_idx].mean()
    
    def recommend(self, user_id, n_recommendations=10):
        """ユーザーに対するトップN推薦"""
        if user_id not in self.ratings_matrix.index:
            # コールドスタート：人気アイテムを推薦
            popular_items = self.ratings_matrix.mean().sort_values(ascending=False)
            return popular_items.head(n_recommendations).index.tolist()
        
        user_idx = self.ratings_matrix.index.get_loc(user_id)
        user_ratings = self.ratings_matrix.iloc[user_idx]
        
        # 未評価アイテムの予測評価を計算
        predictions = {}
        for item_id in self.ratings_matrix.columns:
            if user_ratings[item_id] == 0:  # 未評価アイテム
                predictions[item_id] = self.predict(user_id, item_id)
        
        # トップNを返す
        sorted_predictions = sorted(predictions.items(), 
                                  key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_predictions[:n_recommendations]]

class ItemBasedCF:
    """アイテムベース協調フィルタリング"""
    def __init__(self, k_neighbors=10):
        self.k_neighbors = k_neighbors
        self.item_similarity = None
        self.ratings_matrix = None
    
    def fit(self, ratings_df):
        """評価データから類似度行列を構築"""
        # ユーザー×アイテムの評価行列を作成
        self.ratings_matrix = ratings_df.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # アイテム間の類似度計算
        self.item_similarity = cosine_similarity(self.ratings_matrix.T)
        np.fill_diagonal(self.item_similarity, 0)
        
        return self
    
    def predict(self, user_id, item_id):
        """特定のユーザー・アイテムペアの評価を予測"""
        if user_id not in self.ratings_matrix.index:
            return self.ratings_matrix.mean().mean()
        
        if item_id not in self.ratings_matrix.columns:
            return self.ratings_matrix.mean().mean()
        
        user_ratings = self.ratings_matrix.loc[user_id]
        item_idx = self.ratings_matrix.columns.get_loc(item_id)
        
        # アイテムの類似度
        similarities = self.item_similarity[item_idx]
        
        # 加重平均で予測
        numerator = 0
        denominator = 0
        
        for other_item_idx, rating in enumerate(user_ratings):
            if rating > 0 and other_item_idx != item_idx:
                similarity = similarities[other_item_idx]
                numerator += similarity * rating
                denominator += abs(similarity)
        
        if denominator > 0:
            return numerator / denominator
        else:
            return user_ratings.mean()
```

### モデルベース協調フィルタリング

```python
from sklearn.decomposition import NMF, TruncatedSVD
import tensorflow as tf
from tensorflow.keras import layers, Model

class MatrixFactorization:
    """行列分解による協調フィルタリング"""
    def __init__(self, n_factors=50, learning_rate=0.01, n_epochs=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
    
    def fit(self, ratings_df):
        """確率的勾配降下法で学習"""
        # ユーザーとアイテムのマッピング
        self.user_ids = ratings_df['user_id'].unique()
        self.item_ids = ratings_df['item_id'].unique()
        
        user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        item_id_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
        
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        
        # パラメータの初期化
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_mean = ratings_df['rating'].mean()
        
        # 学習
        for epoch in range(self.n_epochs):
            # データをシャッフル
            shuffled_df = ratings_df.sample(frac=1)
            
            for _, row in shuffled_df.iterrows():
                user_idx = user_id_map[row['user_id']]
                item_idx = item_id_map[row['item_id']]
                rating = row['rating']
                
                # 予測値
                prediction = (self.global_mean + 
                            self.user_biases[user_idx] + 
                            self.item_biases[item_idx] +
                            np.dot(self.user_factors[user_idx], 
                                  self.item_factors[item_idx]))
                
                # 誤差
                error = rating - prediction
                
                # パラメータ更新
                self.user_biases[user_idx] += self.learning_rate * (
                    error - 0.01 * self.user_biases[user_idx]
                )
                self.item_biases[item_idx] += self.learning_rate * (
                    error - 0.01 * self.item_biases[item_idx]
                )
                
                user_factors_grad = error * self.item_factors[item_idx] - \
                                   0.01 * self.user_factors[user_idx]
                item_factors_grad = error * self.user_factors[user_idx] - \
                                   0.01 * self.item_factors[item_idx]
                
                self.user_factors[user_idx] += self.learning_rate * user_factors_grad
                self.item_factors[item_idx] += self.learning_rate * item_factors_grad
            
            if epoch % 10 == 0:
                rmse = self.compute_rmse(ratings_df)
                print(f"Epoch {epoch}: RMSE = {rmse:.4f}")
    
    def predict(self, user_id, item_id):
        """評価値を予測"""
        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
            item_idx = np.where(self.item_ids == item_id)[0][0]
            
            prediction = (self.global_mean + 
                        self.user_biases[user_idx] + 
                        self.item_biases[item_idx] +
                        np.dot(self.user_factors[user_idx], 
                              self.item_factors[item_idx]))
            
            return np.clip(prediction, 1, 5)  # 1-5の範囲にクリップ
        except:
            return self.global_mean
    
    def compute_rmse(self, ratings_df):
        """RMSEを計算"""
        predictions = []
        actuals = []
        
        for _, row in ratings_df.iterrows():
            pred = self.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
            actuals.append(row['rating'])
        
        return np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))

# SVDによる実装
class SVDRecommender:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components)
        self.ratings_matrix = None
    
    def fit(self, ratings_df):
        # 評価行列の作成
        self.ratings_matrix = ratings_df.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)
        
        # SVD分解
        self.user_features = self.svd.fit_transform(self.ratings_matrix)
        self.item_features = self.svd.components_.T
        
        return self
    
    def predict_all(self):
        """全ユーザー・アイテムペアの予測"""
        predictions = np.dot(self.user_features, self.item_features.T)
        return pd.DataFrame(
            predictions,
            index=self.ratings_matrix.index,
            columns=self.ratings_matrix.columns
        )
```

## コンテンツベースフィルタリング

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.item_features = None
        self.item_ids = None
        self.item_similarity = None
    
    def fit(self, items_df, text_column='description'):
        """アイテムの特徴量を学習"""
        self.item_ids = items_df['item_id'].values
        
        # テキスト特徴量の抽出
        tfidf_matrix = self.tfidf.fit_transform(items_df[text_column])
        self.item_features = tfidf_matrix
        
        # アイテム間類似度の計算
        self.item_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        return self
    
    def get_user_profile(self, user_ratings_df, items_df):
        """ユーザープロファイルの構築"""
        # ユーザーが高評価したアイテムのインデックス
        liked_items = user_ratings_df[user_ratings_df['rating'] >= 4]['item_id']
        liked_indices = [
            i for i, item_id in enumerate(self.item_ids) 
            if item_id in liked_items.values
        ]
        
        if not liked_indices:
            return None
        
        # 高評価アイテムの特徴量の平均
        user_profile = self.item_features[liked_indices].mean(axis=0)
        
        return user_profile
    
    def recommend(self, user_ratings_df, items_df, n_recommendations=10):
        """コンテンツベース推薦"""
        user_profile = self.get_user_profile(user_ratings_df, items_df)
        
        if user_profile is None:
            # コールドスタート：人気アイテムを返す
            return []
        
        # ユーザープロファイルとアイテムの類似度
        similarities = linear_kernel(user_profile, self.item_features).flatten()
        
        # 既に評価済みのアイテムを除外
        rated_items = set(user_ratings_df['item_id'])
        
        recommendations = []
        for idx in similarities.argsort()[::-1]:
            item_id = self.item_ids[idx]
            if item_id not in rated_items:
                recommendations.append(item_id)
                if len(recommendations) >= n_recommendations:
                    break
        
        return recommendations

# 複数の特徴量を使用
class MultiFeatureContentRecommender:
    def __init__(self):
        self.feature_weights = {
            'text': 0.4,
            'category': 0.3,
            'tags': 0.3
        }
    
    def extract_features(self, items_df):
        """複数の特徴量を抽出・結合"""
        features = []
        
        # テキスト特徴量
        if 'description' in items_df.columns:
            tfidf = TfidfVectorizer(max_features=1000)
            text_features = tfidf.fit_transform(items_df['description'])
            features.append(('text', text_features, self.feature_weights['text']))
        
        # カテゴリ特徴量
        if 'category' in items_df.columns:
            category_encoded = pd.get_dummies(items_df['category'])
            features.append(('category', category_encoded.values, 
                           self.feature_weights['category']))
        
        # タグ特徴量
        if 'tags' in items_df.columns:
            # マルチラベルエンコーディング
            mlb = MultiLabelBinarizer()
            tags_encoded = mlb.fit_transform(items_df['tags'].str.split(','))
            features.append(('tags', tags_encoded, self.feature_weights['tags']))
        
        return features
    
    def combine_similarities(self, features_list):
        """重み付き類似度の結合"""
        combined_similarity = None
        
        for feature_name, feature_matrix, weight in features_list:
            if hasattr(feature_matrix, 'toarray'):
                feature_matrix = feature_matrix.toarray()
            
            similarity = cosine_similarity(feature_matrix)
            
            if combined_similarity is None:
                combined_similarity = weight * similarity
            else:
                combined_similarity += weight * similarity
        
        return combined_similarity
```

## ハイブリッド推薦システム

```python
class HybridRecommender:
    def __init__(self, cf_weight=0.6, cb_weight=0.4):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_recommender = ItemBasedCF()
        self.cb_recommender = ContentBasedRecommender()
        
    def fit(self, ratings_df, items_df):
        """両方のモデルを学習"""
        self.cf_recommender.fit(ratings_df)
        self.cb_recommender.fit(items_df)
        return self
    
    def recommend(self, user_id, user_ratings_df, items_df, n_recommendations=10):
        """ハイブリッド推薦"""
        # 協調フィルタリングの推薦
        cf_recommendations = self.cf_recommender.recommend(user_id, n_recommendations * 2)
        
        # コンテンツベースの推薦
        cb_recommendations = self.cb_recommender.recommend(
            user_ratings_df[user_ratings_df['user_id'] == user_id],
            items_df,
            n_recommendations * 2
        )
        
        # スコアの結合
        recommendation_scores = {}
        
        # CFスコア
        for i, item_id in enumerate(cf_recommendations):
            score = self.cf_weight * (len(cf_recommendations) - i) / len(cf_recommendations)
            recommendation_scores[item_id] = recommendation_scores.get(item_id, 0) + score
        
        # CBスコア
        for i, item_id in enumerate(cb_recommendations):
            score = self.cb_weight * (len(cb_recommendations) - i) / len(cb_recommendations)
            recommendation_scores[item_id] = recommendation_scores.get(item_id, 0) + score
        
        # トップN選択
        sorted_recommendations = sorted(
            recommendation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [item_id for item_id, _ in sorted_recommendations[:n_recommendations]]
```

## 深層学習ベースの推薦

### Neural Collaborative Filtering (NCF)

```python
class NCF(tf.keras.Model):
    def __init__(self, n_users, n_items, embedding_size=50, 
                 mlp_layers=[64, 32, 16, 8]):
        super(NCF, self).__init__()
        
        # 埋め込み層
        self.user_embedding_gmf = layers.Embedding(n_users, embedding_size)
        self.item_embedding_gmf = layers.Embedding(n_items, embedding_size)
        
        self.user_embedding_mlp = layers.Embedding(n_users, embedding_size)
        self.item_embedding_mlp = layers.Embedding(n_items, embedding_size)
        
        # MLP層
        self.mlp_layers = []
        input_size = embedding_size * 2
        for units in mlp_layers:
            self.mlp_layers.append(layers.Dense(units, activation='relu'))
            self.mlp_layers.append(layers.Dropout(0.2))
        
        # 予測層
        self.prediction_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        user_ids, item_ids = inputs
        
        # GMF部分
        user_embedding_gmf = self.user_embedding_gmf(user_ids)
        item_embedding_gmf = self.item_embedding_gmf(item_ids)
        gmf_vector = user_embedding_gmf * item_embedding_gmf
        
        # MLP部分
        user_embedding_mlp = self.user_embedding_mlp(user_ids)
        item_embedding_mlp = self.item_embedding_mlp(item_ids)
        mlp_vector = tf.concat([user_embedding_mlp, item_embedding_mlp], axis=1)
        
        for layer in self.mlp_layers:
            mlp_vector = layer(mlp_vector, training=training)
        
        # 結合
        concat_vector = tf.concat([gmf_vector, mlp_vector], axis=1)
        
        # 予測
        prediction = self.prediction_layer(concat_vector)
        
        return tf.squeeze(prediction)

# データローダー
class NCFDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, ratings_df, n_users, n_items, batch_size=256, 
                 n_negative=4, shuffle=True):
        self.ratings_df = ratings_df
        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = batch_size
        self.n_negative = n_negative
        self.shuffle = shuffle
        
        # ポジティブサンプル
        self.user_item_set = set(
            zip(ratings_df['user_id'], ratings_df['item_id'])
        )
        
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.ratings_df) * (1 + self.n_negative) / 
                          self.batch_size))
    
    def __getitem__(self, index):
        # バッチデータの生成
        start = index * self.batch_size // (1 + self.n_negative)
        end = min((index + 1) * self.batch_size // (1 + self.n_negative), 
                  len(self.ratings_df))
        
        batch_users = []
        batch_items = []
        batch_labels = []
        
        for idx in range(start, end):
            # ポジティブサンプル
            row = self.ratings_df.iloc[idx]
            batch_users.append(row['user_id'])
            batch_items.append(row['item_id'])
            batch_labels.append(1)
            
            # ネガティブサンプル
            for _ in range(self.n_negative):
                neg_item = np.random.randint(0, self.n_items)
                while (row['user_id'], neg_item) in self.user_item_set:
                    neg_item = np.random.randint(0, self.n_items)
                
                batch_users.append(row['user_id'])
                batch_items.append(neg_item)
                batch_labels.append(0)
        
        return ([np.array(batch_users), np.array(batch_items)], 
                np.array(batch_labels))
    
    def on_epoch_end(self):
        if self.shuffle:
            self.ratings_df = self.ratings_df.sample(frac=1).reset_index(drop=True)
```

### Wide & Deep

```python
class WideDeepRecommender(tf.keras.Model):
    def __init__(self, n_users, n_items, n_categories, embedding_size=32):
        super(WideDeepRecommender, self).__init__()
        
        # Wide部分（線形モデル）
        self.wide_user_embedding = layers.Embedding(n_users, 1)
        self.wide_item_embedding = layers.Embedding(n_items, 1)
        self.wide_category_embedding = layers.Embedding(n_categories, 1)
        
        # Deep部分（深層モデル）
        self.deep_user_embedding = layers.Embedding(n_users, embedding_size)
        self.deep_item_embedding = layers.Embedding(n_items, embedding_size)
        self.deep_category_embedding = layers.Embedding(n_categories, embedding_size)
        
        # Deep層
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.dense3 = layers.Dense(32, activation='relu')
        
        # 最終層
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        user_ids, item_ids, category_ids = inputs
        
        # Wide部分
        wide_user = tf.squeeze(self.wide_user_embedding(user_ids), axis=1)
        wide_item = tf.squeeze(self.wide_item_embedding(item_ids), axis=1)
        wide_category = tf.squeeze(self.wide_category_embedding(category_ids), axis=1)
        wide_output = wide_user + wide_item + wide_category
        
        # Deep部分
        deep_user = self.deep_user_embedding(user_ids)
        deep_item = self.deep_item_embedding(item_ids)
        deep_category = self.deep_category_embedding(category_ids)
        
        deep_concat = tf.concat([deep_user, deep_item, deep_category], axis=1)
        deep_output = self.dense1(deep_concat)
        deep_output = self.dropout1(deep_output, training=training)
        deep_output = self.dense2(deep_output)
        deep_output = self.dropout2(deep_output, training=training)
        deep_output = self.dense3(deep_output)
        
        # Wide & Deepの結合
        combined = tf.concat([tf.expand_dims(wide_output, 1), deep_output], axis=1)
        output = self.output_layer(combined)
        
        return tf.squeeze(output)
```

### Factorization Machines

```python
class FactorizationMachines(tf.keras.Model):
    def __init__(self, n_features, k_factors=10):
        super(FactorizationMachines, self).__init__()
        
        # 線形項
        self.w0 = self.add_weight(shape=(1,), initializer='zeros', name='w0')
        self.w = self.add_weight(shape=(n_features,), initializer='zeros', name='w')
        
        # 因子行列
        self.V = self.add_weight(
            shape=(n_features, k_factors),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name='V'
        )
    
    def call(self, inputs):
        # inputs: (batch_size, n_features)
        
        # 線形項
        linear_terms = self.w0 + tf.reduce_sum(self.w * inputs, axis=1)
        
        # 交互作用項
        # sum_square = (Σ v_i * x_i)^2
        sum_square = tf.square(tf.matmul(inputs, self.V))
        
        # square_sum = Σ (v_i * x_i)^2
        square_sum = tf.matmul(tf.square(inputs), tf.square(self.V))
        
        # 交互作用 = 0.5 * Σ (sum_square - square_sum)
        interactions = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1)
        
        return linear_terms + interactions

class DeepFM(tf.keras.Model):
    """Deep Factorization Machines"""
    def __init__(self, n_features, k_factors=10, dnn_hidden_units=[128, 64, 32]):
        super(DeepFM, self).__init__()
        
        # FM部分
        self.fm = FactorizationMachines(n_features, k_factors)
        
        # Deep部分
        self.dnn_layers = []
        for units in dnn_hidden_units:
            self.dnn_layers.append(layers.Dense(units, activation='relu'))
            self.dnn_layers.append(layers.BatchNormalization())
            self.dnn_layers.append(layers.Dropout(0.3))
        
        # 出力層
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        # FM部分
        fm_output = self.fm(inputs)
        
        # Deep部分
        deep_input = inputs
        for layer in self.dnn_layers:
            if isinstance(layer, layers.Dropout) or isinstance(layer, layers.BatchNormalization):
                deep_input = layer(deep_input, training=training)
            else:
                deep_input = layer(deep_input)
        
        # 結合
        combined = tf.concat([
            tf.expand_dims(fm_output, 1), 
            deep_input
        ], axis=1)
        
        output = self.output_layer(combined)
        return tf.squeeze(output)
```

## セッションベース推薦

```python
class GRU4Rec(tf.keras.Model):
    """セッションベース推薦のためのGRUモデル"""
    def __init__(self, n_items, embedding_size=50, gru_units=100):
        super(GRU4Rec, self).__init__()
        
        self.n_items = n_items
        self.embedding = layers.Embedding(n_items, embedding_size)
        self.gru = layers.GRU(gru_units, return_sequences=True)
        self.dropout = layers.Dropout(0.2)
        self.output_layer = layers.Dense(n_items)
    
    def call(self, inputs, training=False):
        # inputs: (batch_size, sequence_length)
        embedded = self.embedding(inputs)
        gru_output = self.gru(embedded)
        gru_output = self.dropout(gru_output, training=training)
        output = self.output_layer(gru_output)
        
        return output

class SessionDataGenerator:
    def __init__(self, sessions_df, batch_size=64):
        self.sessions_df = sessions_df
        self.batch_size = batch_size
        self.sessions = self._prepare_sessions()
    
    def _prepare_sessions(self):
        """セッションデータの準備"""
        sessions = []
        
        for session_id, group in self.sessions_df.groupby('session_id'):
            items = group.sort_values('timestamp')['item_id'].values
            if len(items) > 1:
                sessions.append(items)
        
        return sessions
    
    def generate_batch(self):
        """バッチ生成"""
        while True:
            # セッションをランダムに選択
            batch_sessions = np.random.choice(
                self.sessions, 
                size=self.batch_size
            )
            
            max_len = max(len(s) for s in batch_sessions)
            
            # パディング
            X = np.zeros((self.batch_size, max_len - 1), dtype=np.int32)
            y = np.zeros((self.batch_size, max_len - 1), dtype=np.int32)
            
            for i, session in enumerate(batch_sessions):
                X[i, :len(session)-1] = session[:-1]
                y[i, :len(session)-1] = session[1:]
            
            yield X, y
```

## 評価指標

```python
class RecommenderEvaluator:
    def __init__(self):
        pass
    
    @staticmethod
    def precision_at_k(recommended_items, relevant_items, k):
        """Precision@K"""
        if len(recommended_items) > k:
            recommended_items = recommended_items[:k]
        
        return len(set(recommended_items) & set(relevant_items)) / k
    
    @staticmethod
    def recall_at_k(recommended_items, relevant_items, k):
        """Recall@K"""
        if len(recommended_items) > k:
            recommended_items = recommended_items[:k]
        
        if len(relevant_items) == 0:
            return 0
        
        return len(set(recommended_items) & set(relevant_items)) / len(relevant_items)
    
    @staticmethod
    def ndcg_at_k(recommended_items, relevant_items, k):
        """NDCG@K"""
        def dcg_at_k(r, k):
            r = np.asfarray(r)[:k]
            if r.size:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            return 0.
        
        dcg_max = dcg_at_k(sorted(relevant_items, reverse=True), k)
        if not dcg_max:
            return 0.
        
        return dcg_at_k(recommended_items, k) / dcg_max
    
    @staticmethod
    def map_at_k(recommended_items, relevant_items, k):
        """MAP@K (Mean Average Precision)"""
        if len(recommended_items) > k:
            recommended_items = recommended_items[:k]
        
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(recommended_items):
            if item in relevant_items and item not in recommended_items[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        if not relevant_items:
            return 0.0
        
        return score / min(len(relevant_items), k)
    
    def evaluate_model(self, model, test_data, k=10):
        """モデルの総合評価"""
        precisions = []
        recalls = []
        ndcgs = []
        maps = []
        
        for user_id in test_data['user_id'].unique():
            # テストデータから関連アイテムを取得
            relevant_items = test_data[
                test_data['user_id'] == user_id
            ]['item_id'].tolist()
            
            # 推薦を取得
            recommended_items = model.recommend(user_id, n_recommendations=k)
            
            # 各指標を計算
            precisions.append(self.precision_at_k(recommended_items, relevant_items, k))
            recalls.append(self.recall_at_k(recommended_items, relevant_items, k))
            ndcgs.append(self.ndcg_at_k(recommended_items, relevant_items, k))
            maps.append(self.map_at_k(recommended_items, relevant_items, k))
        
        return {
            'precision@k': np.mean(precisions),
            'recall@k': np.mean(recalls),
            'ndcg@k': np.mean(ndcgs),
            'map@k': np.mean(maps)
        }
```

## 実践的な実装

### リアルタイム推薦システム

```python
import redis
import pickle
from datetime import datetime, timedelta

class RealtimeRecommender:
    def __init__(self, model, redis_host='localhost', redis_port=6379):
        self.model = model
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.cache_ttl = 3600  # 1時間
    
    def get_recommendations(self, user_id, n_recommendations=10):
        """キャッシュを使った高速推薦"""
        # キャッシュキー
        cache_key = f"recommendations:{user_id}:{n_recommendations}"
        
        # キャッシュから取得
        cached = self.redis_client.get(cache_key)
        if cached:
            return pickle.loads(cached)
        
        # モデルで推薦を生成
        recommendations = self.model.recommend(user_id, n_recommendations)
        
        # キャッシュに保存
        self.redis_client.setex(
            cache_key, 
            self.cache_ttl, 
            pickle.dumps(recommendations)
        )
        
        return recommendations
    
    def update_user_interaction(self, user_id, item_id, interaction_type='view'):
        """ユーザーインタラクションの更新"""
        # インタラクション履歴の更新
        history_key = f"history:{user_id}"
        self.redis_client.lpush(history_key, f"{item_id}:{interaction_type}")
        self.redis_client.ltrim(history_key, 0, 99)  # 最新100件を保持
        
        # キャッシュの無効化
        pattern = f"recommendations:{user_id}:*"
        for key in self.redis_client.scan_iter(match=pattern):
            self.redis_client.delete(key)
    
    def get_trending_items(self, time_window_hours=24):
        """トレンドアイテムの取得"""
        # 時間窓内のインタラクションを集計
        now = datetime.now()
        start_time = now - timedelta(hours=time_window_hours)
        
        # Redisのソート済みセットを使用
        trending_key = f"trending:{start_time.strftime('%Y%m%d%H')}"
        
        trending_items = self.redis_client.zrevrange(
            trending_key, 0, 9, withscores=True
        )
        
        return [(item.decode(), score) for item, score in trending_items]

# A/Bテスト対応
class ABTestRecommender:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        self.assignment_cache = {}
    
    def get_model_assignment(self, user_id):
        """ユーザーのモデル割り当て"""
        if user_id not in self.assignment_cache:
            # 一貫性のある割り当て
            hash_value = hash(str(user_id)) % 100
            self.assignment_cache[user_id] = 'A' if hash_value < self.split_ratio * 100 else 'B'
        
        return self.assignment_cache[user_id]
    
    def recommend(self, user_id, n_recommendations=10):
        """A/Bテストを考慮した推薦"""
        assignment = self.get_model_assignment(user_id)
        
        if assignment == 'A':
            recommendations = self.model_a.recommend(user_id, n_recommendations)
            model_version = 'A'
        else:
            recommendations = self.model_b.recommend(user_id, n_recommendations)
            model_version = 'B'
        
        # メトリクスの記録
        self.log_recommendation(user_id, recommendations, model_version)
        
        return recommendations
    
    def log_recommendation(self, user_id, recommendations, model_version):
        """推薦ログの記録"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'model_version': model_version,
            'recommendations': recommendations
        }
        # ログをデータベースやファイルに保存
```

### 説明可能な推薦

```python
class ExplainableRecommender:
    def __init__(self, base_model):
        self.base_model = base_model
    
    def recommend_with_explanation(self, user_id, n_recommendations=10):
        """説明付き推薦"""
        recommendations = self.base_model.recommend(user_id, n_recommendations)
        explanations = []
        
        for item_id in recommendations:
            explanation = self.generate_explanation(user_id, item_id)
            explanations.append({
                'item_id': item_id,
                'explanation': explanation,
                'confidence': explanation['confidence']
            })
        
        return explanations
    
    def generate_explanation(self, user_id, item_id):
        """推薦理由の生成"""
        explanation = {
            'reasons': [],
            'confidence': 0.0
        }
        
        # 理由1: 類似ユーザーの評価
        similar_users = self.get_similar_users_who_liked(user_id, item_id)
        if similar_users:
            explanation['reasons'].append({
                'type': 'collaborative',
                'text': f"{len(similar_users)}人の似た嗜好のユーザーが高評価",
                'weight': 0.4
            })
            explanation['confidence'] += 0.4
        
        # 理由2: 過去の購買履歴
        similar_items = self.get_similar_items_from_history(user_id, item_id)
        if similar_items:
            explanation['reasons'].append({
                'type': 'content',
                'text': f"あなたが気に入った「{similar_items[0]['title']}」に似ています",
                'weight': 0.3
            })
            explanation['confidence'] += 0.3
        
        # 理由3: カテゴリの一致
        category_match = self.check_category_preference(user_id, item_id)
        if category_match:
            explanation['reasons'].append({
                'type': 'category',
                'text': f"よく見る{category_match}カテゴリの商品です",
                'weight': 0.2
            })
            explanation['confidence'] += 0.2
        
        # 理由4: トレンド
        if self.is_trending(item_id):
            explanation['reasons'].append({
                'type': 'trend',
                'text': "現在人気急上昇中",
                'weight': 0.1
            })
            explanation['confidence'] += 0.1
        
        return explanation
```

## まとめ

推薦システムの主要技術：

1. **協調フィルタリング**：ユーザーやアイテムの類似性を利用
2. **コンテンツベース**：アイテムの特徴を利用
3. **ハイブリッド手法**：複数手法の組み合わせ
4. **深層学習ベース**：NCF、Wide&Deep、DeepFM
5. **実践的な実装**：キャッシング、A/Bテスト、説明可能性

推薦システムは、ビジネスに直結する重要な技術であり、継続的な改善が求められます。

## 次へ

以上で、AI知識体系化リポジトリの全セクションが完成しました。