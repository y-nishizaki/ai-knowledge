# 自然言語処理（NLP）

## 概要

自然言語処理（Natural Language Processing, NLP）は、人間の言語をコンピュータに理解・生成させる技術です。テキスト分類から機械翻訳、対話システムまで、幅広い応用があります。

## 基礎技術

### テキストの前処理

```python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import MeCab  # 日本語処理用

# 英語の前処理
class EnglishTextPreprocessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text):
        # 小文字化
        text = text.lower()
        
        # 特殊文字の除去
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # トークン化
        tokens = word_tokenize(text)
        
        # ストップワード除去とレンマ化
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words]
        
        return tokens

# 日本語の前処理
class JapaneseTextPreprocessor:
    def __init__(self):
        self.mecab = MeCab.Tagger()
        
    def preprocess(self, text):
        # 全角スペースを半角に
        text = text.replace('　', ' ')
        
        # MeCabで形態素解析
        parsed = self.mecab.parse(text)
        
        tokens = []
        for line in parsed.split('\n'):
            if line == 'EOS' or line == '':
                break
            
            surface, features = line.split('\t')
            pos = features.split(',')[0]
            
            # 名詞、動詞、形容詞のみ抽出
            if pos in ['名詞', '動詞', '形容詞']:
                # 原形を取得
                base_form = features.split(',')[6]
                if base_form != '*':
                    tokens.append(base_form)
                else:
                    tokens.append(surface)
        
        return tokens

# 使用例
eng_processor = EnglishTextPreprocessor()
jp_processor = JapaneseTextPreprocessor()

eng_text = "Natural language processing is fascinating!"
jp_text = "自然言語処理は素晴らしい技術です。"

print("英語:", eng_processor.preprocess(eng_text))
print("日本語:", jp_processor.preprocess(jp_text))
```

### 単語埋め込み（Word Embeddings）

```python
import numpy as np
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Word2Vecの学習
sentences = [
    ['機械', '学習', 'は', '面白い'],
    ['深層', '学習', 'は', '強力', 'だ'],
    ['自然', '言語', '処理', 'は', '重要'],
    ['AI', 'は', '未来', 'の', '技術']
]

# Word2Vecモデルの学習
w2v_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=100
)

# 類似単語の検索
similar_words = w2v_model.wv.most_similar('学習', topn=3)
print("'学習'に類似した単語:")
for word, score in similar_words:
    print(f"  {word}: {score:.3f}")

# 単語ベクトルの可視化
def visualize_word_vectors(model, words):
    word_vectors = np.array([model.wv[word] for word in words if word in model.wv])
    word_labels = [word for word in words if word in model.wv]
    
    # PCAで2次元に削減
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(word_vectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
    
    for i, word in enumerate(word_labels):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))
    
    plt.title('Word Vectors Visualization')
    plt.show()

# FastTextの使用（サブワード情報を含む）
fasttext_model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=2,
    max_n=5,
    epochs=100
)

# 未知語への対応
print("\n未知語のベクトル生成:")
unknown_word = "深層学習"  # 訓練データにない組み合わせ
if unknown_word in fasttext_model.wv:
    print(f"'{unknown_word}'のベクトルが生成されました")
```

## テキスト分類

### 感情分析

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import pandas as pd

# データの準備（例）
texts = [
    "この映画は素晴らしい！感動しました。",
    "つまらない内容でがっかりした。",
    "普通の映画だった。",
    "最高の体験でした！",
    "時間の無駄だった。"
]
labels = [1, 0, 0.5, 1, 0]  # 1: ポジティブ, 0: ネガティブ

# テキストの数値化
max_words = 10000
max_length = 100

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding='post')

# BiLSTMモデル
def create_sentiment_model(vocab_size, embedding_dim=128):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# モデルの作成
model = create_sentiment_model(max_words)
model.summary()

# BERTを使った感情分析
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

class BertSentimentAnalyzer:
    def __init__(self, model_name='bert-base-multilingual-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3  # ポジティブ、ネガティブ、ニュートラル
        )
    
    def preprocess(self, texts, max_length=128):
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
    
    def predict(self, texts):
        inputs = self.preprocess(texts)
        outputs = self.model(inputs)
        predictions = tf.nn.softmax(outputs.logits, axis=-1)
        return predictions.numpy()

# 使用例
# bert_analyzer = BertSentimentAnalyzer()
# predictions = bert_analyzer.predict(["素晴らしい映画でした！"])
```

### 多クラステキスト分類

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ニュースカテゴリ分類の例
news_texts = [
    "株価が大幅に上昇した",
    "新しいスマートフォンが発売",
    "サッカーワールドカップ開催",
    "新型ウイルスの感染拡大",
    "映画祭で日本作品が受賞"
]
categories = ["経済", "技術", "スポーツ", "健康", "エンタメ"]

# TF-IDFベクトル化
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(news_texts)

# 従来の機械学習手法
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)

# トランスフォーマーベースの分類
from transformers import pipeline

# 事前学習済みモデルを使った分類
classifier = pipeline(
    "text-classification",
    model="bert-base-japanese",
    device=0 if torch.cuda.is_available() else -1
)

# ゼロショット分類
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

def classify_text(text, candidate_labels):
    result = zero_shot_classifier(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template="This text is about {}."
    )
    return result

# 使用例
text = "AIが囲碁で人間に勝利"
labels = ["スポーツ", "技術", "ゲーム", "科学"]
# result = classify_text(text, labels)
```

## 系列変換タスク

### 機械翻訳

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np

class Seq2SeqTranslator:
    def __init__(self, input_vocab_size, target_vocab_size, latent_dim=256):
        self.latent_dim = latent_dim
        
        # エンコーダ
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(input_vocab_size, latent_dim)(encoder_inputs)
        encoder_lstm = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        # デコーダ
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(target_vocab_size, latent_dim)
        decoder_embedding_outputs = decoder_embedding(decoder_inputs)
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(
            decoder_embedding_outputs, 
            initial_state=encoder_states
        )
        decoder_dense = Dense(target_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # モデル
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 推論用モデル
        self.encoder_model = Model(encoder_inputs, encoder_states)
        
        # デコーダの推論モデル
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_embedding_inf = decoder_embedding(decoder_inputs)
        decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
            decoder_embedding_inf, 
            initial_state=decoder_states_inputs
        )
        decoder_states_inf = [state_h_inf, state_c_inf]
        decoder_outputs_inf = decoder_dense(decoder_outputs_inf)
        
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs_inf] + decoder_states_inf
        )
    
    def translate(self, input_seq, input_token_index, target_token_index, max_length=50):
        # エンコード
        states_value = self.encoder_model.predict(input_seq)
        
        # デコード開始
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = target_token_index['<start>']
        
        decoded_sentence = []
        
        while True:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value
            )
            
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            
            if sampled_token_index == target_token_index['<end>'] or len(decoded_sentence) > max_length:
                break
            
            decoded_sentence.append(sampled_token_index)
            
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            
            states_value = [h, c]
        
        return decoded_sentence

# Attention機構付き翻訳モデル
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
    
    def call(self, query, values):
        # query: (batch_size, hidden_size)
        # values: (batch_size, seq_len, hidden_size)
        
        query_with_time_axis = tf.expand_dims(query, 1)
        
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        ))
        
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
```

### 要約生成

```python
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

class TextSummarizer:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = TFT5ForConditionalGeneration.from_pretrained(model_name)
    
    def summarize(self, text, max_length=150, min_length=30):
        # T5は"summarize: "プレフィックスを使用
        input_text = f"summarize: {text}"
        
        # トークン化
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors='tf',
            max_length=512,
            truncation=True
        )
        
        # 要約生成
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # デコード
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

# 抽出型要約
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def extractive_summarization(text, num_sentences=3):
    """TextRankアルゴリズムによる抽出型要約"""
    # 文に分割
    sentences = text.split('。')
    sentences = [s.strip() + '。' for s in sentences if s.strip()]
    
    if len(sentences) <= num_sentences:
        return ''.join(sentences)
    
    # TF-IDFベクトル化
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    
    # 類似度行列の計算
    similarity_matrix = cosine_similarity(sentence_vectors)
    
    # グラフの構築
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # スコアの高い文を選択
    ranked_sentences = sorted(
        ((scores[i], i) for i in range(len(sentences))), 
        reverse=True
    )
    
    # 元の順序を保持して要約を作成
    selected_indices = sorted([idx for _, idx in ranked_sentences[:num_sentences]])
    summary = ''.join([sentences[i] for i in selected_indices])
    
    return summary
```

## 質問応答システム

```python
from transformers import pipeline, AutoTokenizer, TFAutoModelForQuestionAnswering

class QuestionAnsweringSystem:
    def __init__(self, model_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
        self.qa_pipeline = pipeline(
            'question-answering',
            model=self.model,
            tokenizer=self.tokenizer
        )
    
    def answer(self, question, context):
        result = self.qa_pipeline({
            'question': question,
            'context': context
        })
        return result
    
    def answer_with_confidence(self, question, context, top_k=3):
        inputs = self.tokenizer(
            question,
            context,
            return_tensors='tf',
            max_length=512,
            truncation=True
        )
        
        outputs = self.model(inputs)
        
        # 開始位置と終了位置の確率
        start_scores = tf.nn.softmax(outputs.start_logits, axis=-1)[0]
        end_scores = tf.nn.softmax(outputs.end_logits, axis=-1)[0]
        
        # トップk個の回答候補を取得
        start_indices = tf.argsort(start_scores, direction='DESCENDING')[:top_k]
        end_indices = tf.argsort(end_scores, direction='DESCENDING')[:top_k]
        
        candidates = []
        for start_idx in start_indices:
            for end_idx in end_indices:
                if start_idx <= end_idx:
                    score = start_scores[start_idx] * end_scores[end_idx]
                    answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
                    answer = self.tokenizer.decode(answer_tokens)
                    candidates.append({
                        'answer': answer,
                        'confidence': float(score),
                        'start': int(start_idx),
                        'end': int(end_idx)
                    })
        
        # スコアでソート
        candidates = sorted(candidates, key=lambda x: x['confidence'], reverse=True)
        return candidates[:top_k]

# 使用例
qa_system = QuestionAnsweringSystem()
context = """
人工知能（AI）は、機械が人間の知能を模倣する技術です。
機械学習は、データから学習するAIの一分野です。
深層学習は、多層のニューラルネットワークを使用する機械学習の手法です。
"""
question = "深層学習とは何ですか？"

# answer = qa_system.answer(question, context)
# print(f"質問: {question}")
# print(f"回答: {answer['answer']}")
# print(f"信頼度: {answer['score']:.3f}")
```

## 大規模言語モデル（LLM）

### GPTスタイルモデルの実装

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout

class MultiHeadSelfAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # Q, K, V の計算
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # ヘッドに分割
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Attention計算
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # マスク適用（因果的マスク）
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        # ヘッドを結合
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, seq_len, self.d_model))
        
        output = self.dense(output)
        return output

class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, x, training, mask=None):
        attn_output = self.attention(x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class GPTModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, dff, num_layers,
                 max_position_embeddings=1024, rate=0.1):
        super(GPTModel, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(max_position_embeddings, d_model)
        
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, rate) 
            for _ in range(num_layers)
        ]
        
        self.dropout = Dropout(rate)
        self.final_layer = Dense(vocab_size)
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    
    def call(self, x, training):
        seq_len = tf.shape(x)[1]
        
        # マスクの作成
        mask = self.create_look_ahead_mask(seq_len)
        
        # 埋め込みと位置エンコーディング
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        # Transformerブロック
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training, mask)
        
        # 最終層
        final_output = self.final_layer(x)
        
        return final_output

# テキスト生成
def generate_text(model, tokenizer, seed_text, max_length=100, temperature=1.0):
    for _ in range(max_length):
        tokenized = tokenizer.encode(seed_text, return_tensors='tf')
        
        predictions = model(tokenized, training=False)
        predictions = predictions[0, -1, :] / temperature
        
        predicted_id = tf.random.categorical(predictions[tf.newaxis, :], num_samples=1)
        predicted_id = tf.squeeze(predicted_id, axis=[0, 1]).numpy()
        
        seed_text += tokenizer.decode([predicted_id])
        
        if predicted_id == tokenizer.eos_token_id:
            break
    
    return seed_text
```

### プロンプトエンジニアリング

```python
class PromptTemplate:
    def __init__(self):
        self.templates = {
            'classification': """
タスク: 以下のテキストを指定されたカテゴリのいずれかに分類してください。

テキスト: {text}
カテゴリ: {categories}

分類結果:
""",
            'summary': """
以下の文章を{length}文字程度で要約してください。

文章:
{text}

要約:
""",
            'translation': """
以下の{source_lang}の文章を{target_lang}に翻訳してください。

原文: {text}

翻訳:
""",
            'qa': """
以下の文脈に基づいて質問に答えてください。

文脈: {context}

質問: {question}

回答:
""",
            'generation': """
以下のトピックについて、{style}な文章を{length}文字程度で生成してください。

トピック: {topic}

生成文:
"""
        }
    
    def get_prompt(self, task, **kwargs):
        if task not in self.templates:
            raise ValueError(f"Unknown task: {task}")
        
        return self.templates[task].format(**kwargs)

# Few-shot Learning
class FewShotPrompt:
    def __init__(self):
        self.examples = []
    
    def add_example(self, input_text, output_text):
        self.examples.append({
            'input': input_text,
            'output': output_text
        })
    
    def create_prompt(self, task_description, query):
        prompt = f"{task_description}\n\n"
        
        for i, example in enumerate(self.examples):
            prompt += f"例{i+1}:\n"
            prompt += f"入力: {example['input']}\n"
            prompt += f"出力: {example['output']}\n\n"
        
        prompt += f"入力: {query}\n出力: "
        
        return prompt

# Chain-of-Thought プロンプティング
def create_cot_prompt(question):
    return f"""
質問: {question}

この問題を段階的に考えてみましょう。

ステップ1: 問題の理解
まず、何を求められているか整理します。

ステップ2: 必要な情報の特定
問題を解くために必要な情報を特定します。

ステップ3: 解法の検討
どのようなアプローチで解くか考えます。

ステップ4: 計算・推論
実際に計算や推論を行います。

ステップ5: 答えの確認
答えが妥当か確認します。

最終的な答え:
"""
```

## 最新技術とトレンド

### RAG（Retrieval-Augmented Generation）

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGSystem:
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
    
    def add_documents(self, documents):
        """ドキュメントをインデックスに追加"""
        self.documents.extend(documents)
        
        # 埋め込みベクトルの生成
        embeddings = self.embedding_model.encode(documents)
        
        # FAISSインデックスの作成
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embeddings.astype('float32'))
    
    def retrieve(self, query, k=3):
        """関連ドキュメントの検索"""
        query_embedding = self.embedding_model.encode([query])
        
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )
        
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        return retrieved_docs
    
    def generate_with_context(self, query, generator_model):
        """検索結果を使った生成"""
        # 関連ドキュメントの検索
        context_docs = self.retrieve(query)
        
        # コンテキストの作成
        context = "\n".join([f"参考情報{i+1}: {doc}" 
                           for i, doc in enumerate(context_docs)])
        
        # プロンプトの作成
        prompt = f"""
以下の参考情報を基に、質問に答えてください。

{context}

質問: {query}

回答:
"""
        
        # 生成（実際のモデル呼び出しは省略）
        # response = generator_model.generate(prompt)
        
        return prompt  # 実際はresponseを返す

# 使用例
rag = RAGSystem()
rag.add_documents([
    "人工知能は機械が人間の知能を模倣する技術です。",
    "機械学習はデータから学習するAIの一分野です。",
    "深層学習は多層のニューラルネットワークを使用します。"
])

# query = "深層学習について教えて"
# context = rag.generate_with_context(query, None)
```

## まとめ

NLPの主要技術：

1. **基礎技術**：前処理、単語埋め込み、トークン化
2. **タスク別手法**：分類、系列変換、生成
3. **最新モデル**：Transformer、BERT、GPT
4. **応用技術**：RAG、プロンプトエンジニアリング
5. **実用化**：チャットボット、翻訳、要約

NLPは急速に進化している分野で、大規模言語モデルにより多くのタスクが統一的に解けるようになっています。

## 次へ

[画像生成](../画像生成/README.md)へ進む