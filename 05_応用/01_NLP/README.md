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

RAG（Retrieval-Augmented Generation）は、大規模言語モデル（LLM）の知識制限を克服するための革新的な技術です。外部の知識ベースから関連情報を検索し、それをコンテキストとして生成に活用することで、より正確で最新の情報に基づいた回答を生成できます。

#### RAGの基本概念

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Any
import json

class RAGSystem:
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.metadata = []
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """ドキュメントをインデックスに追加"""
        self.documents.extend(documents)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{"id": i, "source": "unknown"} for i in range(len(documents))])
        
        # 埋め込みベクトルの生成
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        # FAISSインデックスの作成
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embeddings.astype('float32'))
    
    def retrieve(self, query: str, k: int = 3, threshold: float = 0.7) -> List[Dict]:
        """関連ドキュメントの検索"""
        query_embedding = self.embedding_model.encode([query])
        
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # 類似度スコアの計算（距離を類似度に変換）
            similarity = 1 / (1 + distance)
            
            if similarity >= threshold:
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "similarity": similarity,
                    "rank": i + 1
                })
        
        return results
    
    def generate_with_context(self, query: str, generator_model, 
                            max_context_length: int = 2000) -> Dict[str, Any]:
        """検索結果を使った生成"""
        # 関連ドキュメントの検索
        retrieved_docs = self.retrieve(query, k=5)
        
        if not retrieved_docs:
            return {
                "query": query,
                "context": "",
                "answer": "関連する情報が見つかりませんでした。",
                "sources": []
            }
        
        # コンテキストの作成（長さ制限付き）
        context_parts = []
        sources = []
        current_length = 0
        
        for doc in retrieved_docs:
            doc_text = f"参考情報{doc['rank']}: {doc['document']}"
            if current_length + len(doc_text) <= max_context_length:
                context_parts.append(doc_text)
                sources.append(doc['metadata'])
                current_length += len(doc_text)
            else:
                break
        
        context = "\n".join(context_parts)
        
        # プロンプトの作成
        prompt = f"""
以下の参考情報を基に、質問に正確に答えてください。
参考情報に含まれていない内容については「情報が不足しています」と答えてください。

参考情報:
{context}

質問: {query}

回答:
"""
        
        # 生成（実際のモデル呼び出しは省略）
        # response = generator_model.generate(prompt)
        
        return {
            "query": query,
            "context": context,
            "prompt": prompt,
            "sources": sources,
            "retrieved_docs": retrieved_docs
        }

# 高度なRAGシステム
class AdvancedRAGSystem(RAGSystem):
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__(embedding_model)
        self.chunk_size = 512
        self.chunk_overlap = 50
    
    def chunk_document(self, text: str) -> List[str]:
        """長いドキュメントをチャンクに分割"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def add_large_documents(self, documents: List[str], metadata: List[Dict] = None):
        """大きなドキュメントをチャンク分割して追加"""
        all_chunks = []
        all_metadata = []
        
        for i, doc in enumerate(documents):
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            
            if metadata and i < len(metadata):
                doc_metadata = metadata[i].copy()
                for j, chunk in enumerate(chunks):
                    chunk_metadata = doc_metadata.copy()
                    chunk_metadata["chunk_id"] = j
                    chunk_metadata["total_chunks"] = len(chunks)
                    all_metadata.append(chunk_metadata)
            else:
                for j, chunk in enumerate(chunks):
                    all_metadata.append({
                        "id": f"{i}_{j}",
                        "source": "unknown",
                        "chunk_id": j,
                        "total_chunks": len(chunks)
                    })
        
        self.add_documents(all_chunks, all_metadata)
    
    def hybrid_search(self, query: str, k: int = 3, 
                     dense_weight: float = 0.7) -> List[Dict]:
        """ハイブリッド検索（密ベクトル + スパースベクトル）"""
        # 密ベクトル検索（既存の実装）
        dense_results = self.retrieve(query, k=k*2)
        
        # スパースベクトル検索（BM25など）
        sparse_results = self.bm25_search(query, k=k*2)
        
        # 結果の統合
        combined_results = self.combine_results(
            dense_results, sparse_results, dense_weight
        )
        
        return combined_results[:k]
    
    def bm25_search(self, query: str, k: int = 3) -> List[Dict]:
        """BM25によるスパースベクトル検索"""
        # 簡易的なBM25実装
        from collections import Counter
        import math
        
        query_terms = query.lower().split()
        
        # ドキュメントのTF-IDF計算
        scores = []
        for i, doc in enumerate(self.documents):
            doc_terms = doc.lower().split()
            doc_term_freq = Counter(doc_terms)
            
            score = 0
            for term in query_terms:
                if term in doc_term_freq:
                    # 簡易BM25スコア
                    tf = doc_term_freq[term]
                    score += tf / (tf + 1.5)
            
            if score > 0:
                scores.append({
                    "document": doc,
                    "metadata": self.metadata[i],
                    "similarity": score,
                    "rank": 0
                })
        
        # スコアでソート
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:k]
    
    def combine_results(self, dense_results: List[Dict], 
                       sparse_results: List[Dict], 
                       dense_weight: float) -> List[Dict]:
        """検索結果の統合"""
        combined = {}
        
        # 密ベクトル結果の追加
        for result in dense_results:
            doc_id = result["metadata"].get("id", result["document"][:50])
            combined[doc_id] = {
                "document": result["document"],
                "metadata": result["metadata"],
                "dense_score": result["similarity"],
                "sparse_score": 0.0,
                "combined_score": result["similarity"] * dense_weight
            }
        
        # スパースベクトル結果の追加
        for result in sparse_results:
            doc_id = result["metadata"].get("id", result["document"][:50])
            if doc_id in combined:
                combined[doc_id]["sparse_score"] = result["similarity"]
                combined[doc_id]["combined_score"] += result["similarity"] * (1 - dense_weight)
            else:
                combined[doc_id] = {
                    "document": result["document"],
                    "metadata": result["metadata"],
                    "dense_score": 0.0,
                    "sparse_score": result["similarity"],
                    "combined_score": result["similarity"] * (1 - dense_weight)
                }
        
        # 統合スコアでソート
        sorted_results = sorted(
            combined.values(), 
            key=lambda x: x["combined_score"], 
            reverse=True
        )
        
        return sorted_results

# 使用例
def create_sample_rag_system():
    """サンプルRAGシステムの作成"""
    rag = AdvancedRAGSystem()
    
    # サンプルドキュメント
    documents = [
        "人工知能（AI）は、機械が人間の知能を模倣する技術です。機械学習、深層学習、自然言語処理などの分野を含みます。",
        "機械学習は、データから学習して予測や分類を行うAIの一分野です。教師あり学習、教師なし学習、強化学習に分類されます。",
        "深層学習は、多層のニューラルネットワークを使用した機械学習手法です。画像認識、音声認識、自然言語処理で高い性能を発揮します。",
        "自然言語処理（NLP）は、人間の言語をコンピュータが理解・生成する技術です。テキスト分類、機械翻訳、質問応答システムなどがあります。",
        "Transformerは、2017年に発表された自然言語処理の革新的なアーキテクチャです。Attention機構により長距離依存関係を効率的に学習できます。",
        "RAG（Retrieval-Augmented Generation）は、外部知識ベースから情報を検索し、それを基に回答を生成する技術です。LLMの知識制限を克服できます。"
    ]
    
    metadata = [
        {"id": "ai_intro", "source": "AI基礎", "category": "overview"},
        {"id": "ml_basics", "source": "機械学習", "category": "learning"},
        {"id": "dl_intro", "source": "深層学習", "category": "neural_networks"},
        {"id": "nlp_basics", "source": "NLP", "category": "language"},
        {"id": "transformer", "source": "アーキテクチャ", "category": "model"},
        {"id": "rag_intro", "source": "RAG", "category": "generation"}
    ]
    
    rag.add_documents(documents, metadata)
    return rag

# 実際の使用例
# rag_system = create_sample_rag_system()
# 
# # 質問応答
# query = "深層学習と機械学習の違いを教えて"
# result = rag_system.generate_with_context(query, None)
# 
# print("質問:", result["query"])
# print("検索されたドキュメント数:", len(result["retrieved_docs"]))
# print("コンテキスト:", result["context"][:200] + "...")
# print("プロンプト:", result["prompt"][:300] + "...")
```

#### RAGの応用例

```python
# ドキュメント検索システム
class DocumentSearchRAG:
    def __init__(self):
        self.rag = AdvancedRAGSystem()
        self.document_store = {}
    
    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        """ドキュメントの追加"""
        self.document_store[doc_id] = {
            "content": content,
            "metadata": metadata or {}
        }
        
        # チャンク分割してRAGシステムに追加
        chunks = self.rag.chunk_document(content)
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy() if metadata else {}
            chunk_meta.update({
                "doc_id": doc_id,
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
            chunk_metadata.append(chunk_meta)
        
        self.rag.add_documents(chunks, chunk_metadata)
    
    def search_and_answer(self, query: str, max_sources: int = 3) -> Dict:
        """検索と回答生成"""
        # ハイブリッド検索
        results = self.rag.hybrid_search(query, k=max_sources*2)
        
        # ソースの整理
        sources = []
        seen_docs = set()
        
        for result in results:
            doc_id = result["metadata"]["doc_id"]
            if doc_id not in seen_docs and len(sources) < max_sources:
                sources.append({
                    "doc_id": doc_id,
                    "content": result["document"],
                    "similarity": result["combined_score"],
                    "metadata": result["metadata"]
                })
                seen_docs.add(doc_id)
        
        # 回答生成
        context = "\n".join([f"参考{i+1}: {source['content']}" 
                           for i, source in enumerate(sources)])
        
        prompt = f"""
以下の参考情報を基に、質問に正確に答えてください。
参考情報に含まれていない内容については「情報が不足しています」と答えてください。

参考情報:
{context}

質問: {query}

回答:
"""
        
        return {
            "query": query,
            "answer_prompt": prompt,
            "sources": sources,
            "total_results": len(results)
        }

# チャットボット用RAG
class ChatbotRAG:
    def __init__(self):
        self.rag = AdvancedRAGSystem()
        self.conversation_history = []
    
    def add_knowledge_base(self, documents: List[str], metadata: List[Dict] = None):
        """知識ベースの追加"""
        self.rag.add_large_documents(documents, metadata)
    
    def chat(self, user_message: str, max_history: int = 5) -> Dict:
        """チャット応答"""
        # 会話履歴の管理
        self.conversation_history.append({"role": "user", "content": user_message})
        
        if len(self.conversation_history) > max_history * 2:
            # 古い履歴を削除（最新のmax_history分を保持）
            self.conversation_history = self.conversation_history[-max_history * 2:]
        
        # 検索クエリの作成（会話履歴を考慮）
        search_query = self._create_search_query(user_message)
        
        # RAG検索
        retrieved_docs = self.rag.retrieve(search_query, k=3)
        
        # 会話履歴と検索結果を組み合わせたプロンプト
        context = self._create_conversation_context(retrieved_docs)
        
        prompt = f"""
あなたは知識豊富なアシスタントです。以下の会話履歴と参考情報を基に、自然で有用な回答を提供してください。

{context}

最新の質問: {user_message}

回答:
"""
        
        # 回答生成（実際のモデル呼び出しは省略）
        # response = self._generate_response(prompt)
        
        # 会話履歴に回答を追加
        # self.conversation_history.append({"role": "assistant", "content": response})
        
        return {
            "user_message": user_message,
            "search_query": search_query,
            "retrieved_docs": retrieved_docs,
            "prompt": prompt,
            "conversation_length": len(self.conversation_history)
        }
    
    def _create_search_query(self, user_message: str) -> str:
        """検索クエリの作成"""
        # 会話履歴から重要な情報を抽出
        recent_context = ""
        for msg in self.conversation_history[-4:]:  # 最新2往復分
            if msg["role"] == "user":
                recent_context += f" {msg['content']}"
        
        # 現在の質問と履歴を組み合わせ
        search_query = f"{recent_context} {user_message}".strip()
        return search_query
    
    def _create_conversation_context(self, retrieved_docs: List[Dict]) -> str:
        """会話コンテキストの作成"""
        context_parts = []
        
        # 参考情報
        if retrieved_docs:
            context_parts.append("参考情報:")
            for i, doc in enumerate(retrieved_docs):
                context_parts.append(f"{i+1}. {doc['document']}")
        
        # 会話履歴
        if len(self.conversation_history) > 1:
            context_parts.append("\n会話履歴:")
            for msg in self.conversation_history[-6:]:  # 最新3往復分
                role = "ユーザー" if msg["role"] == "user" else "アシスタント"
                context_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_parts)
```

#### RAGの評価指標

```python
class RAGEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Dict], 
                          relevant_docs: List[str]) -> Dict[str, float]:
        """検索性能の評価"""
        retrieved_texts = [doc["document"] for doc in retrieved_docs]
        
        # Precision@K
        relevant_retrieved = sum(1 for doc in retrieved_texts 
                               if any(self._is_relevant(doc, rel) for rel in relevant_docs))
        precision_at_k = relevant_retrieved / len(retrieved_texts) if retrieved_texts else 0
        
        # Recall@K
        total_relevant = len(relevant_docs)
        recall_at_k = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        
        # F1@K
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) \
                  if (precision_at_k + recall_at_k) > 0 else 0
        
        return {
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "f1_at_k": f1_at_k,
            "relevant_retrieved": relevant_retrieved,
            "total_retrieved": len(retrieved_texts),
            "total_relevant": total_relevant
        }
    
    def evaluate_generation(self, generated_answer: str, 
                          reference_answer: str) -> Dict[str, float]:
        """生成性能の評価"""
        # BLEUスコア（簡易版）
        from collections import Counter
        
        def get_ngrams(text, n):
            words = text.split()
            return Counter([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])
        
        # 1-gramと2-gramのBLEU
        gen_1grams = get_ngrams(generated_answer, 1)
        gen_2grams = get_ngrams(generated_answer, 2)
        ref_1grams = get_ngrams(reference_answer, 1)
        ref_2grams = get_ngrams(reference_answer, 2)
        
        # 1-gram precision
        overlap_1gram = sum((gen_1grams & ref_1grams).values())
        precision_1gram = overlap_1gram / sum(gen_1grams.values()) if gen_1grams else 0
        
        # 2-gram precision
        overlap_2gram = sum((gen_2grams & ref_2grams).values())
        precision_2gram = overlap_2gram / sum(gen_2grams.values()) if gen_2grams else 0
        
        # 簡易BLEU
        bleu_score = (precision_1gram + precision_2gram) / 2
        
        return {
            "bleu_score": bleu_score,
            "precision_1gram": precision_1gram,
            "precision_2gram": precision_2gram,
            "generated_length": len(generated_answer.split()),
            "reference_length": len(reference_answer.split())
        }
    
    def _is_relevant(self, retrieved_doc: str, relevant_doc: str) -> bool:
        """関連性判定（簡易版）"""
        # 実際の実装では、より高度な類似度計算を使用
        retrieved_words = set(retrieved_doc.lower().split())
        relevant_words = set(relevant_doc.lower().split())
        
        overlap = len(retrieved_words & relevant_words)
        union = len(retrieved_words | relevant_words)
        
        return overlap / union > 0.3 if union > 0 else False

# 使用例
# evaluator = RAGEvaluator()
# 
# # 検索評価
# query = "深層学習について"
# retrieved_docs = [{"document": "深層学習は多層のニューラルネットワーク..."}]
# relevant_docs = ["深層学習は多層のニューラルネットワークを使用する技術"]
# 
# retrieval_metrics = evaluator.evaluate_retrieval(query, retrieved_docs, relevant_docs)
# print("検索評価:", retrieval_metrics)
# 
# # 生成評価
# generated = "深層学習は多層のニューラルネットワークを使用した機械学習手法です。"
# reference = "深層学習は多層のニューラルネットワークを使った学習方法です。"
# 
# generation_metrics = evaluator.evaluate_generation(generated, reference)
# print("生成評価:", generation_metrics)
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

[画像生成](../02_画像生成/README.md)へ進む