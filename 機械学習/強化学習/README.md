# 強化学習

## 概要

強化学習（Reinforcement Learning, RL）は、エージェントが環境との相互作用を通じて、報酬を最大化する行動を学習する手法です。試行錯誤を通じて最適な方策（ポリシー）を見つけ出します。

## 基本概念

### 強化学習の要素

1. **エージェント（Agent）**：学習し行動する主体
2. **環境（Environment）**：エージェントが相互作用する世界
3. **状態（State）**：環境の現在の状況
4. **行動（Action）**：エージェントが取れる選択肢
5. **報酬（Reward）**：行動の結果として得られる評価
6. **方策（Policy）**：状態から行動への写像

### 強化学習のプロセス

```
エージェント → 行動 → 環境
    ↑                    ↓
    報酬 ← 新しい状態 ←
```

## 主要なアルゴリズム

### 1. Q学習（Q-Learning）

価値ベースの手法で、状態-行動ペアの価値（Q値）を学習します。

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# 簡単な迷路環境の実装
class SimpleMaze:
    def __init__(self):
        # 0: 通路, 1: 壁, 2: ゴール
        self.maze = np.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 0, 0]
        ])
        self.start_pos = (0, 0)
        self.goal_pos = (2, 3)
        self.current_pos = self.start_pos
        
    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action):
        # 行動: 0=上, 1=右, 2=下, 3=左
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_pos = (self.current_pos[0] + moves[action][0],
                   self.current_pos[1] + moves[action][1])
        
        # 境界チェック
        if (0 <= new_pos[0] < 4 and 0 <= new_pos[1] < 4 and
            self.maze[new_pos] != 1):
            self.current_pos = new_pos
        
        # 報酬の計算
        if self.current_pos == self.goal_pos:
            reward = 100
            done = True
        else:
            reward = -1
            done = False
            
        return self.current_pos, reward, done
    
    def render(self):
        display = self.maze.copy()
        display[self.current_pos] = 3  # エージェントの位置
        
        cmap = colors.ListedColormap(['white', 'black', 'gold', 'red'])
        plt.imshow(display, cmap=cmap)
        plt.grid(True)
        plt.show()

# Q学習の実装
class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, 
                 discount_factor=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(4)  # ランダム行動
        else:
            return np.argmax(self.q_table[state])  # 貪欲行動
    
    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q

# 学習の実行
env = SimpleMaze()
agent = QLearningAgent(16, 4)  # 4x4の迷路なので16状態

# 状態を1次元のインデックスに変換
def state_to_index(state):
    return state[0] * 4 + state[1]

# 学習ループ
episodes = 1000
rewards_history = []

for episode in range(episodes):
    state = env.reset()
    state_idx = state_to_index(state)
    total_reward = 0
    
    while True:
        action = agent.choose_action(state_idx)
        next_state, reward, done = env.step(action)
        next_state_idx = state_to_index(next_state)
        
        agent.update(state_idx, action, reward, next_state_idx)
        
        state_idx = next_state_idx
        total_reward += reward
        
        if done:
            break
    
    rewards_history.append(total_reward)

# 学習結果の可視化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(rewards_history)
plt.xlabel('エピソード')
plt.ylabel('合計報酬')
plt.title('学習の進捗')

plt.subplot(1, 2, 2)
# Q値をヒートマップで表示
q_values_grid = agent.q_table.reshape(4, 4, 4)
best_actions = np.argmax(q_values_grid, axis=2)
action_symbols = ['↑', '→', '↓', '←']

plt.imshow(np.zeros((4, 4)), cmap='gray', alpha=0.1)
for i in range(4):
    for j in range(4):
        if env.maze[i, j] == 0:  # 通路の場合
            plt.text(j, i, action_symbols[best_actions[i, j]], 
                    ha='center', va='center', fontsize=20)
        elif env.maze[i, j] == 2:  # ゴール
            plt.text(j, i, 'G', ha='center', va='center', 
                    fontsize=20, color='gold')
plt.title('学習した最適方策')
plt.grid(True)
plt.show()
```

### 2. SARSA（State-Action-Reward-State-Action）

Q学習と似ていますが、実際に選択された次の行動を使って更新します。

```python
class SARSAAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, 
                 discount_factor=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        new_q = current_q + self.lr * (reward + self.gamma * next_q - current_q)
        self.q_table[state, action] = new_q

# SARSAの学習
sarsa_agent = SARSAAgent(16, 4)
sarsa_rewards = []

for episode in range(episodes):
    state = env.reset()
    state_idx = state_to_index(state)
    action = sarsa_agent.choose_action(state_idx)
    total_reward = 0
    
    while True:
        next_state, reward, done = env.step(action)
        next_state_idx = state_to_index(next_state)
        next_action = sarsa_agent.choose_action(next_state_idx)
        
        sarsa_agent.update(state_idx, action, reward, next_state_idx, next_action)
        
        state_idx = next_state_idx
        action = next_action
        total_reward += reward
        
        if done:
            break
    
    sarsa_rewards.append(total_reward)

# Q学習とSARSAの比較
plt.figure(figsize=(10, 6))
plt.plot(rewards_history, label='Q-Learning', alpha=0.7)
plt.plot(sarsa_rewards, label='SARSA', alpha=0.7)
plt.xlabel('エピソード')
plt.ylabel('合計報酬')
plt.title('Q学習とSARSAの比較')
plt.legend()
plt.show()
```

### 3. Deep Q-Network (DQN)

ニューラルネットワークを使ってQ関数を近似します。

```python
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.model = self._build_model()
        
    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# CartPole環境でのDQN実装例
import gym

# 環境の作成
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# DQNエージェントの作成
dqn_agent = DQNAgent(state_size, action_size)

# 学習
episodes = 100
scores = []

for episode in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = np.reshape(state, [1, state_size])
    
    score = 0
    for time in range(500):
        action = dqn_agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        
        next_state = np.reshape(next_state, [1, state_size])
        dqn_agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        score += reward
        
        if done:
            print(f"エピソード: {episode+1}/{episodes}, スコア: {score}, ε: {dqn_agent.epsilon:.2f}")
            break
        
        if len(dqn_agent.memory) > 32:
            dqn_agent.replay()
    
    scores.append(score)

# 学習結果の可視化
plt.figure(figsize=(10, 6))
plt.plot(scores)
plt.xlabel('エピソード')
plt.ylabel('スコア')
plt.title('DQNの学習進捗（CartPole）')
plt.show()
```

### 4. Policy Gradient手法

直接方策を最適化する手法です。

```python
class PolicyGradientAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.states = []
        self.actions = []
        self.rewards = []
        
    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', 
                     optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        prob_distribution = self.model.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=prob_distribution)
        return action
    
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * 0.99 + rewards[t]
            discounted_rewards[t] = running_add
        
        # 正規化
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards) + 1e-7
        
        return discounted_rewards
    
    def train(self):
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        
        discounted_rewards = self.discount_rewards(rewards)
        
        # one-hot エンコーディング
        actions_onehot = keras.utils.to_categorical(actions, self.action_size)
        
        # 勾配の重み付け
        advantages = discounted_rewards
        
        self.model.fit(states, actions_onehot, 
                      sample_weight=advantages,
                      epochs=1, verbose=0)
        
        # メモリをクリア
        self.states = []
        self.actions = []
        self.rewards = []
```

## 強化学習の応用例

### 1. ゲームAI

チェス、囲碁、ビデオゲームなどで人間を超える性能を達成しています。

### 2. ロボット制御

```python
# 簡単なロボットアームの制御例
class RobotArmEnv:
    def __init__(self):
        self.target = np.array([0.5, 0.5])
        self.arm_length = 0.3
        self.angles = np.array([0.0, 0.0])  # 2関節
        
    def reset(self):
        self.angles = np.array([0.0, 0.0])
        return self.get_state()
    
    def get_state(self):
        # エンドエフェクタの位置と目標との差
        end_pos = self.forward_kinematics(self.angles)
        return np.concatenate([self.angles, end_pos - self.target])
    
    def forward_kinematics(self, angles):
        x = self.arm_length * (np.cos(angles[0]) + np.cos(angles[0] + angles[1]))
        y = self.arm_length * (np.sin(angles[0]) + np.sin(angles[0] + angles[1]))
        return np.array([x, y])
    
    def step(self, action):
        # 各関節の角度を更新
        self.angles += action * 0.1
        self.angles = np.clip(self.angles, -np.pi, np.pi)
        
        # 報酬の計算
        end_pos = self.forward_kinematics(self.angles)
        distance = np.linalg.norm(end_pos - self.target)
        reward = -distance
        
        done = distance < 0.05
        
        return self.get_state(), reward, done
```

### 3. 自動運転

交通ルールの遵守と安全な運転を学習します。

### 4. リソース管理

データセンターの電力管理、在庫管理などの最適化に使用されます。

## 強化学習の課題と最新動向

### 課題
1. **サンプル効率**：多くの試行が必要
2. **報酬設計**：適切な報酬関数の設計が困難
3. **安全性**：学習中の危険な行動
4. **汎化性能**：新しい環境への適応

### 最新動向
1. **オフライン強化学習**：事前収集データから学習
2. **マルチエージェント強化学習**：複数エージェントの協調
3. **メタ強化学習**：学習の仕方を学習
4. **人間のフィードバックからの強化学習（RLHF）**：ChatGPTなどで使用

## まとめ

強化学習は、試行錯誤を通じて最適な行動を学習する強力な手法です。ゲーム、ロボティクス、最適化問題など、幅広い分野で応用されています。深層学習との組み合わせにより、さらに複雑な問題にも対応できるようになっています。

## 次へ

[ディープラーニング](../../ディープラーニング/README.md)へ進む