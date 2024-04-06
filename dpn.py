import copy
import gym
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
import replay_buffer

class QNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)
    
    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class DQNAgent:
    def __init__(self) -> None:
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size =10000
        
        self.batch_size = 32
        self.action_size = 2
        self.replay_buffer = replay_buffer.ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)#원본 신경망
        self.qnet_target = QNet(self.action_size)# 목표 신경망
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)# 옵티마이저에 qnet 등록
        
    def sync_qnet(self): # 두신경망 동기화
        self.qnet_target = copy.deepcopy(self.qnet)
        
    def get_action(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]
        
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis= 1)
        next_q.unchain()
        
        target = reward + (1 - done) * self.gamma * next_q
        
        loss = F.mean_squared_error(q, target)
        
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()
        
if __name__ == "__main__":
    episodes = 300
    sync_interval = 20
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    agent = DQNAgent()
    reward_history=[]
    
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        if episode % sync_interval == 0:
            agent.sync_qnet()
        reward_history.append(total_reward)