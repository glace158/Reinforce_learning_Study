import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import gym
import numpy as np
import random as rand

class Agent(object):
    def __init__(self) -> None:
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.node_num = 12#인공 신경망 레이어에 들어있는 노드의 개수
        self.learning_rate = 0.001
        self.epochs_cnt = 5
        self.model = self.build_model()
        
        self.discount_rate = 0.97
        self.penalty = -100
        
        self.episode_num = 500
        
        self.replay_memory_limit = 2048
        self.replay_size = 32
        self.replay_memory = []
        
        self.epsilon = 0.99
        self.epsilon_decay = 0.2
        self.epsilon_min = 0.05
        
        self.moving_avg_size = 20 #몇개의 에피소드에 대한 보상으로 평균을 구하는 개수
        self.reward_list = [] # 에피소드에서 받은 보상의 핪을 저장
        self.count_list = [] # 각 에피소드에서 카트폴이 실행된 횟수를 기록
        self.moving_avg_list = [] # 가장 최근 실행된 에피소드를 기준으로 이전 moving_avg_size만큼 이동 평균을 구한 카트폴의 실행 횟수
        
    def build_model(self):
        input_states = Input(shape=(1, self.state_size), name='Input_states') #input
        x = (input_states)
        x = Dense(self.node_num, activation='relu')(x)# layer
        out_actions = Dense(self.action_size, activation='linear', name='output')(x) # output
        model = tf.keras.models.Model(inputs=[input_states], outputs=[out_actions]) # 모델 구성
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error') #학습환경 설정 
        model.summary() 
        return model
    
    def train(self):
        for episode in range(self.episode_num):
            state = self.env.reset()[0]

            Q, count, reward_tot = self.take_action_and_append_memory(episode, state)# replay_memory 변수에 저장
            
            if count < 500:
                reward_tot = reward_tot - self.penalty
                
            self.reward_list.append(reward_tot - self.penalty)# 보상 저장
            self.count_list.append(count)# 실행 횟수 저장
            self.moving_avg_list.append(self.moving_avg(self.count_list, self.moving_avg_size)) # 이동 평균 구하기
            
            self.train_mini_batch(Q) 
            
            if(episode % 10 == 0):
                print("episode: {}, moving_avg: {}, rewards_avg {}".format(episode, self.moving_avg_list[-1], np.mean(self.reward_list)))
        
        self.save_model()
        
    def take_action_and_append_memory(self, episode, state):
        reward_tot = 0
        count = 0
        done = False
        epsilon = self.get_epsilon(episode)
        while not done:
            count += 1
            state_t = np.reshape(state, [1,1,self.state_size])# 배치로 학습하기 때문에 모델에 입력되는 데이터는 실질적으로 (n,1,4)가 되어 한 건의 입력 데이터를 사용하려면 (1,1,4)로 변경 필요

            Q = self.model.predict(state_t)# Q값 예측
            action = self.greed_search(epsilon, episode, Q) # 행동 선택
            state_next, reward, done, none,none = self.env.step(action) # 선택한 행동 수행
            
            if done:
                reward - self.penalty # 막대가 떨어진 경우 페널티 
                
            self.replay_memory.append([state_t, action, reward, state_next, done]) 
            
            if len(self.replay_memory) > self.replay_memory_limit:
                del self.replay_memory[0]
                
            reward_tot += reward
            state = state_next
        
        return Q, count, reward_tot

    def train_mini_batch(self, Q):
        array_state = []
        array_Q = []
        this_replay_size = self.replay_size
        if len(self.replay_memory) < self.replay_size:
            this_replay_size = len(self.replay_memory)
            
        for sample in rand.sample(self.replay_memory, this_replay_size):
            state_t, action, reward, state_next, done = sample
            
            # Q 값 계산
            if done:
                Q[0,0,action] = reward 
            else:
                state_t = np.reshape(state_next, [1,1,self.state_size])
                Q_new = self.model.predict(state_t) # 다음 상태 Q 값
                Q[0,0,action] = reward + self.discount_rate * np.max(Q_new) # R + r max(Q')
            
            # Q값을 사용 가능한 형태로 변경
            array_state.append(state_t.reshape(1,self.state_size)) 
            array_Q.append(Q.reshape(1, self.action_size))
        
        # numpy 형식으로 변형
        array_state_t = np.array(array_state)
        array_Q_t = np.array(array_Q)
        
        # 수집된 데이터를 입력해서 학습 진행
        hist = self.model.fit(array_state_t, array_Q_t, epochs=self.epochs_cnt, verbose=0)
    
    def get_epsilon(self, episode):
        result = self.epsilon * (1 - episode / (self.episode_num * self.epsilon_decay))
        if result < self.epsilon_min:
            result = self.epsilon_min
        
        return result
    
    def greed_search(self, epsilon, episode, Q):
        if epsilon > np.random.rand(1):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(Q) 
        
        return action
    
    def moving_avg(self, data, size= 10):
        if len(data) > size:
            c = np.array(data[len(data) - size:len(data)]) # size 크기 만큼 자름
        else:
            c = np.array(data)
        
        return np.mean(c)
    
    def save_model(self):
        self.model.save("./model/dqn")
        print("*****end learning")
    
        
if __name__ == '__main__':
    agent = Agent()
    agent.train()
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(agent.reward_list, label='rewards')
    plt.plot(agent.moving_avg_list, linewidth=4, label='moving average')
    plt.legend(loc='upper left')
    plt.title('DQN')
    plt.show()