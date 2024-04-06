import gym
import numpy as np
from replay_buffer import ReplayBuffer

env = gym.make('CartPole-v1', render_mode= 'human')
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

for episode in range(10):
    state = env.reset()[0]
    done = False
    
    while not done:
        action = 0
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
    
state, action, reward, next_state, done = replay_buffer.get_batch()

print(state.shape)
print(action.shape)
print(reward.shape)
print(next_state.shape)
print(done.shape)
