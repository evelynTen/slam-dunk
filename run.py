#!/usr/bin/env python
# coding: utf-8

# In[19]:


import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random


# In[2]:


env = gym.make('BasketBall-v0', s=7.865, h=1.8)


# In[17]:


print(env.observation_space.sample())


# In[18]:


episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        # env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))


# In[20]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


# In[21]:


states = env.observation_space.shape
actions = env.action_space.n


# In[22]:


def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


# In[28]:


del model 


# In[30]:


model = build_model(states, actions)


# In[31]:


model.summary()


# In[25]:


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


# In[32]:


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


# In[33]:


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)


# In[34]:


scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))


# In[ ]:





# In[ ]:





# In[ ]:




