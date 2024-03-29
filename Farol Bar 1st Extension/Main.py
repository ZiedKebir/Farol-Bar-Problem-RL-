import os
os.chdir("C:/Users/ziedk/OneDrive/Bureau/Strasbourg/Master data Strasbourg/RL Muller/Farol Bar 2")

import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPG import *
from Trainer import *
from models import * 

env = NormalizedEnv()


agent = DDPGagent()
#noise = OUNoise(env.action_space)


batch_size = 128
rewards = []
actions=[]
avg_rewards = []
epsilon = 0.7

for episode in range(0,200):
    #state = env.reset()
    #noise.reset()
    episode_reward = 0
    state = np.array([0,0,0,0,0,0,0])

    for step in range(50):
        
        action = agent.get_action(state=state,exploration_rate=epsilon)      
        #print("ACTION",action)
        
        
        new_state, reward, done= env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
        #print("Len Memory", len(agent.memory))
        print("Percentage of completion", episode/(2000))
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {}, action:{} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:]),action))
            break

    rewards.append(episode_reward)
    actions.append(action)
    avg_rewards.append(np.mean(rewards[-10:]))
    epsilon = epsilon-0.01*episode






#plt.plot(rewards)
#plt.plot(avg_rewards)
plt.plot(actions)
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.legend(days_of_week, loc='upper right')
plt.xlabel('Episode')
plt.ylabel('actions')
#plt.ylabel('Reward')
plt.show()




plt.plot(rewards)
#plt.plot(avg_rewards)

plt.xlabel('Episode')
plt.ylabel('rewards')
#plt.ylabel('Reward')
plt.show()


