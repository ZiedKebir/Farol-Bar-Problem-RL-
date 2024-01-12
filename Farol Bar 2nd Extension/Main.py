import os
os.chdir("C:/Users/ziedk/OneDrive/Bureau/Strasbourg/Master data Strasbourg/RL Muller/Farol Bar 2 Complex Extension 2 models")

import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DDPG import *
from Trainer import *
from models import * 

env = NormalizedEnv()
env2 = NormalizedEnv()


agent = DDPGagent()
agent2 = DDPGagent()
#noise = OUNoise(env.action_space)


batch_size = 128
rewards1 = []
rewards2 = []

actions=[]
actions2=[]

avg_rewards = []
epsilon = 0.7

for episode in range(0,70):
    print("Percentage of completion", episode/(70))

    #state = env.reset()
    #noise.reset()
    episode_reward1 = 0
    episode_reward2 = 0

    state =np.random.rand(2)
    state2 = np.random.rand(2)

    
    for step in range(50):
        ############ 1st Model ##################### 
        action = agent.get_action(state=state2,exploration_rate=epsilon) 
        
        action2 = agent.get_action(state=state,exploration_rate=epsilon) 
        
        
        #print("action 1", action)
        #print("action 2", action2)
        
        
        #print("ACTION",action)
        
        
        
        reward,new_state,new_state2,done = env.overall_step(action,action2)
        
        
        
        #new_state, done= env.step(action) 
        #print("new state 1", new_state)
        
        #new_state2, done2= env.step(action2) 
        #print("new state 2", new_state2)

        
        agent.memory.push(state2, action, reward[1], new_state2, done)
        #print("state 1",state2)
        print("reward1",reward[0])
        
        agent2.memory.push(state, action2, reward[0], new_state, done)
        #print("state 2",state)
        print("reward2",reward[1])



        #print("Len Memory", len(agent.memory))
        if len(agent.memory) > batch_size:
            agent.update(batch_size)  
            
        if len(agent2.memory) > batch_size:
            agent2.update(batch_size)        

        state = new_state.copy()
        state2 = new_state2.copy()


        episode_reward1 += reward[0]
        episode_reward2 += reward[1]


        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {}, action:{} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:]),action))
            break
        
        
        
        
    rewards1.append(episode_reward1)
    rewards2.append(episode_reward2)

    actions.append(action)
    actions2.append(action2)

    #avg_rewards.append(np.mean(rewards[-10:]))
    
    epsilon = epsilon-0.01*episode





#plt.plot(rewards)
#plt.plot(avg_rewards)
plt.plot(actions)
days_of_week = ['Day1', 'Day2']
plt.legend(days_of_week, loc='upper right')
plt.xlabel('Episode')
plt.ylabel('actions')
plt.title("Model 1")
#plt.ylabel('Reward')
plt.show()




#plt.plot(rewards)
#plt.plot(avg_rewards)
plt.plot(actions2)
days_of_week = ['Day1','Day2']
plt.legend(days_of_week, loc='upper right')
plt.xlabel('Episode')
plt.ylabel('actions')
plt.title("Model 2")
#plt.ylabel('Reward'")
plt.show()



r = list()
for i in range(0,len(rewards1)):
    r.append(rewards2[i]+rewards1[i])




plt.plot(r)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()




print(rewards1)


print(actions[299])



print(actions2[299])