import os
os.chdir("C:/Users/ziedk/OneDrive/Bureau/Strasbourg/Master data Strasbourg/RL Muller/Farol Bar 2")
import numpy as np
import gym
from collections import deque
import random

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv():
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
    
    def reward(self,actions_per_week,optimal_crowd=60):
        number_of_people_per_day = [sum(i) for i in actions_per_week]
        rewards = list()
        
        
        for i in actions_per_week:   #actions_per_week = [[0,1...],[1,0,0...],...[1,0,1...]] une liste de 7 listes chaque sublist a 100 élements 
            reward = list()
            number_of_people_in_day_i = sum(i) # Donne pour chaque jour le nombre d'individus dans le bar
            
            
            if number_of_people_in_day_i == 0:
                reward = [-5 for j in range(0,100)]
            elif number_of_people_in_day_i < 0.5 * optimal_crowd:
                reward= [-5 if j ==1 else 0 for j in i]
            elif number_of_people_in_day_i < 0.9 * optimal_crowd:
                reward = [1 if j ==1 else 0 for j in i]
            elif number_of_people_in_day_i < 1.1 * optimal_crowd:
                reward = [15 if j ==1 else 0 for j in i]
            elif number_of_people_in_day_i < 1.3 * optimal_crowd:
                reward = [1 if j ==1 else 0 for j in i]
            else:
                reward = [-5 if i ==1 else 0 for j in i]  
            rewards.append(sum(reward)/100)  #rewards contient la moyenne des rewards de chaque jour
        return sum(rewards)/7 # retourne le reward global de toute la semaine
    



        
    def step(self,action):
        
        actions_per_week = list()
        for i in range(0,7):
            individual_actions = list()
            for j in range(0,100):
                if random.uniform(0, 1) <= action[i]:
                    individual_actions.append(1) # Go to the bar
                else:
                    individual_actions.append(0) # Don't go to the bar 
            actions_per_week.append(individual_actions)
        new_state = action
        rewards_score = self.reward(actions_per_week)
        done=False
        return new_state, rewards_score,done

        

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)


    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
