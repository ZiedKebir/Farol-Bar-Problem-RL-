import os
os.chdir("C:/Users/ziedk/OneDrive/Bureau/Strasbourg/Master data Strasbourg/RL Muller/Farol Bar 2")
import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from models import *
from Trainer import *
from torch.autograd import Variable


#X = Actor(input_size=1, hidden_size=256, output_size=1,learning_rate = 3e-4)



class DDPGagent:
    def __init__(self,hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=50000):
        # Params
        #self.num_states = env.observation_space.shape[0]
        #self.num_actions = env.action_space.shape[0]
        self.num_states = 1 
        self.num_actions = 1

        self.gamma = gamma
        self.tau = tau

        # Networks
        
        
        self.actor =  Actor(input_size=1, hidden_size=hidden_size, output_size=1,learning_rate = 3e-4)
        #self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(input_size=1, hidden_size=hidden_size, output_size=1,learning_rate = 3e-4)
        #self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)

        self.critic = Critic(input_size=2, hidden_size=hidden_size, output_size=1)
       
        #self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        self.critic_target = Critic(input_size=2, hidden_size=hidden_size, output_size=1)

        #self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    def get_action(self, state,exploration_rate: float): ### I updated this function ### 
        if random.uniform(0,1)<=exploration_rate: #in this case the agent explores 
            return random.uniform(0,1)
        else: #The agent exploits 
            state = Variable(torch.from_numpy(state).float().unsqueeze(0))
            action = self.actor.forward(state)
            action = action.detach().numpy()[0,0]
            return action
            
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        actions = actions.unsqueeze(dim=1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        next_states = next_states.unsqueeze(dim=1)
        #print("next_states",next_states.shape)
        # Critic loss   
        #print("states",states)
        #print("actions",states)

        
        Qvals = self.critic.forward(states, actions)
        #print("Qvals",Qvals)
        #print("Qvqls",Qvals)
        next_actions = self.actor_target.forward(next_states)
        #print("++++++",next_actions.shape)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
