import os
os.chdir("C:/Users/ziedk/OneDrive/Bureau/Strasbourg/Master data Strasbourg/RL Muller/Farol Bar 2")
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable



class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        #print("-------",state.shape)
        #print("-------",state)

        #print("***********",action.shape)
        
        #state = state.view(128)
        #state = state.view(1, -1)
        #action = action.view(1, -1)
        #action = action.unsqueeze(dim=1)        
        #print("action shape", action.shape)
        #print("state shape",state.shape)

        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.sigmoid(self.linear3(x))

        return x
    
