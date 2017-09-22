#importing the required library

import numpy as np
#for taking random samples when you use experience replay
import random
#load and save the model (brain)
import os 
#neural network is impleted with pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
#for optimizing gradient decent
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


#developing the neural network
#5 input neurons because the car has 5 inuput states for the car

#inhereting from the parent class which is nn.Module
class Network(nn.Module):
    
    #second argument for #of input neurons -> 5 in this case (vectors of the car)
    #third argument for #of output neurons ->action that the car takes (forward, right or left)
    def __init__(self, input_size, nb_action):
        #super function to use nn.Module
        super(Network, self).__init__()
        #defining the input layer of the neural network
        self.input_size = input_size
        self.nb_action = nb_action
        #connection between different layers
        #one connection between hidden and first
        #other connection between hidden and last
        #nn.linear is used to connect ALL the input to ALL the hidden neurons
        #30 is hidden neurons gives a good result
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    
    #function for forward propagation
    #the return value is the q values
    def forward(self, state):
        #activiting hidden neurons using rectifier fucntion from F
        #need to pass in state to get from input state to hidden state
        x = F.relu(self.fc1(state))
        #output neurons -> q values
        q_values = self.fc2(x)
        #need to return the q values
        return q
    
    
        