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
        return q_values
    
#implementing experience replay
#store batches making it long term memory instead of short term memory
class ReplayMemory(object): 
    #capacity -> batches of data to store -> 100
    def __init__(self, capacity):
        self.capacity = capacity
        #memory contains the last 100 events
        self.memory = []
        
    # event has three parts:
        #last state, last action, last reward
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    #Gets random samples from out memory
    #batch size is how many samples do we return
    def sample(self, batch_size):
        #random ->sample function from random library 
        #take random sample from memory of batch_size
        #zip(*) reshapes your list -> separates the states action and reward and stores each as a batch
        samples = zip(*random.sample(self.memory, batch_size))
        #maps variables to torch variables
        #the variable library converts x
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
#Deep Q Learning Model
class Dqn():
    
    #takes the same params as Network class
    #gamma -> discount factor
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        #mean of the last 100 rewards
        self.reward_window = []
        #model is just the neural network obj
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        #uses torch.optim class to optimize the network
        #Creating an obj of the Adam class from optim
        #the first param connects the network model to the optimizer
        #lr (learning rate) needs to be optimized; if too high the ai wont explore
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        #last state, last action and last reward
        #last_state is a 5D vector: 3 sensor signals, orientation and -orientation
        #for pytorch can't just be a vector needs to be a tensor
        #However the network expects 6 dimensions so add a new dimension at index 0 using unsqueeze
        self.last_state =  torch.tensor(input_size).unsqueeze(0)
        #Have an action2rotation in map.py-> 0 = 0degrees, 1 = 20degress, -1 = -20degress
        self.last_action = 0 #just initializing it with 0
        self.last_reward = 0 #reward is between 0 and 1
        
        

    
    
        