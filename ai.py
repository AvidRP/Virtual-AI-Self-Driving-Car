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
    
    #Take input_state as a param because select_action depends directly on the q values
    #q values are the output of the neural network
    #output of the neural network directly depends on the input of the neural network
    #states is 5D
    def select_action(self, state):
        #softmax function is used to select an action
        #softmax lets the ai to explore but also exploits what it already knows
        #probs is the probability of each q values that comes from softmax
        #Need to use the F lib to implement it
        #state is a torch tensor. Wrap the state around a tensor var to convert
        #temp param (t=7) tells which action to play. temp is between 0 and 1
        #if closer to 0 less likely, if closer to 1 ai is more likey to take that action
        probs = F.softmax(self.model(Variable(state, volatile = True))*7) #the output of model are the q values
        #what does the t = 7 actually do?:
            #softmax([1,2,3]) = [0.04, 0.11, 0.85] => softmax([1,2,3]*3) = [0, 0.02, 0.98]
            #Basically inflates the probability of the higher q value
            #So ai will be even more confident about taking this action
        #Random draw of the probs
        action = probs.multinomial() #action is the "extra" six dimension
        return action.data[0,0] #the action (0, 1 or 2) is encoded in data[0,0]
    
    #forward propagation and back propagation
    #take batches of params instead of every single data so the ai is less biased
    #transition of MDP
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #output only wants the action that is chosen not all the q values of all actions
        #1, batch-action helps acomplish that
        #The squeeze at the end is to get rid of the batches since only need batches in the neural network
        #in the learning process ai needs the actual output -> up left right
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) #need to give batch_action that extra dimension too
        #next_output to calculate the target value
        #need to detach the output of batch_next_state to compare and get the max (according to the equation)
        #1->to specify taking max wrt action (represented by index 1)
        #0->to get the max of the next state wrt action
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        #comes straight out of the formula
        target = self.gamma*next_outputs + batch_reward
        #need to use all this info to calculate Temporal Difference Loss
        td_loss = F.smooth_l1_loss(outputs, target)
        #Need to now back propagate the td_loss 
        #Need to reinitialize optimizer for every loop
        self.optimizer.zero_grad()
        #retain_variables frees some space
        td_loss.backward(retain_variables = True)
        #Need to update the weights of the actions
        self.optimizer.step()
        
    #This updates all the required things when ai transitions into new state 
    #It also adds the new stuff to memory
    #connects ai to map
    def update(self, reward, new_signal):
        #new to convert signal into a tensor type and add the extra dimension
        new_state = torch.tensor(new_signal).float().unsqueeze(0)
        #need to update the new transition into mem
        #all are torch tensors except for last_action -> need to convert
        #last reward is already  a float so no need to use Long Tensor
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.tensor([self.last_reward])))
        action = self.select_action(new_state)
        #Now time for ai to learn->see if it's doing things correctly
        if len(self.memory.memory) > 100:
            #random 100 transitions from mem
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        #updating the last action, state and reward for the ai 
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        #sliding window: window has a fixed size
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    
        
            
        
        
    
        
        
        

    
    
        