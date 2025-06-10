import torch
from torch import nn


#Layer size
#n_input_dim = 128
n_hidden1 = 128  # Number of hidden nodes
n_hidden2 = 64
n_hidden3 = 32
n_hidden4 = 8
n_output =  1   # Number of output nodes = for binary classifier

class FND(torch.nn.Module):
    def __init__(self, n_input_dim = 128):
        super(FND, self).__init__()
        #self.tnet = tnet
        self.layer_1 = nn.Linear(n_input_dim, n_hidden1) 
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        #self.layer_3 = nn.Linear(n_hidden2, n_hidden3)
        #self.layer_4 = nn.Linear(n_hidden3, n_hidden4)

        self.layer_out = nn.Linear(n_hidden2, n_output) 
        
        self.relu = nn.LeakyReLU()
        self.sigmoid =  nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.functional.softmax
        self.dropout = nn.Dropout(p=0.01)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden2)

    def forward(self, embed):
    
        x = self.relu(self.layer_1(embed))
        #x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        #x = self.relu(self.layer_3(x))
        #x = self.relu(self.layer_4(x))
        #x = self.batchnorm2(x)
        #x = self.dropout(x)
        #print("x",x)
        #x = self.softmax(x)
        x = self.tanh(x)
        x = self.sigmoid(self.layer_out(x))
        
        return x