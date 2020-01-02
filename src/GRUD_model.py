import torch
import numpy as np
import pandas as pd
import os
import math
import warnings
import itertools
import numbers
import torch.utils.data as utils
from GRUD_layer import GRUD_cell



def grud_model_old( input_size, hidden_size, output_size, num_layers=1, x_mean=0,\
                bias=True, batch_first=False, bidirectional=False, dropout_type='mloss', dropout=0):

    layer_list =[]
    #intermediate layers return input size as output size
    for i in range(num_layers-1): #subtract 1 bc last layer needs to be called with different params
        layer = GRUD_cell(input_size = input_size, hidden_size= hidden_size, output_size=input_size, dropout=dropout, dropout_type=dropout_type, x_mean=x_mean, num_layers=num_layers, return_hidden = True)
        layer_list.append(layer)

    #last layer with final output size
    layer = GRUD_cell(input_size = input_size, hidden_size=hidden_size, output_size=output_size, dropout=dropout, dropout_type=dropout_type, x_mean=x_mean, num_layers=num_layers, return_hidden = False)
    layer_list.append(layer)

    model = torch.nn.Sequential(*layer_list)

    return model


class grud_model(torch.nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layers = 1, x_mean = 0,\
     bias =True, batch_first = False, bidirectional = False, dropout_type ='mloss', dropout = 0):
        super(grud_model, self).__init__()

        self.gru_d = GRUD_cell(input_size = input_size, hidden_size= hidden_size, output_size=output_size, 
                dropout=dropout, dropout_type=dropout_type, x_mean=x_mean)
        self.hidden_to_output = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.num_layers = num_layers

        if self.num_layers >1:
            #(batch, seq, feature)
            self.gru_layers = torch.nn.GRU(input_size = hidden_size, hidden_size = hidden_size, batch_first = True, num_layers = self.num_layers -1)

    def forward(self,input):

        #pass through GRU-D
        output, hidden = self.gru_d(input)
        #print(self.gru_d.return_hidden)
        #output = self.gru_d(input)
        #print(output.size())

        # batch_size, n_hidden, n_timesteps

        if self.num_layers >1:
            output, hidden = self.gru_layers(hidden)
  

            output = self.hidden_to_output(output)
            output = torch.sigmoid(output)

        print("final output size passed as model result")
        print(output.size())
        return output

