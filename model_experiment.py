import torch
import torch.nn as nn
import torch.optim as optim

#from torchtext.legacy.datasets import Multi30k
#from torchtext.legacy.data import Field, BucketIterator

#import spacy
import numpy as np

import random
import math
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
class only_Decoder2(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, output_dim):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim

        self.rnn = nn.LSTM(input_dim, hid_dim, num_layers=n_layers, bidirectional=True, batch_first=True)#, dropout = dropout
        
        self.fc_out = nn.Linear(2*hid_dim, output_dim)

        #self.tan = nn.Tanh()
        
        
    def forward(self, input, len_):
        
        #input = [batch size]
        #hidden = [n directions*num_layers, batch size, hid dim]
        
        #input = [batch size, 1]
        #packed_seq = nn.utils.rnn.pack_padded_sequence(input.permute(1,0,2), len_.to('cpu'), enforce_sorted=False)

        #output, _ = self.rnn(packed_seq.to(torch.float32))
        output, (hidden, cell) = self.rnn(input.to(torch.float32))#MODIFIED

        #outputs, _ = nn.utils.rnn.pad_packed_sequence(output) 
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        prediction = self.fc_out(output)

        #prediction = self.tan(prediction)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden#