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


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers):
        super().__init__()
        
        self.hid_dim = hid_dim
        
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=128, bidirectional=True, batch_first=True, num_layers=num_layers)
        
    def forward(self, x):

        #The input to the encoder are the landmarks
        #x = [batch size, sequence_len , 68*3]

        out, hid = self.rnn(x)
        
        #out = [batch size, src len, hid dim * n directions]
        #hid = [n directions * num_layers, batch size, hid dim]

        #print("ENCODER: hid.shape: ", hid.shape)
        return hid
    



class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, num_layers):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim

        self.rnn = nn.GRU(input_size=1, hidden_size=128, bidirectional=True, batch_first=True, num_layers=num_layers)
        
        self.fc_out = nn.Linear(2*hid_dim, output_dim)
    
        
    def forward(self, input, hidden):
        
        #input = [batch size]
        #hidden = [n directions*num_layers, batch size, hid dim]
        
        #input = [batch size, 1]
                
        output, hidden = self.rnn(input.to(torch.float32), hidden)
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        prediction = self.fc_out(output)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden
    


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
 
        batch_size = trg.shape[0]
        trg_len = src.shape[1]#FIXME Forse qui non ho passato i landmark con la dimensione del batch in testa
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(src)
        #hidden = hidden.unsqueeze(1)
        
        #first input to the decoder is the <blank> token
        input = torch.full((batch_size , 1 , 1), 37, dtype=torch.long).to(self.device)
        
        for t in range(0, trg_len):
            
            #insert input token embedding, previous hidden state
            #receive output tensor (predictions) and new hidden

            output, hidden = self.decoder(input, hidden)
            
            #output = output.unsqueeze(0)#ADDED TO BE CHECKED

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output[:, 0]
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(-1, keepdim=True) 

            input = top1
        
        return outputs
    

class only_Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers, output_dim):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim

        self.rnn = nn.GRU(input_size=input_dim, hidden_size=128, bidirectional=True, batch_first=True, num_layers=num_layers)
        
        self.fc_out = nn.Linear(2*hid_dim, output_dim)
    
        
    def forward(self, input):
        
        #input = [batch size]
        #hidden = [n directions*num_layers, batch size, hid dim]
        
        #input = [batch size, 1]
                
        output, hidden = self.rnn(input.to(torch.float32))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        prediction = self.fc_out(output)
        
        #prediction = [batch size, output dim]
        
        return prediction#, hidden