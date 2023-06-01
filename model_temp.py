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
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        
        self.hid_dim = hid_dim
        
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers=1)#, dropout = dropout
        
    def forward(self, x):

        #The input to the encoder are the landmarks
        #x = [batch size, sequence_len , 68*3]

        out, (hid, cell) = self.rnn(x, batch_first=True)
        
        #out = [batch size, src len, hid dim * n directions]
        #hid = [n directions, batch size, hid dim]
        #cell = [n directions, batch size, hid dim]

        #print("ENCODER: hid.shape: ", hid.shape)
        return hid, cell
    

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim

        self.rnn = nn.LSTM(input_size=output_dim, hidden_size=128, bidirectional=True, batch_first=True)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
    
        
    def forward(self, input, hidden):
        
        #input = [batch size]
        #hidden = [n directions, batch size, hid dim]
        
        #input = [batch size, 1]
                
        out, (hid, cell) = self.rnn(input.to(torch.float32), hidden)
        
        #out = [seq len, batch size, hid dim * n directions]
        #hid = [n layers * n directions, batch size, hid dim]
        
        pred = self.fc_out(out.squeeze(0))
        
        #pred = [batch size, output dim]
        
        return pred, hid, cell
    


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
 
        batch_size = trg.shape[0]
        trg_len = src.shape[0]#FIXME Forse qui non ho passato i landmark con la dimensione del batch in testa
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        hidden = hidden.unsqueeze(1)
        
        #first input to the decoder is the <sos> tokens
        input = torch.tensor(31).unsqueeze(0).unsqueeze(0).unsqueeze(0)#trg[0,:] 31 is the index of <sos>
        input = torch.full((0,), output.size(0), dtype=torch.long)
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state
            #receive output tensor (predictions) and new hidden

            output, hidden, cell = self.decoder(input, hidden, cell)
            
            output = output.unsqueeze(0)#ADDED TO BE CHECKED

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(2) 

            input = top1.unsqueeze(0)
        
        return outputs