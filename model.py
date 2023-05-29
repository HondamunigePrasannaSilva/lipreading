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
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        
        #self.embedding = nn.Embedding(input_dim, emb_dim) FOR NOW WE DON't HAVE INPUT EMBEDDINGS
        
        self.rnn = nn.RNN(input_size=68*3, hidden_size=128, bidirectional=True, batch_first=True)
        
        #self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):

        #The input to the encoder are the landmarks
        #x = [batch size, sequence_len , 68*3]
        
        #embedded = self.dropout(self.embedding(src))FOR NOW WE DON't HAVE INPUT EMBEDDINGS
        
        #embedded = [src len, batch size, emb dim]

        out, hid = self.rnn(x)
        
        #out = [batch size, src len, hid dim * n directions]
        #hidd = [n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer

        #print("ENCODER: hid.shape: ", hid.shape)
        return hid
    



class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        #self.n_layers = n_layers
        
        #self.embedding = nn.Embedding(output_dim, emb_dim)FOR NOW WE DON't HAVE INPUT EMBEDDINGS
        self.rnn = nn.RNN(input_size=1, hidden_size=128, bidirectional=True, batch_first=True)
        
        self.fc_out = nn.Linear(2*hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        #input = input.unsqueeze(0).unsqueeze(0)COMMENTED TO DEBUG

        #print("DECODER: input.shape", input.shape)
        
        #input = [batch size, 1]
        
        #embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, hidden = self.rnn(input.to(torch.float32), hidden)
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]

        #print("DECODER: prediction.shape", prediction.shape)
        
        return prediction, hidden
    


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        #assert encoder.hid_dim == decoder.hid_dim, \
        #    "Hidden dimensions of encoder and decoder must be equal!"
        #assert encoder.n_layers == decoder.n_layers, \
        #    "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[0]
        trg_len = src.shape[0]#FIXME Forse qui non ho passato i landmark con la dimensione del batch in testa
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(src)
        hidden = hidden.unsqueeze(1)
        #print("SEQ2SEQ: hidden.shape: ", hidden.shape)
        
        #first input to the decoder is the <sos> tokens
        input = torch.tensor(31).unsqueeze(0).unsqueeze(0).unsqueeze(0)#trg[0,:] 31 is the index of <sos>
        #print("SEQ2SEQ: input.shape: ", input.shape)
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            #print("SEQ2SEQ-FOR: input.shape: ", input.shape)
            #print("SEQ2SEQ-FOR: hidden.shape: ", hidden.shape)
            output, hidden = self.decoder(input, hidden)
            #print("SEQ2SEQ-FOR: output.shape: ", output.shape)
            
            output = output.unsqueeze(0)#ADDED TO BE CHECKED

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(2) 

            input = top1.unsqueeze(0)
            #print("SEQ2SEQ-FOR: top1.shape: ", top1.shape)
        
        return outputs