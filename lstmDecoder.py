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
    def __init__(self, input_dim, hid_dim, n_layers):
        super().__init__()
        
        self.rnn = nn.LSTM(input_dim, hid_dim, num_layers=n_layers, bidirectional=True, batch_first=True)#, dropout = dropout
        
    def forward(self, x):

        #The input to the encoder are the landmarks
        #x = [batch size, sequence_len , 68*3]
        #embedded = self.embedding(x.to(torch.int))

        out, (hid, cell) = self.rnn(x)
        
        #out = [batch size, src len, hid dim * n directions]
        #hid = [n directions, batch size, hid dim]
        #cell = [n directions, batch size, hid dim]

        #print("ENCODER: hid.shape: ", hid.shape)
        return hid, cell
    

class Decoder(nn.Module):
    def __init__(self, hid_dim, n_layers, output_dim):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim

        #self.embedding = nn.Embedding(output_dim, emb_dim)

        #put 1 instead of embed dim
        self.rnn = nn.LSTM(input_size=1, hidden_size=hid_dim, bidirectional=True, batch_first=True, num_layers=n_layers)
        
        self.fc_out = nn.Linear(2*hid_dim, output_dim)
    
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n directions, batch size, hid dim]
        
        #input = [batch size, 1]
        #embedded = self.embedding(input)
                
        out, (hid, cell) = self.rnn(input.to(torch.float32), (hidden, cell))#input.to(torch.float32)
        
        #out = [seq len, batch size, hid dim * n directions]
        #hid = [n layers * n directions, batch size, hid dim]
        
        pred = self.fc_out(out)#.squeeze(0))
        
        #pred = [batch size, output dim]
        
        return pred, hid, cell
    

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
        hidden, cell = self.encoder(src)
        #hidden = hidden.unsqueeze(1)
        
        #first input to the decoder is the # token
        input = torch.full((batch_size , 1 , 1), 1, dtype=torch.long).to(self.device)
        
        for t in range(0, trg_len):
            
            #insert input token embedding, previous hidden state
            #receive output tensor (predictions) and new hidden

            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #output = output.unsqueeze(0)#ADDED TO BE CHECKED

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output[:, 0]
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(-1, keepdim=True) 

            input = top1
        
        return outputs.permute(1,0,2)
    
class only_Decoder_linear_emb(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, output_dim):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim

        self.fc_in = nn.Linear(input_dim, 256)

        self.rnn = nn.LSTM(256, hid_dim, num_layers=n_layers, bidirectional=True, batch_first=True)#, dropout = dropout
        
        self.fc_out = nn.Linear(2*hid_dim, output_dim)
        
        
    def forward(self, input, len_):
        
        #input = [batch size]
        #hidden = [n directions*num_layers, batch size, hid dim]
        
        #input = [batch size, 1]
        x = self.fc_in(input.to(torch.float32))

        output, _ = self.rnn(x)
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        prediction, (hidden, cell) = self.fc_out(output)

        #prediction = self.tan(prediction)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell


class only_Decoder(nn.Module):
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

        output, (hidden, cell) = self.rnn(input.to(torch.float32))

        #outputs, _ = nn.utils.rnn.pad_packed_sequence(output) 
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        prediction = self.fc_out(output)

        #prediction = self.tan(prediction)
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell#, hidden

"""
    def forward(self, x):
        
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)"""


class only_Decoder_MLP_emb(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, output_dim, path_emb):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim

        self.emb = MLP_emb(input_dim)#.to(device)
        if path_emb != None:
            #self.emb.load_state_dict(torch.load("./models/model_embexp6_trvalloss070.pt"))
            self.emb.load_state_dict(torch.load(path_emb))

        #+substitute input_dim with 768

        self.rnn = nn.LSTM(768, hid_dim, num_layers=n_layers, bidirectional=True, batch_first=True)#, dropout = dropout
        
        self.fc_out = nn.Linear(2*hid_dim, output_dim)
              
    def forward(self, input, len_):
        
        #input = [batch size]
        #hidden = [n directions*num_layers, batch size, hid dim]
        
        #input = [batch size, 1]

        #Frozen because we expect a pretrained embedding
        with torch.no_grad():
            input = self.emb(input.to(torch.float32))

        output, (hidden, cell) = self.rnn(input)#MODIFIED
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        prediction = self.fc_out(output)

        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell#
    
class MLP_emb(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.linear0 = nn.Linear(input_dim, 768)
        self.linear1 = nn.Linear(768, 512)
        self.linear2 = nn.Linear(512, 768)

        self.actv = nn.ReLU()

    def forward(self, x):
        x = self.linear0(x)
        x = self.actv(x)
        x = self.linear1(x)
        x = self.actv(x)
        x = self.linear2(x)
        return x