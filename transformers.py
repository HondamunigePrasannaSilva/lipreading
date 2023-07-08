import torch
import torch.nn as nn
import torch.optim as optim

from utils import linear_interpolation

class Transformer_Encoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.output_dim = output_dim

        self.fc_in = nn.Linear(204, 768)

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc_out = nn.Linear(768, output_dim)
        
        
    def forward(self, land):
        
        #input = [batch size]
        #hidden = [n directions*num_layers, batch size, hid dim]
        
        #input = [batch size, 1]
        #packed_seq = nn.utils.rnn.pack_padded_sequence(input.permute(1,0,2), len_.to('cpu'), enforce_sorted=False)

        #output, _ = self.rnn(packed_seq.to(torch.float32))
        

        land = self.fc_in(land)

        output = self.transformer(land)

        #outputs, _ = nn.utils.rnn.pad_packed_sequence(output) 
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        prediction = self.fc_out(output)

        #prediction = self.tan(prediction)
        
        #prediction = [batch size, output dim]
        
        return prediction#, hidden
    

class Transformer_Decoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.output_dim = output_dim

        #encoder_layer = nn.TransformerEncoderLayer(d_model=204, nhead=3)
        #self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc_in = nn.Linear(204, 768)
        #self.actv = nn.ReLU()
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=3)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        self.fc_out = nn.Linear(768, output_dim)

        #self.tan = nn.Tanh()
        
        
    def forward(self, land, len_land, audio):
        
        #input = [batch size]
        #hidden = [n directions*num_layers, batch size, hid dim]
        
        #input = [batch size, 1]
        #packed_seq = nn.utils.rnn.pack_padded_sequence(input.permute(1,0,2), len_.to('cpu'), enforce_sorted=False)

        #output, _ = self.rnn(packed_seq.to(torch.float32))
        

        land = self.fc_in(land)

        audio = linear_interpolation(audio, 50, 60,output_len=len_land)
        output = self.transformer_decoder(land, audio)

        #outputs, _ = nn.utils.rnn.pad_packed_sequence(output) 
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        prediction = self.fc_out(output)

        #prediction = self.tan(prediction)
        
        #prediction = [batch size, output dim]
        
        return prediction, output#, hidden

class Transformer_Decoder_LSTM(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.output_dim = output_dim

        #encoder_layer = nn.TransformerEncoderLayer(d_model=204, nhead=3)
        #self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc_in = nn.Linear(204, 768)
        #self.actv = nn.ReLU()
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=3)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        self.rnn = nn.LSTM(768, 64, num_layers=2, bidirectional=True, batch_first=True)#, dropout = dropout
        
        self.fc_out = nn.Linear(2*64, output_dim)

        #self.tan = nn.Tanh()
        
        
    def forward(self, land, len_land, audio):
        
        #input = [batch size]
        #hidden = [n directions*num_layers, batch size, hid dim]
        
        #input = [batch size, 1]
        #packed_seq = nn.utils.rnn.pack_padded_sequence(input.permute(1,0,2), len_.to('cpu'), enforce_sorted=False)

        #output, _ = self.rnn(packed_seq.to(torch.float32))
        

        land = self.fc_in(land)

        audio = linear_interpolation(audio, 50, 60,output_len=len_land)
        output = self.transformer_decoder(land, audio)

        #outputs, _ = nn.utils.rnn.pad_packed_sequence(output) 
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]

        seq, _ = self.rnn(output)
        
        prediction = self.fc_out(seq)

        #prediction = self.tan(prediction)
        
        #prediction = [batch size, output dim]
        
        return prediction, output#, hidden