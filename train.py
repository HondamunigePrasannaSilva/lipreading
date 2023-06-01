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

# model file, encoder, decoder and seqtoseq
from model import *
# utils file
from utils import *
# Get landmark using vocadataset.py
from data.vocaset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



trainset = vocadataset("train", landmark=True)
dataloader = DataLoader(trainset, batch_size=32, collate_fn=collate_fn, num_workers=8)
print(device)

vocabulary = create_vocabulary(blank='@')

LANDMARK_DIM = 68
INPUT_DIM = LANDMARK_DIM*3
HID_DIM = 128
output_dim = len(vocabulary)

enc = Encoder(INPUT_DIM, HID_DIM)
dec = Decoder(output_dim, HID_DIM)
model = Seq2Seq(enc, dec, device).to(device)


# With batch

# Define the CTC loss function
ctc_loss = nn.CTCLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for landmarks, len_landmark, label, len_label in dataloader:
        # reshape the batch from [batch_size, frame_size, num_landmark, 3] to [batch_size, frame_size, num_landmark * 3] 
        landmarks = torch.reshape(landmarks, (landmarks.shape[0], landmarks.shape[1], landmarks.shape[2]*landmarks.shape[3]))
        # label char to index
        label = char_to_index_batch(label, vocabulary)

        landmarks = landmarks.to(device)
        len_landmark = len_landmark.to(device)
        label = label.to(device)
        len_label = len_label.to(device)
        optimizer.zero_grad()

        output = model(landmarks, label)
        
        #input_lengths = torch.full((batch_size,), output.size(0), dtype=torch.long) Serve se le sequenze hanno lunghezze uguali
        
        loss = ctc_loss(output, label, len_landmark, len_label)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
            #e = torch.argmax(output, dim=2).squeeze(1)
            #output_sequence = ''.join([vocabulary[index] for index in e])
            #print(output_sequence)
            torch.save(model.state_dict(), "models/model.pt")