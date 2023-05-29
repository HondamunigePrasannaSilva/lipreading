import torch
import torch.nn as nn
import torch.optim as optim

# Define the sequence and target labels
sequence = "hhhheeelllooo0iiiiaaaamm0sssssillllvvvaa"

target ="hello0iam0silva"

# Define the vocabulary
vocabulary = ['-','h', 'e', 'l', 'o','a','b','c','d', '0','i','m','s','v']

# Create a mapping from characters to indices
char_to_index = {char: index for index, char in enumerate(vocabulary)}

# Convert the sequence and target to indices
sequence_indices = [char_to_index[char] for char in sequence]
target_indices = [char_to_index[char] for char in target]

# Convert the indices to PyTorch tensors
sequence_tensor = torch.tensor(sequence_indices).unsqueeze(0)  # Add a batch dimension
target_tensor = torch.tensor(target_indices)

# Define the model
class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.rnn = nn.RNN(input_size=1, hidden_size=128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.permute(1,0)
        out, _ = self.rnn(x.to(torch.float32))
        out = self.fc(out)
        return out

# Initialize the model
model = Model(len(vocabulary))

# Define the CTC loss function
ctc_loss = nn.CTCLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(sequence_tensor)

    output = output[None,:,:]
    output = output.permute(1, 0, 2)  # Swap batch and sequence dimensions
    
    e = torch.argmax(output, dim=2).squeeze(1)
    output_sequence = ''.join([vocabulary[index] for index in e])
    print(output_sequence)

    
    input_lengths = torch.full((sequence_tensor.size(0),), output.size(0), dtype=torch.long)
    target_lengths = torch.full((target_tensor.size(0),), target_tensor.size(0), dtype=torch.long)
    
    loss = ctc_loss(output, target_tensor, input_lengths, target_lengths[0])
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Decode the output sequence
output_indices = torch.argmax(output, dim=2).squeeze(1)
output_sequence = ''.join([vocabulary[index] for index in output_indices])

print("Original Sequence:", sequence)
print("Target Sequence:", target)

def process_string(input_string):
    output_string = ""
    current_char = ""

    for char in input_string:
        if char != current_char:
            if char.isalpha() or char == '0':
                if char == '0':
                    output_string += ' '
                else:
                    output_string += char   
            current_char = char

    return output_string.strip()

print("Decoded Output:", process_string(output_sequence))