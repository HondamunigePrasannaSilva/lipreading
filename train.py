
import torch
import wandb
from data.vocaset import *
from utils import *
from model import *
import tqdm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hyper = {
    'LANDMARK_DIM' : 36,
    'INPUT_DIM' : 36*3,
    'HID_DIM' : 32,
    'BATCH_SIZE': 1,
    'EPOCHS': 100,
    'NUM_LAYERS': 2
}



def model_pipeline():

    with wandb.init(project="lip-reading", config=hyper, mode="disabled"):
        #access all HPs through wandb.config, so logging matches executing
        config = wandb.config

        #make the model, data and optimization problem
        model, ctc_loss, optimizer,trainloader, valloader, vocabulary = create(config)

        #train the model
        mean_loss = train(model, ctc_loss, optimizer,trainloader, vocabulary, config)
        print("Mean train loss:", mean_loss)

        #test the model
        #print("Accuracy test: ",test(model, valloader, vocabulary))
        
    return model

def create(config):

    #get dataloader
    trainset = vocadataset("train", landmark=True, mouthOnly=True)
    valset = vocadataset("val", landmark=True)
    trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=8, shuffle=True)
    valloader = DataLoader(trainset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=8)

    #define the vocabulary
    vocabulary = create_vocabulary(blank='@')

    # define the models
    enc = Encoder(config.INPUT_DIM, config.HID_DIM, config.NUM_LAYERS)
    dec = Decoder(len(vocabulary), config.HID_DIM, config.NUM_LAYERS)
    model = Seq2Seq(enc, dec, device).to(device)

    # Define the CTC loss function
    ctc_loss = nn.CTCLoss(zero_infinity=True)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    return model, ctc_loss, optimizer,trainloader, valloader, vocabulary

# Function to train a model.
def train(model, ctc_loss, optimizer,trainloader, vocabulary, config, modeltitle= "_prova"):
    
    #telling wand to watch
    if wandb.run is not None:
        wandb.watch(model, optimizer, log="all", log_freq=1)

    #model.load_state_dict(torch.load("/home/prasanna/Documents/UNIFI/Computer Graphics/LipReading/lipreading/models/model_prova.pt"))
    model.train()
    
    # Training loop

    for epoch in range(config.EPOCHS):

        #list to save sentences
        real_sentences = []
        pred_sentences = []
        losses = []
        progress_bar = tqdm.tqdm(total=len(trainloader), unit='step')

        for landmarks, len_landmark, label, len_label in trainloader:
                
            # reshape the batch from [batch_size, frame_size, num_landmark, 3] to [batch_size, frame_size, num_landmark * 3] 
            landmarks = torch.reshape(landmarks, (landmarks.shape[0], landmarks.shape[1], landmarks.shape[2]*landmarks.shape[3]))
            
            #variable to recover later the target sequences
            label_list = label

            # label char to index
            label = char_to_index_batch(label, vocabulary)

            # move data to GPU!
            landmarks = landmarks.to(device)
            len_landmark = len_landmark.to(device)
            label = label.to(device)
            len_label = len_label.to(device)
            optimizer.zero_grad()

            output = model(landmarks, label)
                        
            loss = ctc_loss(torch.nn.functional.log_softmax(output, dim=2), label, len_landmark, len_label)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            #progress bar stuff
            progress_bar.set_description(f"Epoch {epoch+1}/{config.EPOCHS}")
            #progress_bar.set_postfix(loss=loss.item())  # Update the loss value
            progress_bar.set_postfix(loss=np.mean(losses), lossb = loss.item())  # Update the loss value
            #progress_bar.set_postfix(loss=np.mean(losses))  # Update the loss value
            progress_bar.update(1)

            real_sentences, pred_sentences = write_results(len_label, label_list, output, trainloader.batch_size, vocabulary, real_sentences, pred_sentences)
        
        # endfor batch 

        if wandb.run is not None:
            wandb.log({"epoch":epoch, "loss":np.mean(losses)}, step=epoch)
        
        # save the model
        torch.save(model.state_dict(), "models/model"+str(modeltitle)+".pt")
        save_results(f"./results/results_{epoch}.txt", real_sentences, pred_sentences, overwrite=True)

    return np.mean(losses)


def test(model, valloader, vocabulary):
    model.eval()

    with torch.no_grad():
        correct, total = 0, 0

        for landmarks, len_landmark, label, len_label in valloader:

            # reshape the batch from [batch_size, frame_size, num_landmark, 3] to [batch_size, frame_size, num_landmark * 3] 
            landmarks = torch.reshape(landmarks, (landmarks.shape[0], landmarks.shape[1], landmarks.shape[2]*landmarks.shape[3]))
            
            # label char to index
            label = char_to_index_batch(label, vocabulary)

            # move data to GPU!
            landmarks = landmarks.to(device)
            len_landmark = len_landmark.to(device)
            label = label.to(device)
            len_label = len_label.to(device)

            output = model(landmarks, label)
            
            final_sentence = torch.argmax(output, dim=2).squeeze(1)
            output_sequence = ''.join([vocabulary[index] for index in final_sentence])         
            # scrittura nel file del outuput e della frase originale


    return 0

model_pipeline()