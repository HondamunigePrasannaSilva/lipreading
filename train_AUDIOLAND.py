
import torch
import torch.nn.functional as F
import wandb
from data.vocaset import *
from utils import *
from lstmDecoder import *
import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hyper = {
    'LANDMARK_DIM' : 68,
    'INPUT_DIM' : 68*3,
    'HID_DIM' : 64,
    'BATCH_SIZE': 1,
    'EPOCHS': 5000,
    'NUM_LAYERS': 2,
    'LR': 3e-4,
    'SERVER':'windu',
    'WEIGHT_SIMILARITY': 1,
    'WEIGHT_CTCLOSS': 1

}

hyper_a = {
    'LANDMARK_DIM' : 768,
    'INPUT_DIM' : 768*1,
    'HID_DIM' : 128,
    'BATCH_SIZE': 1,
    'EPOCHS': 5000,
    'NUM_LAYERS': 2,
    'LR': 3e-4,
    'SERVER':'W'
}

"""Train an LSTM with landmarks input as a student of an LSTM trained on audio. 
You will need the weights of the LSTM trained on audio"""

def model_pipeline():

    with wandb.init(project="AudioLand-3D", config=hyper, mode="disabled"):
        #access all HPs through wandb.config, so logging matches executing
        config = wandb.config

        #make the model, data and optimization problem
        model, ctc_loss, optimizer,trainloader, valloader, vocabulary = create(config)

        #train the model
        mean_loss = train(model, ctc_loss, optimizer,trainloader, vocabulary,config,valloader )

        #test the model
        #print("Accuracy test: ",test(model, valloader, vocabulary))
        
    return model

def create(config):

    #get dataloader
    #trainset = vocadataset("train", landmark=True)
    trainset = vocadataset("train", landmark=True, onlyAudio=True)
    #valset = vocadataset("val", landmark=True)
    valset = vocadataset("val", landmark=True, onlyAudio=True)
    
    trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=8, shuffle=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    #define the vocabulary
    vocabulary = create_vocabulary(blank='@')

    # define the models
    model = only_Decoder(config.INPUT_DIM, config.HID_DIM, config.NUM_LAYERS, len(vocabulary)).to(device)
    #m = torch.compile(model)

    # Define the CTC loss function
    ctc_loss = nn.CTCLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    return model, ctc_loss, optimizer,trainloader, valloader, vocabulary

# Function to train a model.
def train(model, ctc_loss, optimizer,trainloader, vocabulary, config,valloader, modeltitle= "_audlan"):
    
    #telling wand to watch
    #if wandb.run is not None:
    wandb.watch(model, optimizer, log="all", log_freq=320)
    #Load wav2vec2
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav2vec = bundle.get_model().to(device)
    sample_rate = 22000

    #CHOSE THE SIMILARITY LOSS
    #sim_loss = nn.MSELoss()
    sim_loss = nn.CrossEntropyLoss()

    model.train()

    model_audio = only_Decoder(hyper_a['INPUT_DIM'], hyper_a['HID_DIM'], hyper_a['NUM_LAYERS'], len(vocabulary)).to(device)
    #HERE  WE HAVE TO LOAD THE WEIGHTS OF AN LSTM TRAINED ON AUDIO
    model_audio.load_state_dict(torch.load("./models/modeltest_24.pt"))
    model_audio.eval()
    # Training loop

    for epoch in range(config.EPOCHS):

        #list to save sentences
        real_sentences = []
        pred_sentences = []
        losses = []
        progress_bar = tqdm.tqdm(total=len(trainloader), unit='step')
        ct_l_s = []
        csim_s = []
        for landmarks, len_landmark, label, len_label, audio in trainloader:
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
            audio = audio.to(device)
            if sample_rate != bundle.sample_rate:
                audio = torchaudio.functional.resample(audio, sample_rate, bundle.sample_rate)

            with torch.inference_mode():
                audio_features, _ = wav2vec.extract_features(audio[0])

            audio_input = audio_features[-1].clone().requires_grad_()
            #len_audio[0] = audio_input.shape[1]

            optimizer.zero_grad()

            output, hidden, cell = model(landmarks,len_landmark)
            output = output.permute(1, 0, 2)#had to permute for the ctc loss. it acceprs [seq_len, batch_size, "num_class"]

            with torch.no_grad():
                #output_a, hidden_a, cell_a = model_audio(audio_input, len_landmark)
                audio_input = linear_interpolation(audio_input, 50, 60,output_len=len_landmark)
                output_a, hidden_a, _ = model_audio(audio_input, len_landmark)
                hidden_a = hidden_a.permute(1,0,2).flatten(1)
                #cell_a = cell_a.permute(1,0,2).flatten(1)
            
            hidden = hidden.permute(1,0,2).flatten(1)
            #cell = cell.permute(1,0,2).flatten(1)

            ct_l = 0.5*ctc_loss(torch.nn.functional.log_softmax(output, dim=2), label, len_landmark, len_label)
            #csim = -1*F.cosine_similarity(hidden, hidden_a)
            
            #IF MSE IS CHOSEN AS SIMILARITY LOSS
            #csim = sim_loss(hidden, hidden_a)
            #IF CROSS ENTROPY IS CHOSEN AS SIMILARITY LOSS
            csim = sim_loss(output.permute(1,0,2)[0], torch.argmax(output_a, dim=2)[0])
            

            loss = ct_l+csim #+1e-3*F.cosine_similarity(cell, cell_a)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            losses.append(loss.item())
            ct_l_s.append(ct_l.item())
            csim_s.append(csim.item())

            #progress bar stuff
            progress_bar.set_description(f"Epoch {epoch+1}/{config.EPOCHS}")
            #progress_bar.set_postfix(loss=loss.item())  # Update the loss value
            progress_bar.set_postfix(loss=np.mean(losses))  # Update the loss value
            progress_bar.update(1)
            if epoch%10 == 0:
                real_sentences, pred_sentences = write_results(len_label, label_list, output.detach(), trainloader.batch_size, vocabulary, real_sentences, pred_sentences)
        
        # endfor batch 
        
        #if wandb.run is not None:
        wandb.log({"epoch":epoch, "loss":np.mean(losses), "ctcloss":np.mean(ct_l_s),"csim":np.mean(csim_s)})
        
        # save the model
        if epoch%1 == 0:
            val_accuracy = test(model, valloader, vocabulary, ctc_loss)
            wandb.log({"val_loss":val_accuracy})

        if epoch%5 == 0:
            torch.save(model.state_dict(), "models/model"+str(modeltitle)+".pt")
            

        if epoch%10 == 0:
            save_results(f"./results/results_{epoch}.txt", real_sentences, pred_sentences, overwrite=True)

    return

def test(model, valloader, vocabulary, ctc_loss):
    model.eval()

    real_sentences = []
    pred_sentences = []
    losses = []
    with torch.no_grad():

        for landmarks, len_landmark, label, len_label, audio in valloader:

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

            output, _, _ = model(landmarks, len_landmark)
            output = output.permute(1, 0, 2)
            # scrittura nel file del outuput e della frase originale
            loss = ctc_loss(torch.nn.functional.log_softmax(output, dim=2), label, len_landmark, len_label)
            losses.append(loss.item())

            real_sentences, pred_sentences = write_results(len_label, label_list, output.detach(), valloader.batch_size, vocabulary, real_sentences, pred_sentences)        

        print(":>",np.mean(losses))
        pred_sentences = list(map(lambda x:process_string(x),pred_sentences))
        save_results(f"./results/validation.txt", real_sentences, pred_sentences, overwrite=True)

    model.train()
    return np.mean(losses)

model_pipeline()