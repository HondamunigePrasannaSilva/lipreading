
import torch
import torch.nn.functional as F
import wandb
from data.vocaset import *
from utils import *
from model_experiment import *
import lstmDecoder
import tqdm 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hyper = {
    'LANDMARK_DIM' : 68,
    'INPUT_DIM' : 68*3,
    'HID_DIM' : 128,
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
    model = lstmDecoder.Transformer_test2(len(vocabulary)).to(device)
    #m = torch.compile(model)

    # Define the CTC loss function
    ctc_loss = nn.CTCLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    return model, ctc_loss, optimizer,trainloader, valloader, vocabulary


def gigio(model):
    model_emb = lstmDecoder.MLP_emb().to(device)
    optimizer = optim.Adam(model_emb.parameters(), lr=3e-2)
    mse_loss = nn.MSELoss()

    trainset = vocadataset("train", landmark=True, onlyAudio=True)
    #valset = vocadataset("val", landmark=True)
    valset = vocadataset("val", landmark=True, onlyAudio=True)
    
    trainloader = DataLoader(trainset, batch_size=1, collate_fn=collate_fn, num_workers=8, shuffle=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=1, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    #Load wav2vec2
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav2vec = bundle.get_model().to(device)
    sample_rate = 22000
    wav2vec.eval()

    def train_emb(model, model_emb, mse_loss, optimizer,trainloader,valloader, epochs, modeltitle="exp"):

        model.train()
        # Training loop

        for epoch in range(epochs):

            losses = []
            progress_bar = tqdm.tqdm(total=len(trainloader), unit='step')
            for landmarks, len_landmark, _, _, audio in trainloader:
                # reshape the batch from [batch_size, frame_size, num_landmark, 3] to [batch_size, frame_size, num_landmark * 3] 
                landmarks = torch.reshape(landmarks, (landmarks.shape[0], landmarks.shape[1], landmarks.shape[2]*landmarks.shape[3]))

                # move data to GPU!
                landmarks = landmarks.to(device)
                len_landmark = len_landmark.to(device)
                audio = audio.to(device)

                if sample_rate != bundle.sample_rate:
                    audio = torchaudio.functional.resample(audio, sample_rate, bundle.sample_rate)

                with torch.inference_mode():
                    audio_features, _ = wav2vec.extract_features(audio[0])

                audio_input = audio_features[-1].clone().requires_grad_()
                #len_audio[0] = audio_input.shape[1]

                optimizer.zero_grad()

                output = model_emb(landmarks)
                
                with torch.no_grad():
                    _, features = model(landmarks,len_landmark, audio_input)

                loss = mse_loss(output, features)
                loss.backward()
                
                optimizer.step()

                losses.append(loss.item())

                #progress bar stuff
                progress_bar.set_description(f"emb:Epoch {epoch+1}/{epochs}")
                #progress_bar.set_postfix(loss=loss.item())  # Update the loss value
                progress_bar.set_postfix(loss=np.mean(losses))  # Update the loss value
                progress_bar.update(1)
        
        torch.save(model_emb.state_dict(), "models/model_emb"+str(modeltitle)+".pt")

    def test_emb(model, model_emb, valloader, mse_loss):
            losses = []
            with torch.no_grad():

                for landmarks, len_landmark, _, _, audio in valloader:

                    # reshape the batch from [batch_size, frame_size, num_landmark, 3] to [batch_size, frame_size, num_landmark * 3] 
                    landmarks = torch.reshape(landmarks, (landmarks.shape[0], landmarks.shape[1], landmarks.shape[2]*landmarks.shape[3]))

                    # move data to GPU!
                    landmarks = landmarks.to(device)
                    len_landmark = len_landmark.to(device)
                    audio = audio.to(device)

                    audio_features, _ = wav2vec.extract_features(audio[0])
                    audio_input = audio_features[-1].clone()

                    output = model_emb(landmarks)
                    _, features = model(landmarks,len_landmark, audio_input)


                    # scrittura nel file del outuput e della frase originale
                    loss = mse_loss(output, features)
                    losses.append(loss.item())

                print("validation emb:",np.mean(losses))

    train_emb(model, model_emb, mse_loss, optimizer,trainloader,valloader, 5)
    test_emb(model, model_emb, valloader, mse_loss)


# Function to train a model.
def train(model, ctc_loss, optimizer,trainloader, vocabulary, config,valloader, modeltitle= "_transformer_decoder"):
    
    #telling wand to watch
    #if wandb.run is not None:
    wandb.watch(model, optimizer, log="all", log_freq=320)
    #Load wav2vec2
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav2vec = bundle.get_model().to(device)
    sample_rate = 22000
    model.train()
    # Training loop

    for epoch in range(config.EPOCHS):

        #list to save sentences
        real_sentences = []
        pred_sentences = []
        losses = []
        progress_bar = tqdm.tqdm(total=len(trainloader), unit='step')
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

            output, _ = model(landmarks,len_landmark, audio_input)
            output = output.permute(1, 0, 2)#had to permute for the ctc loss. it acceprs [seq_len, batch_size, "num_class"]

            loss = ctc_loss(torch.nn.functional.log_softmax(output, dim=2), label, len_landmark, len_label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            losses.append(loss.item())

            #progress bar stuff
            progress_bar.set_description(f"Epoch {epoch+1}/{config.EPOCHS}")
            #progress_bar.set_postfix(loss=loss.item())  # Update the loss value
            progress_bar.set_postfix(loss=np.mean(losses))  # Update the loss value
            progress_bar.update(1)
            if epoch%1 == 0:
                real_sentences, pred_sentences = write_results(len_label, label_list, output.detach(), trainloader.batch_size, vocabulary, real_sentences, pred_sentences)
        
        # endfor batch
        
        #if wandb.run is not None:
        wandb.log({"epoch":epoch, "loss":np.mean(losses)})
        
        # save the model
        if epoch%1 == 0:
            val_accuracy = test(model, valloader, vocabulary, ctc_loss)
            wandb.log({"val_loss":val_accuracy})

        if epoch%5 == 0:
            torch.save(model.state_dict(), "models/model"+str(modeltitle)+".pt")
            

        if epoch%1 == 0:
            save_results(f"./results/results_{epoch}.txt", real_sentences, pred_sentences, overwrite=True)

        """if epoch % 5 == 0:
            gigio(model)"""

    return

def test(model, valloader, vocabulary, ctc_loss):
    model.eval()

    real_sentences = []
    pred_sentences = []
    losses = []

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav2vec = bundle.get_model().to(device)
    sample_rate = 22000
    wav2vec.eval()
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
            audio = audio.to(device)

            audio_features, _ = wav2vec.extract_features(audio[0])
            audio_input = audio_features[-1].clone()

            output, _ = model(landmarks,len_landmark, audio_input)

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