
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
    'SERVER':'Local'
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
    model = MLP_emb().to(device)
    #m = torch.compile(model)

    # Define the CTC loss function
    mse_loss = nn.MSELoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    return model, mse_loss, optimizer,trainloader, valloader, vocabulary

# Function to train a model.
def train(model, mse_loss, optimizer,trainloader, vocabulary, config,valloader, modeltitle= "_embedding"):
    
    #telling wand to watch
    #if wandb.run is not None:
    #cate = nn.CrossEntropyLoss()
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

            audio_input = lstmDecoder.linear_interpolation(audio_input, 50, 60,output_len=len_landmark)

            optimizer.zero_grad()

            output = model(landmarks)

            loss = mse_loss(output, audio_input)
            loss.backward()
            
            optimizer.step()

            losses.append(loss.item())

            #progress bar stuff
            progress_bar.set_description(f"Epoch {epoch+1}/{config.EPOCHS}")
            #progress_bar.set_postfix(loss=loss.item())  # Update the loss value
            progress_bar.set_postfix(loss=np.mean(losses))  # Update the loss value
            progress_bar.update(1)
        
        # endfor batch 
        
        #if wandb.run is not None:
        wandb.log({"epoch":epoch, "loss":np.mean(losses)})
        
        # save the model
        if epoch%2 == 0:
            val_accuracy = test(model, valloader, vocabulary, mse_loss)
            wandb.log({"val_loss":val_accuracy})

        if epoch%5 == 0:
            torch.save(model.state_dict(), "models/model"+str(modeltitle)+".pt")

    return

def test(model, valloader, vocabulary, mse_loss):
    model.eval()

    real_sentences = []
    pred_sentences = []
    losses = []

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav2vec = bundle.get_model().to(device)
    wav2vec.eval()

    sample_rate = 22000

    model_audio = lstmDecoder.only_Decoder2(hyper_a['INPUT_DIM'], hyper_a['HID_DIM'], hyper_a['NUM_LAYERS'], len(vocabulary)).to(device)
    model_audio.load_state_dict(torch.load("./models/modeltest_24.pt"))
    model_audio.eval()

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

            if sample_rate != bundle.sample_rate:
                audio = torchaudio.functional.resample(audio, sample_rate, bundle.sample_rate)

            #with torch.inference_mode():
            audio_features, _ = wav2vec.extract_features(audio[0])

            audio_input = audio_features[-1].clone()

            audio_input = lstmDecoder.linear_interpolation(audio_input, 50, 60,output_len=len_landmark)

            outputs = model(landmarks)
            # scrittura nel file del outuput e della frase originale
            loss = mse_loss(outputs, audio_input)
            losses.append(loss.item())
            out_test = model_audio(outputs, len_landmark)
            out_test = out_test.permute(1, 0, 2)

            real_sentences, pred_sentences = write_results(len_label, label_list, out_test.detach(), valloader.batch_size, vocabulary, real_sentences, pred_sentences)        

            pred_sentences = list(map(lambda x:process_string(x),pred_sentences))
            save_results(f"./results/validation.txt", real_sentences, pred_sentences, overwrite=True)
            
        print(":>",np.mean(losses))

    model.train()
    return np.mean(losses)

model_pipeline()