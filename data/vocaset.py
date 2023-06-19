import torch
import torchvision
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import trimesh
from data.getlandmark import *
import h5py

class vocadataset(Dataset):

    def __init__(self, type_="train", landmark = True, index=None, mouthOnly = False, savelandmarks = False, onlyAudio = False):
        
        """
            - type : "test", "train" or "val".Default type = "train"
            - landmark: if True returns landmark, if false returns the vertex. Default = landmark = True
            - index: test and validation voice index. Default=None, it's selected randomly!
            - mouthOnly: get only mouth landmark!
        """
        self.onlyAudio = onlyAudio
        # read vertex from data_verts.npy file and seq to index from seq_to_idx
        self.face_vert_mmap = np.load("dataset/data_verts.npy", mmap_mode='r+')
        self.face_vert =  torch.from_numpy(self.face_vert_mmap)
        self.seq_index = pickle.load(open("dataset/subj_seq_to_idx.pkl", 'rb'))
        self.audio_processed = pickle.load(open("dataset/processed_audio_deepspeech.pkl", 'rb'), encoding='latin1')
        
        self.audio_processed['FaceTalk_170811_03274_TA']['sentence01'] = self.audio_processed['FaceTalk_170811_03274_TA']['sentence03']
        self.audio_processed['FaceTalk_170811_03274_TA']['sentence02'] = self.audio_processed['FaceTalk_170811_03274_TA']['sentence03']
        self.audio_processed['FaceTalk_170811_03274_TA']['sentence24'] = self.audio_processed['FaceTalk_170811_03274_TA']['sentence03']


        self.audio_processed['FaceTalk_170912_03278_TA']['sentence11'] = self.audio_processed['FaceTalk_170912_03278_TA']['sentence01']
        
        self.audio_processed['FaceTalk_170913_03279_TA']['sentence12'] = self.audio_processed['FaceTalk_170913_03279_TA']['sentence02']
        self.audio_processed['FaceTalk_170913_03279_TA']['sentence38'] = self.audio_processed['FaceTalk_170913_03279_TA']['sentence02']

        self.audio_processed['FaceTalk_170809_00138_TA']['sentence32'] = self.audio_processed['FaceTalk_170809_00138_TA']['sentence01']



        #get voice names and labels = sentences
        self.keys = list(self.seq_index)
        self.labels = self.getlabels()
        self.mouthonly = mouthOnly
        
        # fill the missing data by copying the first sentence and vertex
         
        self.labels['FaceTalk_170811_03274_TA'][0] = self.labels['FaceTalk_170811_03274_TA'][2]
        self.labels['FaceTalk_170811_03274_TA'][1] = self.labels['FaceTalk_170811_03274_TA'][2]
        self.labels['FaceTalk_170811_03274_TA'][23] = self.labels['FaceTalk_170811_03274_TA'][2]
        self.seq_index['FaceTalk_170811_03274_TA']['sentence24'] = self.seq_index['FaceTalk_170811_03274_TA']['sentence03']
        self.seq_index['FaceTalk_170811_03274_TA']['sentence01'] = self.seq_index['FaceTalk_170811_03274_TA']['sentence03']
        self.seq_index['FaceTalk_170811_03274_TA']['sentence02'] = self.seq_index['FaceTalk_170811_03274_TA']['sentence03']
        #
        self.labels['FaceTalk_170912_03278_TA'][10] = self.labels['FaceTalk_170912_03278_TA'][0]
        self.seq_index['FaceTalk_170912_03278_TA']['sentence10'] = self.seq_index['FaceTalk_170912_03278_TA']['sentence01']
        #
        self.labels['FaceTalk_170913_03279_TA'][11] = self.labels['FaceTalk_170913_03279_TA'][0]
        self.labels['FaceTalk_170913_03279_TA'][36] = self.labels['FaceTalk_170913_03279_TA'][0]
        self.seq_index['FaceTalk_170913_03279_TA']['sentence11'] = self.seq_index['FaceTalk_170913_03279_TA']['sentence01']
        self.seq_index['FaceTalk_170913_03279_TA']['sentence36'] = self.seq_index['FaceTalk_170913_03279_TA']['sentence01']
        #
        self.labels['FaceTalk_170809_00138_TA'][31] = self.labels['FaceTalk_170809_00138_TA'][0]
        self.seq_index['FaceTalk_170809_00138_TA']['sentence32'] = self.seq_index['FaceTalk_170809_00138_TA']['sentence01']


        # test and validation index can be set manually
        if index is not None:
            self.index = index
        else:
            random.seed(0) # set seed to 0 to select train/val/test set
            
            self.index = random.sample(list(range(0,12)), k=4) # sample 4 index [test_1, test_2, val_1, val_2]
            print(self.index[0])

        self.type = type_
        self.landmark = landmark

        self.trainIndex, self.testIndex, self.valIndex = self.getTrainIndex_Mixed()# if you do not want to mixed index getTrainIndex

        # Save landmarks
        if savelandmarks is not False:
            self.landmarks, self.landmark_lens = self.createLandmarkTrain()
        else:
            self.landmarks, self.landmark_lens = None, None

        # read file to get only mouth vertex
        file = h5py.File('dataset/mouthIdx_CoMa.mat', 'r')
        self.idxInsideMouth = file['idxInsideMouth'][0]


    def getlabels(self):
        """
            - method that returns a dict. key:voice_name, value: sentence, 40 sentence for each voice
        """
        kl = list(self.seq_index)   # get key(voice) names
        d = {}
        for i in range(len(kl)):
            f = open("dataset/labels/"+str(kl[i])+".txt", "r")
            l = []
            for j in range(40): #TODO 40 hardcoded, fixme!
                sent = f.readline()
                sent = sent.lower().replace('\n', '')
                # TODO refactor this shit!
               # sent = sent.replace('.', '')
               # sent = sent.replace('?', '')
               # sent = sent.replace(',', '')
               # sent = sent.replace('!', '')
               # sent = sent.replace("’", ' ')
               # sent = sent.replace("'", ' ')
               # sent = sent.replace(":", '')
               # sent = sent.replace(";", '')
               # sent = sent.replace("-", '')

                l.append(sent)

            d[kl[i]] = l
        
        return d
    
    def getVoice_Sentence_Index(self, index, type = 'train'):
        """
            method that returns vertex given the type and index
        """
        # get list of seq index!
        if(type == "train"):
            idx = self.trainIndex[index]
        elif(type == "test"):
            idx = self.testIndex[index]
        elif(type == "val"):
            idx = self.valIndex[index]
        else:
            print("Type must be: train, test or val")
            return

        voice_idx, sentence_idx = int(idx/40), idx%40
        #print(self.keys[voice_idx])
        sentence_idx += 1   # sentence name start from 01 not from 00
        if(sentence_idx < 10):
            sentence_idx = f"sentence0{sentence_idx}"
        else:
            sentence_idx = f"sentence{sentence_idx}"

        return self.keys[voice_idx], sentence_idx
    
    def getVertex(self, index, type = "train"):
        """
            method that returns vertex given the type and index
        """
        # get list of seq index!
        if(type == "train"):
            idx = self.trainIndex[index]
        elif(type == "test"):
            idx = self.testIndex[index]
        elif(type == "val"):
            idx = self.valIndex[index]
        else:
            print("Type must be: train, test or val")
            return

        voice_idx, sentence_idx = int(idx/40), idx%40
        #print(self.keys[voice_idx])
        sentence_idx += 1   # sentence name start from 01 not from 00
        if(sentence_idx < 10):
            sentence_idx = f"sentence0{sentence_idx}"
        else:
            sentence_idx = f"sentence{sentence_idx}"

        # get seq index of the voice and trasform it in a list
        seq_idx = list(self.seq_index[self.keys[voice_idx]][sentence_idx].values())

        # get list of vertex
        vertex = self.face_vert[seq_idx]

        if (self.landmark == False) and (self.mouthonly==True):
            vertex = vertex[:,self.idxInsideMouth,:]
        
        return vertex
    
    def getLabel(self, index, type = "train"):
        """
            method that returns the label(sentence) given the type and index
        """
        # get list of seq index!
        if(type == "train"):
            idx = self.trainIndex[index]
        elif(type == "test"):
            idx = self.testIndex[index]
        elif(type == "val"):
            idx = self.valIndex[index]
        else:
            print("Type must be: train, test or val")
            return
        
        voice_idx, sentence_idx = int(idx/40), idx%40
        sentence = self.labels[self.keys[voice_idx]][sentence_idx]

        return sentence  #"#"+sentence
         
    def getLandmark(self, vertex, index, type):
        """
            method that return the landmarks given the vertex
        """
        # get list of seq index!
        if(type == "train"):
            idx = self.trainIndex[index]
        elif(type == "test"):
            idx = self.testIndex[index]
        elif(type == "val"):
            idx = self.valIndex[index]
        else:
            print("Type must be: train, test or val")
            return
        
        voice_idx = int(idx/40)   
        voice_name = self.keys[voice_idx]

        v  = trimesh.load(f"dataset/mesh/{voice_name}.ply", process=False)

        #Create empty tensor to save the landmarks        
        landmarks = torch.Tensor(size=[vertex.shape[0], 68, 3])

        # Trasform vertex in landmark! Heavy duty!
        for i in range(vertex.shape[0]):
            landmarks[i] = torch.from_numpy( get_landmarks(vertex[i],v) )
            
        return landmarks 
    
    def createLandmarkTrain(self,):

        if(self.type == "train"):
            dim = len(self.trainIndex)
        if(self.type == "test"):
            dim = len(self.testIndex)
        if(self.type == "val"):
            dim = len(self.valIndex)
        train_frame_size = torch.Tensor( size=[dim,2,1])
        count = 0
        for i in range(dim):
            vertex = self.getVertex(i, type = self.type)
            
            if i > 0:
                ll =  self.getLandmark(vertex, i, type=self.type)
                train_frame_size[i][0] = ll.shape[0]    
                landmark = torch.cat( (landmark,ll) , dim=0 ) 
            if i == 0:
               landmark = self.getLandmark(vertex, i, type=self.type)
               train_frame_size[i][0] = landmark.shape[0]
            
            
            train_frame_size[i][1] = count
            # [sum_n_frame, 68, 3]
            count = landmark.shape[0]
        
        return landmark, train_frame_size

    def getSavedLandmarksTrain(self, index):
        start_index = int(self.landmark_lens[index][1])
        final_index = start_index + int(self.landmark_lens[index][0])
        
        return self.landmarks[start_index:final_index]

    def getOnlyMouthlandmark(self, landmarks):
        """
            - Function to get only landmark that involves mouth movments!
        """
        # selecting the landmaark that involves mouth movments!
        l = [i for i in range(1,18)]+[i for i in range(49,68)]
        
        #return landmarks[:,l,0:2]  # to get only x and y
        return landmarks[:,l,:]

    def getTrainIndex_Mixed(self):
        """
            - Method that return the index of train, test and validation set
        """
        #rember the set the seed
        random.seed(0)

        label_len = 12*40
        result_list = []
        result_list.extend(list(range(label_len)))
        # first 80 is for validation and the last random is for testing 
        random_subset = random.sample(result_list, 160)
        # train list
        train_index = [num for num in result_list if num not in random_subset]

        test_index = random_subset[0:80]
        val_index = random_subset[80:]

        return train_index, test_index, val_index
    
    def getTrainIndex(self):
        """
            - Method that return the index of train, test and validation set
        """
        label_len = self.getlen("train")+self.getlen("test")+self.getlen("val")
        result_list = []

        #get index of test and validation
        for num in self.index:
            result_list.extend(list(range(num*40, (num+1)*40, 1)))

        train_index  = [item for item in list(range(label_len)) if item not in result_list]

        test_index = result_list[0:(1*40)]
        val_index = result_list[(1*40):]

        return train_index, test_index, val_index

    def getAudio(self,index, type = "train"):
        
        # get facetalk and sentence index given the global index
        faceTalk, sentence = self.getVoice_Sentence_Index(index, type)

        def getAudioInterval(audio):
            list_ = []
            for i in range(len(audio)):
                list_ = list_+ audio[i].tolist()
            
            return torch.tensor(list_)[None,:]

        #get length of the audio
        audio = self.audio_processed[faceTalk][sentence]['audio']

        len_audio = len(audio)

        for i in range(len_audio):
            if i == 0:
                audio_ = getAudioInterval(audio[i])
            else:
                audio_ = torch.cat([audio_, getAudioInterval(audio[i])], dim = 0)
        
        return audio_[None, :, :] # size [1, len_landmark, audio_interval]
    
    
    def __getitem__(self, index):
        
        label = self.getLabel(index, self.type)
        audio =  self.getAudio(index, type = self.type)

        

        if (self.landmark == True) and (self.landmarks is not None):
            lan = self.getSavedLandmarksTrain(index)
            return lan, label
          
        vertex = self.getVertex(index, self.type)

        if (self.onlyAudio == True) and (self.landmark == False):
            return audio, label

        if (self.onlyAudio == False) and (self.landmark == False):
            return vertex, label

        if (self.onlyAudio == False) and (self.landmark == True):
            landmark = self.getLandmark(vertex, index, self.type)
            if self.mouthonly == True:
                landmark = self.getOnlyMouthlandmark(landmark)
            return landmark, label

        if (self.onlyAudio == True) and (self.landmark == True):
            landmark = self.getLandmark(vertex, index, self.type)
            return landmark, audio, label
    
    def getlen(self, type):
        if type == "train":
            train_idx = [item for item in list(range(0,12)) if item not in self.index] #get voice index of train
            count = 0
            for i in train_idx:
                count += len(self.labels[self.keys[i]])   # count num of sentence
        elif type == "test":
            count = 0
            for i in self.index[:2]:
                count += len(self.labels[self.keys[i]])   # count num of sentence
        elif type == "val":
            count = 0
            for i in self.index[2:]:
                count += len(self.labels[self.keys[i]])   # count num of sentence
        return count

    def __len__(self):
        
        if self.type == "train":
            train_idx = [item for item in list(range(0,12)) if item not in self.index] #get voice index of train
            count = 0
            for i in train_idx:
                count += len(self.labels[self.keys[i]])   # count num of sentence
        elif self.type == "test":
            count = 0
            for i in self.index[:2]:
                count += len(self.labels[self.keys[i]])   # count num of sentence
        elif self.type == "val":
            count = 0
            for i in self.index[2:]:
                count += len(self.labels[self.keys[i]])   # count num of sentence
        else:
            print("Error: type should be: train, test or val")
            return

        return count


def collate_fn(batch):

    data_batch, label_batch = zip(*batch)

    # Get the sequences and their lengths
    sequences = [sample for sample in batch]
    lengths = torch.tensor([sample[0].size(0) for sample in batch])
    #audio = batch[0][2]

    lengths_labels = [len(item) for item in label_batch]
    padded_labels = [item + item[len(item)-1] * (max(lengths_labels) - len(item)) for item in label_batch]
    #padded_labels = torch.tensor(padded_labels)

    # Find the maximum length in the batch
    max_length = max(lengths)

    repeated_sequences = []
    for sample in sequences:
        lastLandmark = sample[0][-1:,:]
        repeated_lm = lastLandmark.repeat(max_length-sample[0].shape[0], 1, 1)
        result_tensor = torch.cat((sample[0], repeated_lm), dim=0)
        repeated_sequences.append(result_tensor)


    # Stack the repeated sequences
    padded_sequences = torch.stack(repeated_sequences, dim=0)

    lengths_labels = torch.tensor(lengths_labels, dtype=torch.long)
    #lengths = torch.tensor(lengths)

    return padded_sequences,lengths, padded_labels, lengths_labels#, audio




