import torch
import torchvision
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import trimesh
from data.getlandmark import *


# TODO: migliore getindex!

class vocadataset(Dataset):

    def __init__(self, type="train", landmark = True, index=None):
        """
            - type : "test", "train" or "val".Default type = "train"
            - landmark: if True returns landmark, if false returns the vertex. Default = landmark = True
            - index: test and validation voice index. Default=None, it's selected randomly!
        """
        # read vertex from data_verts.npy file and seq to index from seq_to_idx
        self.face_vert_mmap = np.load("dataset/data_verts.npy", mmap_mode='r+')
        self.face_vert =  torch.from_numpy(self.face_vert_mmap)
        self.seq_index = pickle.load(open("dataset/subj_seq_to_idx.pkl", 'rb'))
        
        #get voice names and labels = sentences
        self.keys = list(self.seq_index)
        self.labels = self.getlabels()
        
        # test and validation index can be set manually
        if index is not None:
            self.index = index
        else:
            random.seed(0) # set seed to 0 to select train/val/test set
            self.index = random.sample(list(range(0,12)), k=4) # sample 4 index [test_1, test_2, val_1, val_2]

        #print(self.index)
        self.type = type
        self.landmark = landmark

        self.trainIndex, self.testIndex, self.valIndex = self.getTrainIndex()
        
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
                l.append(f.readline().lower().replace('\n', ''))

            d[kl[i]] = l
        
        return d

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

        voice_idx, sentence_idx = int(idx/40), idx%40   #TODO: trovare un modo divero per accedere
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

        return sentence
         
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
        
        voice_idx = int(idx/40)   #TODO: trovare un modo divero per accedere
        voice_name = self.keys[voice_idx]

        v  = trimesh.load(f"dataset/mesh/{voice_name}.ply", process=False)
        landmarks = torch.Tensor(size=[vertex.shape[0], 68, 3])
        
        for i in range(vertex.shape[0]):
            landmarks[i] = torch.from_numpy( get_landmarks(vertex[i],v) )

        return landmarks

    def getTrainIndex(self):
        """
            - Method that return the index of train, test and validation set
        """
        label_len = self.__len__("train")+self.__len__("test")+self.__len__("val")
        result_list = []

        #get index of test and validation
        for num in self.index:
            result_list.extend(list(range(num*40, (num+1)*40, 1)))

        train_index  = [item for item in list(range(label_len)) if item not in result_list]

        test_index = result_list[0:(2*40)]
        val_index = result_list[(2*40):]

        return train_index, test_index, val_index
        
    def __getitem__(self, index):
        
        vertex = self.getVertex(index, self.type)
        label = self.getLabel(index, self.type)
        
        if self.landmark == False:
            return vertex, label
        else:
            landmark = self.getLandmark(vertex, index, self.type)
            return landmark, label

    def __len__(self, type):

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
        else:
            print("Error: type should be: train, test or val")
            return

        return count
    
