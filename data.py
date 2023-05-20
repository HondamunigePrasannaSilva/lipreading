import torch
import torchvision
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class vocadataset(Dataset):

    def __init__(self, type="train", landmark = True):
        """
            - type : "test", "train" or "val".Default type = "train"
            - landmark: if True returns landmark, if false returns the vertex. Default = landmark = True
        """
        # read vertex from data_verts.npy file and seq to index from seq_to_idx
        self.face_vert_mmap = np.load("dataset/data_verts.npy", mmap_mode='r+')
        self.face_vert =  torch.from_numpy(self.face_vert_mmap)
        self.seq_index = pickle.load(open("dataset/subj_seq_to_idx.pkl", 'rb'))
        
        #get voice names and labels = sentences
        self.keys = list(self.seq_index)
        self.labels = self.getlabels()
        
        # set seed to 0 to select train/val/test set
        random.seed(0)
        self.index = random.sample(list(range(1,11)), k=4) # sample 4 index, [val_1, val_2, test_1, test_2]

        self.type = type

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
                l.append(f.readline())

            d[kl[i]] = l
        
        return d

    def getVertex(self, index, type = "train"):
        """
            method that returns vertex given the type and index
        """
        # get list of seq index!
        idx = self.trainIndex[index]
        voice_idx, sentence_idx = int(idx/40), idx%40
        si = list(self.seq_index[self.keys[voice_idx]])

        # get seq index of the voice and trasform it in a list
        seq_idx = list(self.seq_index[self.keys[voice_idx]][si[sentence_idx]].values())

        




        # get list of vertex

        # return the vertex


        return 0
    
    def getLabel(self, index, type = "train"):
        """
            method that returns the label(sentence) given the type and index
        """

        return 0
    
    def getLandmark(self, vertex):
        """
            method that return the landmarks given the vertex
        """


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

        
        if(self.type == "train"):
            #prova

            return
        elif(self.type == "test"):
            return
        elif(self.type == "val"):
            return
        else:
            print("Error: type should be: train, test or val")
            return


    def __len__(self):

        if self.type == "train":
            train_idx = [item for item in list(range(1,13)) if item not in self.index] #get voice index of train
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




    
