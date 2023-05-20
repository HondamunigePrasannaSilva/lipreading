import numpy as np
import torch
import pickle

face_vert_mmap = np.load("dataset/data_verts.npy", mmap_mode='r+')
x = torch.from_numpy(face_vert_mmap)


data2array_verts = pickle.load(open("dataset/subj_seq_to_idx.pkl", 'rb'))

kl = list(data2array_verts)
d = {}
for i in range(len(kl)):
    f = open("dataset/labels/"+str(kl[i])+".txt", "r")
    l = []
    for j in range(40):
        l.append(f.readline())

    d[kl[i]] = l


print("ciao")
