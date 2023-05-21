import numpy as np
import torch
import pickle
import trimesh


face_vert_mmap = np.load("dataset/data_verts.npy", mmap_mode='r+', allow_pickle=False, max_header_size=30000)
x = torch.from_numpy(face_vert_mmap)

data2array_verts = pickle.load(open("dataset/subj_seq_to_idx.pkl", 'rb'))


v  = trimesh.load("dataset/FaceTalk_170725_00137_TA.ply", process=False)
#prova = pickle.load(open("dataset/FaceTalk_170725_00137_TA.ply", 'rb'))

kl = list(data2array_verts)
d = {}
for i in range(len(kl)):
    f = open("dataset/labels/"+str(kl[i])+".txt", "r")
    l = []
    for j in range(40):
        l.append(f.readline())

    d[kl[i]] = l




print("ciao")