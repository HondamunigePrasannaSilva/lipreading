## Modified from https://ringnet.is.tue.mpg.de.

import numpy as np
import trimesh
import _pickle as pickle
import matplotlib.pyplot as plt
import time
import torch

def load_static_embedding(static_embedding_path):
    with open(static_embedding_path, 'rb') as f:
        lmk_indexes_dict = pickle.load(f, encoding='latin1')
    lmk_face_idx = lmk_indexes_dict[ 'lmk_face_idx' ].astype( np.uint32 )
    lmk_b_coords = lmk_indexes_dict[ 'lmk_b_coords' ]
    return lmk_face_idx, lmk_b_coords

def mesh_points_by_barycentric_coordinates(mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords):
    # function: evaluation 3d points given mesh and landmark embedding
    # modified from https://github.com/Rubikplayer/flame-fitting/blob/master/fitting/landmarks.py
    dif1 = np.vstack([(mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)]).T
    return dif1

def load_dynamic_contour(vertices, faces, contour_embeddings_path='None', static_embedding_path='None', angle=0):
    contour_embeddings_path = contour_embeddings_path
    dynamic_lmks_embeddings = np.load(contour_embeddings_path, allow_pickle=True, encoding='latin1').item()
    lmk_face_idx_static, lmk_b_coords_static = load_static_embedding(static_embedding_path)
    lmk_face_idx_dynamic = dynamic_lmks_embeddings['lmk_face_idx'][angle]
    lmk_b_coords_dynamic = dynamic_lmks_embeddings['lmk_b_coords'][angle]
    dynamic_lmks = mesh_points_by_barycentric_coordinates(vertices, faces, lmk_face_idx_dynamic, lmk_b_coords_dynamic)
    static_lmks = mesh_points_by_barycentric_coordinates(vertices, faces, lmk_face_idx_static, lmk_b_coords_static)
    total_lmks = np.vstack([dynamic_lmks, static_lmks])

    return total_lmks 

def process_landmarks(landmarks):
        points = np.zeros(np.shape(landmarks))
        ## Centering
        mu_x = np.mean(landmarks[:, 0])
        mu_y = np.mean(landmarks[:, 1])
        mu_z = np.mean(landmarks[:, 2])
        mu = [mu_x, mu_y, mu_z]

        landmarks_gram=np.zeros(np.shape(landmarks))
        for j in range(np.shape(landmarks)[0]):
            landmarks_gram[j,:]= np.squeeze(landmarks[j,:])-np.transpose(mu)

        normFro = np.sqrt(np.trace(np.matmul(landmarks_gram, np.transpose(landmarks_gram))))
        land = landmarks_gram / normFro
        points[:,:]=land
        return points
    


def get_landmarks(vertices, template):
    angle = 0.0 #in degrees
    if angle < 0:
        angle = 39 - angle
    contour_embeddings_path = 'dataset/flame_dynamic_embedding.npy'
    static_embedding_path = 'dataset/flame_static_embedding.pkl'
    template_mesh = trimesh.load(template, process=False)
    faces=template_mesh.faces
    total_lmks=load_dynamic_contour(vertices, faces, contour_embeddings_path=contour_embeddings_path, static_embedding_path=static_embedding_path, angle=int(angle))
    ########## Add process Landmarks
    total_lmks=process_landmarks(total_lmks)


    return total_lmks





"""v  = trimesh.load("dataset/FaceTalk_170725_00137_TA.ply", process=False)
face_vert_mmap = np.load("dataset/data_verts.npy", mmap_mode='r+', allow_pickle=False, max_header_size=30000)
x = torch.from_numpy(face_vert_mmap)

land = get_landmarks(x[1000],v)

plt.scatter(land[:,0], land[:,1], marker='.')
plt.savefig("prova_2.png")
print("ciao")"""