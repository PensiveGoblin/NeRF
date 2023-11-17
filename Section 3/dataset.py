import torch
from torch.utils.data import DataLoader
import numpy as np

import os
import imageio

import matplotlib.pyplot as plt

def get_rays(datapath, mode="train"):

    pose_file_names=os.listdir(datapath + f"/{mode}/pose")
    intrinsics_file_names=os.listdir(datapath + f"/{mode}/intrinsics")
    img_file_names=[f for f in os.listdir(datapath + "/imgs") if mode in f]

    assert len(pose_file_names) == len(intrinsics_file_names)
    assert len(pose_file_names) == len(img_file_names)

    N=len(pose_file_names)
    poses=np.zeros((N,4,4))
    intrinsics=np.zeros((N,4,4))

    images=[]

    for i in range(N):
        name = pose_file_names[i]

        pose=open(datapath + f"/{mode}/pose/" + name).read().split()
        poses[i]=np.array(pose, dtype=np.float32).reshape(4,4)

        name = intrinsics_file_names[i]
        intrinsic=open(datapath + f"/{mode}/intrinsics/" + name).read().split()
        intrinsics[i]=np.array(intrinsic, dtype=np.float32).reshape(4,4)

        name = img_file_names[i]
        img = imageio.imread(datapath + "/imgs/" + name) /255. # we normalize the image to [0,1] because conventions

        images.append(img[None, ...])

    images=np.concatenate(images, axis=0)
    print(images.shape)

    H=images.shape[1]
    W=images.shape[2]

    if images.shape[3]==4:
        images=images[..., :3]*images[..., -1:] + (1.-images[..., -1:]) # empty areas are (0,0,0,0) and would become black.
        #We want them to be white (or whatever the background color is), so we add the background color to the image.
        # notice the double dots, it adds a dimension of size 1 to the end of the array that is useful for operation

    # Now we want to create the rays
    rays_o=np.zeros((N,H*W,3)) #ray origin for every pixel, 3D
    rays_d=np.zeros((N,H*W,3)) #ray direction for every pixel, 3D
    target_px_values=images.reshape((N,H*W,3)) # we flatten the image


    for i in range(N):
        u = np.arange(W) #u is the x axis
        v = np.arange(H) #v is the y axis

        # we create a grid of coordinates
        u,v=np.meshgrid(u,v)

        dirs = np.stack([(u-intrinsics[i,0,2])/intrinsics[i,0,0], #x
                         -(v-intrinsics[i,1,2])/intrinsics[i,1,1], #y
                         -np.ones_like(u)], axis=-1) # size (H,W,3)
        
        dirs = np.matmul(poses[i, :3, :3], dirs[..., None]).squeeze(-1) # we rotate the directions

        dirs=dirs/np.linalg.norm(dirs, axis=-1, keepdims=True)
        rays_d[i]=dirs.reshape(-1,3)
        rays_o[i]+=poses[i, :3, -1]

    return rays_o, rays_d, target_px_values