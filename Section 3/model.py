import torch
from torch import nn


class Voxels(nn.Module):

    def __init__(self, nb_voxels=100, scale=1, device="cpu"):
        super(Voxels, self).__init__()

        self.voxels = torch.nn.Parameter(torch.rand((nb_voxels, nb_voxels, nb_voxels, 4), device=device))

        self.nb_voxels = nb_voxels
        self.device = device
        self.scale=scale
    
    def forward(self, xyz):
        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]

        cond = (x.abs()<self.scale/2) & (y.abs()<self.scale/2) & (z.abs()<self.scale/2) #The gris is centered on (0,0,0)

        indx = ((x[cond]+self.scale/2) * self.nb_voxels/self.scale).type(torch.long)
        indy = ((y[cond]+self.scale/2) * self.nb_voxels/self.scale).type(torch.long)
        indz = ((z[cond]+self.scale/2) * self.nb_voxels/self.scale).type(torch.long)

        colors_and_densities = torch.zeros((xyz.shape[0], 4)).to(self.device)
        colors_and_densities[cond, :3] = self.voxels[indx, indy, indz, :3]
        colors_and_densities[cond, 3] = self.voxels[indx, indy, indz, 3]*10

        return torch.sigmoid(colors_and_densities[:, :3]), torch.relu(colors_and_densities[:, 3])
    
    def intersect(self,x):
        return self.forward(x)
