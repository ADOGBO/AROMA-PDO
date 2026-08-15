import os
import random
import shelve
from pathlib import Path

import einops
import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# import xarray as xr
from scipy import io
from torch.utils.data import Dataset

class Dataset_Burgers(Dataset):
    def __init__(self,data,x_dim,x_min=0,x_max=16,t_min=0,t_max=4,t_train_max=25,T_max=250):
        super().__init__()
        """
        for Burgers equation :
        data (numpy): Dataset values, with shape (N T Dx). Where N is the
            number of trajectories, Dx the size of the first spatial dimension, and T the
            number of timestamps.
        
        """
        assert T_max%t_train_max ==0, f"{T_max} must be  divisible by t_train_max"
        data=torch.tensor(data,dtype=torch.float32).unsqueeze(-1) #shape (N T Dx C) where C channel (always 1)
        self.data=rearrange(data, "N (n T) Dx C -> (N n) T Dx C",T=t_train_max) #shape (N T Dx C) where C channel (always 1)

        self.num_traj=data.shape[0]
        self.t_resolution=data.shape[1]
        if x_dim==1:
            self.x_resolution=data.shape[2]

            self.x_space=torch.linspace(x_min,x_max,self.x_resolution).view(-1,1)
            self.x_space_expand=self.x_space.expand(*self.data.shape)
        


    def __len__(self):
        return len(self.data)

    def __getitem__(self,ind):
        return (self.x_space_expand[ind], self.data[ind])