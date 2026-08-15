
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
#from tqdm import tqdm
import random

from typing import List
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict
from pathlib import Path
import math
import h5py
import time
import re
import argparse

import os

def cycle(iterable):
    while True:
        for x in iterable: 
            yield x

def loss_model(mean,log_var_square,x,x_hat,beta,return_KL=False):
    """
         mean,log_var_square,x,x_hat : (bach,seq_length,d)
    
    """
    
    KL_divergence_per_batch=0.5*torch.mean( -1-log_var_square+mean**2+ torch.exp(log_var_square),dim=[1,2] )
    
    if return_KL==False:

        return beta*torch.mean(KL_divergence_per_batch)+F.mse_loss(x_hat,x)
    else:
        return KL_divergence_per_batch, beta*torch.mean(KL_divergence_per_batch)+F.mse_loss(x_hat,x)



def whole_save_checkpoint(state,tmp_path, final_path):
    """
    Sauvegarde un checkpoint de façon robuste sur HPC.
    
    state : Object avec model, optimizer, epoch, etc.
    final_path : chemin final du checkpoint (ex: 'checkpoint.pt')
    tmp_path : chemin temporaire du checkpoint (ex: 'checkpoint.tmp')
    """
    

    # 1️⃣ Sauvegarde dans fichier temporaire
    torch.save(state, tmp_path)

    # 2️⃣ Vérification simple : le fichier temporaire existe et a une taille > 0
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        raise RuntimeError(f"Checkpoint temporaire corrompu : {tmp_path}")

    # 3️⃣ Remplacement atomique du fichier final
    os.replace(tmp_path, final_path)
    print(f"Checkpoint sauvegardé avec succès : {final_path}")

#-------------------------------- CHECKPOINT FOR ENCODER-DECODER-------------------------------------
def save_checkpoint_encoDeco(state, tmp_path, final_path):

    checkpoint = {
        "model_enco": state.model_enco.module.state_dict(),
        "model_deco": state.model_deco.module.state_dict(),

        "optim_enco": state.optim_enco.state_dict(),
        "optim_deco": state.optim_deco.state_dict(),

        "scheduler_enco": state.scheduler_enco.state_dict(),
        "scheduler_deco": state.scheduler_deco.state_dict(),

        "epoch": state.epoch,
        "best_valid_loss": state.best_valid_loss,
    }

    
    # 1️⃣ Sauvegarde dans fichier temporaire
    torch.save(checkpoint, tmp_path)

    # 2️⃣ Vérification simple : le fichier temporaire existe et a une taille > 0
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        raise RuntimeError(f"Checkpoint temporaire corrompu : {tmp_path}")

    # 3️⃣ Remplacement atomique du fichier final
    os.replace(tmp_path, final_path)
    print(f"Checkpoint sauvegardé avec succès : {final_path}")

def unwrap(model):
    return model.module if hasattr(model, "module") else model

def load_checkpoint_encoDeco(state, path, device,mode="train"):
    """"
    Here state.model_enco and state.model_deco are not not DDP wrappers: only plain model state_dicts
    """
    checkpoint = torch.load(path, map_location=device)

    #state.model_enco.module.load_state_dict(checkpoint["model_enco"])
    #state.model_deco.module.load_state_dict(checkpoint["model_deco"])
    unwrap(state.model_enco).load_state_dict(checkpoint["model_enco"])
    unwrap(state.model_deco).load_state_dict(checkpoint["model_deco"])

    state.epoch = checkpoint["epoch"]
    state.best_valid_loss = checkpoint["best_valid_loss"]

    if mode=="train": # During evaluation we don't need the below variable
        state.optim_enco.load_state_dict(checkpoint["optim_enco"])
        state.optim_deco.load_state_dict(checkpoint["optim_deco"])

        state.scheduler_enco.load_state_dict(checkpoint["scheduler_enco"])
        state.scheduler_deco.load_state_dict(checkpoint["scheduler_deco"])

        

    return state

def save_checkpoint_DiT(state, tmp_path, final_path):

    checkpoint = {
        "model": state.model.module.state_dict(),

        "optim": state.optim.state_dict(),

        "scheduler": state.scheduler.state_dict(),

        "epoch": state.epoch,
        "best_valid_loss": state.best_valid_loss,
    }

    
    # 1️⃣ Sauvegarde dans fichier temporaire
    torch.save(checkpoint, tmp_path)

    # 2️⃣ Vérification simple : le fichier temporaire existe et a une taille > 0
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        raise RuntimeError(f"Checkpoint temporaire corrompu : {tmp_path}")

    # 3️⃣ Remplacement atomique du fichier final
    os.replace(tmp_path, final_path)
    print(f"Checkpoint sauvegardé avec succès : {final_path}")


def load_checkpoint_DiT(state, path, device,mode="train"):
    """"
    Here state.model is not not DDP wrappers: only plain model state_dicts
    """
    checkpoint = torch.load(path, map_location=device)

    #state.model_enco.module.load_state_dict(checkpoint["model_enco"])
    #state.model_deco.module.load_state_dict(checkpoint["model_deco"])
    unwrap(state.model).load_state_dict(checkpoint["model"])
    
    state.epoch = checkpoint["epoch"]
    state.best_valid_loss = checkpoint["best_valid_loss"]


    if mode=="train": # During evaluation we don't need the below variable
        state.optim.load_state_dict(checkpoint["optim"])

        state.scheduler.load_state_dict(checkpoint["scheduler"])

        
    return state



class StateMetrics(object):
    def __init__(self,model_m,model_l=None,model_a=None,optim_m=None,optim_l=None,optim_a=None,scheduler_m=None,scheduler_l=None,scheduler_a=None):
        self.model_mean=model_m
        self.model_last=model_l
        self.model_attn=model_a

        self.optim_mean=optim_m
        self.optim_last=optim_l
        self.optim_attn=optim_a

        self.scheduler_mean=scheduler_m
        self.scheduler_last=scheduler_l
        self.scheduler_attn=scheduler_a

        self.epoch=0
        self.best_valid_loss=np.inf




# A changer
class State(object):
    def __init__(self,model_enco, model_deco, optim_enco, optim_deco,scheduler_enco,scheduler_deco):
        self.model_enco=model_enco
        self.model_deco=model_deco

        self.optim_deco=optim_deco
        self.optim_enco=optim_enco

        self.scheduler_enco=scheduler_enco
        self.scheduler_deco=scheduler_deco

        self.epoch=0
        self.best_valid_loss=np.inf

class State_DiT(object):
    def __init__(self,model,optim,scheduler):
        self.model=model
        self.optim=optim
        self.scheduler=scheduler
        self.epoch=0
        self.best_valid_loss=np.inf

class CheckNaN(object):
    def __init__(self,model_enco, model_deco, optim_enco, optim_deco,scheduler_enco,scheduler_deco):
        self.model_enco=model_enco
        self.model_deco=model_deco

        self.optim_deco=optim_deco
        self.optim_enco=optim_enco

        self.scheduler_enco=scheduler_enco
        self.scheduler_deco=scheduler_deco

        self.name_param_nan=[]
        self.value_param_nan=[]

        self.name_param_gradNaN=[]
        self.value_param_gradNaN=[]

        
    def add_attribut(self,x,u,sample,means,log_var_square,out,lr_enco,lr_deco):
        self.x=x
        self.u=u
        
        self.sample=sample
        self.means=means
        self.log_var_square=log_var_square
        self.out=out

        self.LRE=lr_enco
        self.LRD=lr_deco
    
    def add_param_nan(self,name,value):
        self.name_param_nan.append(name)
        self.value_param_nan.append(value)

    def add_gradNaN(self,name,value):
        self.name_param_gradNaN.append(name)
        self.value_param_gradNaN.append(value)           
        
def make_grid2d(self, ns):
        h, w = ns
        y, x = torch.meshgrid(
            torch.linspace(0, 1, h, device=self.reg_grid.device),
            torch.linspace(0, 1, w, device=self.reg_grid.device),
            indexing='ij'
        )
        return torch.stack((x, y), dim=-1)

@torch.no_grad()
def compute_gradient_norm(model):
    """
    Compute the total gradient norm (L2 norm of all gradients)
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm