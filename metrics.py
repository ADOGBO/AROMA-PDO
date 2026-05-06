#import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
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
import timm

from einops import rearrange, repeat
from torch import einsum

from torch.fft import fft2,ifft2,fftfreq

from pathlib import Path
from timm.models.vision_transformer import Attention, Mlp

from utilis import PreNormCross, MultiHeadAttention, Prenorm, FeedForward


class EnergySpectrum(object):
    def __init__(self,dx, N ,L,device):
        self.dx=L/N
        self.N=N

        kx=2*torch.pi*fftfreq(N,d=self.dx,device=device)
        ky=2*torch.pi*fftfreq(N,d=self.dx,device=device)


        self.kxx,self.kyy=torch.meshgrid(kx,ky,indexing='ij')
        

        self.kzz=torch.sqrt(self.kxx**2 +self.kyy**2)



    def spectrum(self,u:torch.tensor,k:float,gap:int=1.):
        #--------------- compute fft of u-------------
        u_hat=fft2(u)
        E_modes=u_hat**2
        
        

        E_spect=torch.pi*E_modes*self.kzz

        kmax = int(torch.max(self.kzz).item())
        E_spectrum = torch.zeros(gap*kmax+1,device=u.device)
        counts = torch.zeros(gap*kmax+1,device=u.device)

        K_int = torch.floor(self.kzz)
        K_int=K_int.to(torch.uint64)

        for i in range(self.N):
            for j in range(self.N):
                k = K_int[i,j]
                m=self.kzz[i,j]-k
                l=torch.floor(m/gap)

                E_spectrum[gap*k+l] += E_spect[i,j]
                counts[gap*k+l] += 1

        # Average
        E_spectrum /= torch.maximum(counts, 1)
        return E_spectrum





class PhysicalPatternsBurgers(object):
    def __init__(self,dx):
        self.dx=dx

    def mass(self,u):
        """ 
        u is the velocity of shape: (num_sampled,batch,T,N,1)
        
        """
        u=u.squeeze(-1)
        return torch.sum(u,dim=-1) # shape (batch,T)
    
    def energy(self,u):

        """ 
        u is the velocity of shape: (num_sampled,batch,T,N,1)
        
        """
        u=u.squeeze(-1)
        return (0.5)*self.dx*u**2 # shape (batch,T,N)

    def gradient(self,u):
        """ 
        u is the velocity of shape: (num_sampled,batch,T,N,1)
        
        """
        u=u.squeeze(-1)       # shape (batch,T,N)
        return ((0.5)/self.dx)*( u[...,2:]-u[...,:-2] ) # shape (batch,T,N-2)

    def boundary_condition(self,u):
        """ 
        u is the velocity of shape: (num_sampled,batch,T,N,1)
        
        """
        u=u.squeeze(-1)       # shape (batch,T,N)
        return  u[...,0]-u[...,-1]  # shape (batch,T)

def physicalMetricsBurgers(Y_pred,Y_true,dx,mass_weight,energy_weight,grad_weight,boundary_weight,threshold=0):
    """
        Y_pred:(num_samples,batch,T,N,1)
        Y_true:(1,batch,T,N,1)
    """
    #criterion=torch.nn.L1Loss(reduction=sum)

    pattern=PhysicalPatternsBurgers(dx)

    #mass=pattern.mass(Y_pred)-pattern.mass(Y_true) # shape (num_sample,batch,T)

    # Attention the mass of each state of the trajectorie is zero
    mass=pattern.mass(Y_pred)                        # shape (num_sample,batch,T)
    mass=dx*torch.abs(mass)
    

    energy=pattern.energy(Y_pred)-pattern.energy(Y_true)  # shape (num_sample,batch,T,N)
    energy=torch.mean( torch.abs(energy) ,dim=-1)       # shape (num_sample,batch,T)
    

    # We masked the valeurs smaller than 0
    grad_pred= pattern.gradient(Y_pred)
    grad_target= pattern.gradient(Y_true)
    if threshold>0:
        grad_pred[grad_pred<threshold]=0.
        grad_target[grad_pred<threshold]=0.

    grad= torch.abs(grad_target-grad_pred )     ## shape (num_sample,batch,T,N-2)
                       
    grad=torch.mean(grad,dim=-1)            # shape (num_sample,batch,T)
    

    boundary=pattern.boundary_condition(Y_pred)-pattern.boundary_condition(Y_true) # shape (num_sample,batch,T)
    boundary=torch.abs(boundary)
    

    metric=mass_weight*mass+energy_weight*energy
    +grad_weight*grad+boundary_weight*boundary # shape (num_sample,batch,T)
    
    return metric

def winLos(Y_pred,Y_true,dx,mass_weight,energy_weight,grad_weight,boundary_weight):
    _,batch_size,T_size,_,_=Y_pred.shape
    
    metric=physicalMetricsBurgers(Y_pred,Y_true,dx,mass_weight,energy_weight,grad_weight,boundary_weight)
    _,winner_idx=torch.min(metric,dim=0)
    _,loser_idx=torch.max(metric,dim=0)

    batch_idx = torch.arange(batch_size).view(batch_size, 1).expand(batch_size, T_size)
    T_idx = torch.arange(T_size).view(1, T_size).expand(batch_size, T_size)

    # Utiliser l'indexation avancée
    winner = Y_pred[winner_idx, batch_idx, T_idx, :, :] #(batch_size,T,N,1)
    loser = Y_pred[loser_idx, batch_idx, T_idx, :, :] #(batch_size,T,N,1)

    return winner, loser # (batch,T)



class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "Positional embeddings should be multiples of 2"

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim) 
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TransformersBlock(nn.Module):
    """
    A transformer block 
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        

    def forward(self, x):
        
        x = x +  self.attn( self.norm1(x) )
        x = x +  self.mlp( self.norm2(x) )
        return x

class AttentionBlock(nn.Module):
    def __init__(self,hidden_size,num_head,mlp_ratio=4.0):
        super().__init__()
        
        self.attention=PreNormCross( hidden_size, MultiHeadAttention(hidden_size,num_heads=num_head,att_dropout=0),k_dim=hidden_size,v_dim=hidden_size )
        self.ff_after_cross_att= Prenorm(
                FeedForward(hidden_size,int(mlp_ratio),use_gelu=False),
                hidden_size
            )

    def forward(self,q,k,v):
        """
        q: (batch,1,hidden_size)
        k:(batch,seq_len,hidden_size)
        v:(batch,seq_len,hidden_size)
        """
        context_,_=self.attention( q,key=k,value=v) #shape (1,seq_le,hidden_size)
        #assert not torch.isnan(context_x).any()," Nan dectected in context_x"

        q=q+context_
        #assert not torch.isnan(T).any(), "Nan detected in T=T+context_x"

        return q+self.ff_after_cross_att(q) #shape (batch,M,d)
        #assert not torch.isnan(T_geo).any(), "Nan detected in T_geo"




class RewardSignal(nn.Module):
    def __init__(self,depth,input_size,num_heads,hidden_size, output_size=1,num_point=None, frequency_embedding_size=256,reduction="last",x_space="regular"):
        super().__init__()
        self.depth=depth
        self.reduction=reduction
        
        if x_space=="regular":
            pos_embeder=SinusoidalPositionEmbeddings(hidden_size)
            assert num_point,"you have to give num_point"
            timess=torch.arange(num_point)
            space_embedding=pos_embeder(timess) #(num_point,hidden_size)
            
            self.register_buffer("space_embedding", space_embedding)

        self.u_embeder=nn.Linear(input_size,hidden_size,bias=True)

        self.layers=nn.ModuleList([])
        for _ in range(depth-1):
            self.layers.append(TransformersBlock(hidden_size, num_heads))
        
        if reduction=="last" or reduction=="mean":
            self.penultimate_layer=TransformersBlock(hidden_size, num_heads)
        
        elif reduction=="attention":
            small_std = False
            sigma = 0.02 if small_std else 1
            self.metric_query = nn.Parameter(torch.randn(1,hidden_size))
            
            self.penultimate_layer=AttentionBlock(hidden_size,num_heads,mlp_ratio=4.0)
            
    


        self.final_layer = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self,X):
        """ 
        shape of X: (batch,T,N,input_size)
        
        """
        batch_size=X.shape[0]
        X=rearrange(X,'b T N d->(b T) N d')
        
        space_embe=self.space_embedding.to( X.device )
        X= self.u_embeder(X) +space_embe.unsqueeze(0) #((batch T),N,hidden_size) #((batch T),N,hidden_size)

        for layer in self.layers:
            X=layer(X)

        if self.reduction=="mean":
            X=self.penultimate_layer(X)
            X= torch.mean(X,dim=1) #((batch T),hidden_size)

        elif self.reduction=="last":
            X=self.penultimate_layer(X)
            X= X[:,-1,:] #((batch T),hidden_size)
        
        else: #self.reduction==attention
            X=self.penultimate_layer(self.metric_query,k=X,v=X) #((batch T),1,hidden_size)
            
            X=X.squeeze(1) #((batch T),hidden_size)



        X=self.final_layer(X) #((batch T), output_size=1)
        
        X=X.squeeze(-1) #(batch T)
        X=rearrange(X,"(b T)->b T",b=batch_size)
        return X





