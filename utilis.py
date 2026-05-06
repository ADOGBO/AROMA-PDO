
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
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
from einops import rearrange, repeat
from torch import einsum, nn
from functools import wraps
#from torch.utils.tensorboard import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter

#-------------- FUNCTIONS -------------------------------------------

def count_parameter(model):
    return sum( [para.numel()  for para in model.parameters() if para.requires_grad==True] )

def cycle(iterable):
    while True:
        for x in iterable: 
            yield x

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


#------------- OBJECT -------------------------------------------------------

class Gelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x,gates=x.chunk(chunks=2,dim=-1)
        return x*F.gelu(gates)
    

class FourrierFeatures(nn.Module):
    """ Randomly """
    def __init__(self,space_dim, space_embed_dim,include_input=False,sigma=10.):
        super().__init__()
        self.include_input=include_input
        B=sigma*torch.randn(space_embed_dim,space_dim)
        self.register_buffer("B",B)

    def forward(self,x):
        """shape of x: (...,space_dim)"""
        device=x.device
        
        omega_x=torch.pi*torch.matmul(x,self.B.T) #shape (..., space_embed_dim)
        sin_omega_x=torch.sin(omega_x)
        cos_omega_x=torch.cos(omega_x)
        proj=torch.cat(( torch.zeros_like(sin_omega_x), torch.zeros_like(cos_omega_x) ), dim=-1) #shape (...,2* space_embed_dim)
        proj[...,0::2]=sin_omega_x
        proj[...,1::2]=cos_omega_x

        if self.include_input:
            proj=proj.cat((proj,x),dim=-1) #shape (...,x_dim+ 2*space_embed_dim)
            
        return proj             #shape (...,2* space_embed_dim)
    

    
class FourierFeaturesBase2(nn.Module):
    """ We use base 2 for all experiments and select k frequencies in logarithmic scale per level.
     We have 
      gamma_1=[ 2^{min} : 2^{k+min}]
       gamma_2=[ 2^{min+1} : 2^{k+min+1}]
       ...
        gamma_{max-min+1}=[ 2^{max} : 2^{k+max}] """
    
    def __init__(self, log_scale_min,log_scale_max,k,use_pi=True,log_sampling=True,include_input=True):
        super().__init__()
        self.use_pi=use_pi
        self.include_input=include_input
        self.log_sampling=log_sampling

        frequence=[]
        if log_sampling:
            for freq in range(log_scale_min,log_scale_max+1): 
                frequence.append([freq**(log_scale) for log_scale in range( k )] ) # shape [k]
    


        else: #linear 
            for base in range(log_scale_min,log_scale_max+1):
                base_k=base**(k-1)
                frequence.append(np.array( np.linspace(1,base_k ,k) ) ) # shape [k]
            
        
        frequence=torch.tensor(frequence,dtype=torch.float32)
        self.register_buffer("omega", frequence) #shape (num_log_scale,k)


    def forward(self, x):

        #shape of x (batch,T,Dx,x_dim)
        device=x.device
        x_dim=x.shape[-1]
        #x=x.unsqueeze(-1) #shape (...,Dx,Dy,1)
        gamma=[]
        for freq in self.omega:
            freq=freq.expand(x_dim,-1).to(device) #shape (x_dim,k)
            if self.use_pi:
                omega_x=torch.pi*torch.matmul(x,freq) #shape (...,Dx,Dy,k)
            else:
                omega_x=torch.matmul(x,freq) #shape (...,Dx,Dy,k)

            #omega_x= omega_x.flatten(start_dim=-2,end_dim=-1) #shape (....,dim*k)

            proj=torch.cat(( torch.zeros_like(omega_x), torch.zeros_like(omega_x) ), dim=-1)
            proj[...,0::2]=torch.sin(omega_x) #shape (....,Dx,Dy,k)
            proj[...,1::2]=torch.cos(omega_x) #shape (....,Dx,Dy,k)
            if self.include_input:
                gamma.append( torch.cat( (proj,x),dim=-1) ) #shape (...,Dx,Dy,x_dim+2*k)
            else:
                gamma.append( proj )  #shape (...,Dx,Dy,2*k)

        
        return gamma # shape : num_log_scale of  (...,Dx,Dy,2*k)


class NeRFEncoding(nn.Module):
    """PyTorch implementation of regular positional embedding, as used in the original NeRF and Transformer papers."""
    def __init__(
        self,
        num_freq,
        max_freq_log2,
        log_sampling=True,
        include_input=True,
        input_dim=3,
        base_freq=2,
        use_pi=True,
    ):
        """Initialize the module.
        Args:
            num_freq (int): The number of frequency bands to sample.
            max_freq_log2 (int): The maximum frequency.
                                 The bands will be sampled at regular intervals in [0, 2^max_freq_log2].
            log_sampling (bool): If true, will sample frequency bands in log space.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.
        Returns:
            (void): Initializes the encoding.
        """
        super().__init__()

        self.num_freq = num_freq
        self.max_freq_log2 = max_freq_log2
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.out_dim = 0
        self.base_freq = base_freq
        self.use_pi = use_pi

        if include_input:
            self.out_dim += input_dim

        if self.log_sampling:
            self.bands = self.base_freq ** torch.linspace(
                0.0, max_freq_log2, steps=num_freq
            )
            if use_pi:
                self.bands = self.bands * np.pi
        else:
            self.bands = torch.linspace(
                1, self.base_freq**max_freq_log2, steps=num_freq
            )
            if use_pi:
                self.bands = self.bands * np.pi

        # The out_dim is really just input_dim + num_freq * input_dim * 2 (for sin and cos)
        self.out_dim += self.bands.shape[0] * input_dim * 2
        self.bands = nn.Parameter(self.bands).requires_grad_(False)

    def forward(self, coords, with_batch=True):
        """Embeds the coordinates.
        Args:
            coords (torch.FloatTensor): Coordinates of shape [N, input_dim]
        Returns:
            (torch.FloatTensor): Embeddings of shape [N, input_dim + out_dim] or [N, out_dim].
        """
        if with_batch:
            N = coords.shape[0]
            winded = (coords[..., None, :] * self.bands[None, None, :, None]).reshape(
                N, coords.shape[1], coords.shape[-1] * self.num_freq
            )
            encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
            if self.include_input:
                encoded = torch.cat([coords, encoded], dim=-1)

        else:
            N = coords.shape[0]
            winded = (coords[:, None] * self.bands[None, :, None]).reshape(
                N, coords.shape[1] * self.num_freq
            )
            encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
            if self.include_input:
                encoded = torch.cat([coords, encoded], dim=-1)
        return encoded

    def name(self) -> str:
        """A human readable name for the given wisp module."""
        return "Positional Encoding"

    def public_properties(self) -> Dict[str, Any]:
        """Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return {
            "Output Dim": self.out_dim,
            "Num. Frequencies": self.num_freq,
            "Max Frequency": f"2^{self.max_freq_log2}",
            "Include Input": self.include_input,
        }



class MultiHeadAttention1(nn.Module):


    def __init__(self, q_in_dim,k_in_dim,v_in_dim, qkv_out_dim, num_heads=1):
        super().__init__()
        self.q_in_dim = q_in_dim
        self.k_in_dim=k_in_dim
        self.v_in_dim=v_in_dim

        self.qkv_out_dim=qkv_out_dim

        self.num_heads=num_heads
        self.head_dim = qkv_out_dim // num_heads
        if qkv_out_dim % num_heads !=0:
            print(f"ERROR: embed_dim={qkv_out_dim} must be divisible by num_heads={num_heads}")
        assert qkv_out_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(q_in_dim, qkv_out_dim)
        self.k_linear = nn.Linear(k_in_dim, qkv_out_dim)
        self.v_linear = nn.Linear(v_in_dim, qkv_out_dim)
        self.out_linear = nn.Linear(qkv_out_dim, qkv_out_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)  # shape of query, key, value :(batch_size,seq_lenght,emnded_dim)
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) #shape of Q,K,V: (batch_size,num_heads,seq_lenght,head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  #shape of scores: (batch_size,num_heads,seq_lenght,seq_lenght)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_output = torch.matmul(attention_weights, V) # shape : (batch_size,num_heads,seq_lenght,head_dim)
        attended_output = attended_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim) # shape : (batch_size,seq_lenght,emnded_dim)
        output = self.out_linear(attended_output) #shape (batch_size,seq_lenght,emnded_dim)
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1,att_dropout=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if embed_dim % num_heads !=0:
            print(f"ERROR: embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout=nn.Dropout(att_dropout)
        self.resid_dropout=nn.Dropout(att_dropout)

    def forward(self, query, key, value, mask=None):
        """ shape of  key, value :(batch_size,seq_lenght,emnded_dim)
        query could change ((1,seq_lenght,emnded_dim))
        """
        batch_size_q = query.size(0)  # 
        batch_size_k = key.size(0)
        batch_size_v = value.size(0)
        
        Q = self.q_linear(query).view(batch_size_q, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size_k, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size_v, -1, self.num_heads, self.head_dim).transpose(1, 2) #shape of Q,K,V: (batch_size,num_heads,seq_lenght,head_dim)
        
        #print("shape of query in the attention", Q.shape)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  #shape of scores: (batch_size,num_heads,seq_lenght,seq_lenght)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1) #shape of scores: (batch_size,num_heads,seq_lenght,seq_lenght)
        attention_weights=self.attn_dropout(attention_weights)

        attended_output = torch.matmul(attention_weights, V) # shape : (batch_size,num_heads,seq_lenght,head_dim)
        attended_output = attended_output.transpose(1, 2).contiguous().view(batch_size_v, -1, self.num_heads * self.head_dim) # shape : (batch_size,seq_lenght,emnded_dim)

        output = self.resid_dropout(self.out_linear(attended_output)) #shape (batch_size,seq_lenght,emnded_dim)
        return output, attention_weights
    

class FeedForward(nn.Module):
    def __init__(self,input_dim,mult_dim,use_gelu=False):
        super().__init__()
        
        if use_gelu:
            self.net=nn.Sequential(
                nn.Linear(input_dim,2*input_dim*mult_dim),
                Gelu(),
                nn.Linear(input_dim*mult_dim,input_dim)
            )

        else:
            self.net=nn.Sequential(
                nn.Linear(input_dim, input_dim*mult_dim),
                nn.GELU(),
                nn.Linear(input_dim*mult_dim,input_dim)
            )


    def forward(self,x):
        return self.net(x)
        
        
class Prenorm(nn.Module):
    def __init__(self,fn,dim):
        super().__init__()
        self.dim=dim
        self.norm=nn.LayerNorm(dim)
        self.fn=fn


    def forward(self,x):
        assert self.dim==x.size(-1),f"Here we are  Prenorm in Error last dim of x {x.size(-1)} must be equal dim={self.dim}"
        return self.fn(self.norm(x))
    
class PreNormCross(nn.Module):
    def __init__(self, dim, fn, k_dim=None, v_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(k_dim) if exists(k_dim) else None
        self.norm_v = nn.LayerNorm(v_dim) if exists(v_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_v):
            key = kwargs["key"]
            value = kwargs["value"]
            normed_k = self.norm_k(key)
            normed_v = self.norm_v(value)
            kwargs.update(key=normed_k, value=normed_v)

        return self.fn(x, **kwargs)


class Postnorm(nn.Module):
    def __init__(self,fn,dim):
        super().__init__()
        self.dim=dim
        self.norm=nn.LayerNorm(dim)
        self.fn=fn


    def forward(self,x):
        assert self.dim==x.size(-1),f"Here we are  Postnorm in Error last dim of x {x.size(-1)} must be equal dim={self.dim}"
        return self.norm(self.fn(x))

class ResidualBlock(nn.Module):

    def __init__(self,fn):
        self.fn=fn

    def forward(self,x):
        return x+self.fn(x)

    
    
class LoglihoodNormal(nn.Module):

    def __init__(self):
        super().__init__()
        self.log_to_pi=np.log( 2*np.pi)
        
    def forward(self,sample,logvar_square,mean):
        var=(0.5)*torch.exp(logvar_square)
        return (0.5)*( self.log_to_pi +logvar_square+ torch.pow((sample-mean)*var,2))
    
class Transformer_enco_block(nn.Module):
    def __init__(self,dim,num_heads,mult_dim,att_dropout,use_gelu=False):
        super().__init__()
        self.attn= MultiHeadAttention(dim,num_heads=num_heads,att_dropout=att_dropout) 
        self.norm1=nn.LayerNorm(dim)
        self.ff=FeedForward(dim,mult_dim)
        self.norm2=nn.LayerNorm(dim)

    def forward(self,querry,key,value):
        context,_=self.attn(querry,key=key,value=value)
        querry= self.norm1(context+querry)
        ff_out=self.ff(querry)

        return self.norm2(ff_out+querry)


class Encoder(nn.Module):
    def __init__(self,x_dim,u_dim,M,d,h,log_scale_min,log_scale_max,k,device,num_enco_head=1,attn_dropout=0, enco_dropout=0,mult_dim_ff=4,
                use_pi=True,log_sampling=True,include_input=True,use_gelu=False,enco_geo=True,include_pos_in_value=False,fourrier_feature_type:str="base2"):
        
        super().__init__()
        if fourrier_feature_type=="base2":
            self.ff=FourierFeaturesBase2(log_scale_min,log_scale_max,k,use_pi,log_sampling,include_input)
        elif fourrier_feature_type=="random":
            self.ff=FourrierFeatures(x_dim, k,include_input)
        else:
            raise ValueError(f"please choose fourroer_feature_type between {'base2'} and {'random'}")
        
        x=torch.zeros(x_dim,device=device).view(1,-1)
        y= self.ff(x)
        try:
            shapes=y.shape
        except AttributeError:
            shapes=y[0].shape
        dim_x_embed=shapes[-1]
        
        self.enco_dropout=nn.Dropout(enco_dropout)
        self.enco_geo=enco_geo
        self.include_pos_in_value=include_pos_in_value
        
        #--------INITIALISATION OF ENCO_GEO
        #self.T=nn.Parameter(torch.ones(1,M,d), requires_grad=True).to(device)
        small_std = False
        sigma = 0.02 if small_std else 1
        self.latents = nn.Parameter(torch.randn(M,d) * sigma)


        # ----------------- Linear after ff
        self.linear_x_after_ff=nn.Linear(dim_x_embed,d)

        # ---------Attention for encoding geo--------------------------------------------
        if enco_geo:
            self.cross_attention_x=PreNormCross( d, MultiHeadAttention(d,num_enco_head,attn_dropout),k_dim=d,v_dim=d )
        
        # ---------Attention for encoding the observation--------------------------------------------
        self.cross_attention_u=PreNormCross( d, MultiHeadAttention(d,num_enco_head,attn_dropout),k_dim=d,v_dim=d )

        # ------------ Linear layer for embeded u ----------------------------
        self.embed_velocity=nn.Linear(u_dim,d)

        #-------------- feedforward after x cross attention
        if enco_geo:
        
            self.ff_x_after_cross_att= Prenorm(
                FeedForward(d,mult_dim_ff,use_gelu),
                d
            )
        

        #-------------  Linear after v cross attention
        self.ff_u_after_cross_att=Prenorm(
            FeedForward(d,mult_dim_ff,use_gelu),
            d
        )

        #--------- Linear for obtaining the mean and the standard deviation-----
        self.linear_mu=nn.Linear(d,h)
        self.linear_log_sigma=nn.Linear(d,h)

        #--------- Loglihoood to perferm log(p(z/x))-----------------
        self.log_z_x=LoglihoodNormal()

    def reparametrization_trick(self,mean,log_var_square):
        log_var_square=torch.clamp(log_var_square,-30,20)
        z=torch.randn_like(mean)
        return mean+(0.5)*torch.exp(log_var_square)*z

    def forward(self,x,u):
        """
        shape of :
        x (batch_size,seq_len,x_dim) or  (seq_len,x_dim)
        u (batch_size,seq_len,u_dim)  or (seq_len,u_dim)
         
           
        """
        
        if len( x.shape)==2: 
            x=x.unsqueeze(0) #we force x to be of shape (batch_size,seq_len,x_dim)
        if len(u.shape)==2:
            u=u.unsqueeze(0) #we force u to be of shape (batch_size,seq_len,u_dim)
        assert x.shape[1]==u.shape[1],f"Must have same sequence lenght. Got { x.shape[1]} and {u.shape[1]}" 
        b=u.shape[0]

        #----------Embedding--------------------------------
        gamma_x=self.linear_x_after_ff( self.ff(x)[0] ) #shape (batch,N,d) # self.ff(x) is always a list

        

        v=self.embed_velocity(u) #shape (batch,N,d)
        assert not torch.isnan(v).any(),"NaN dectecte dans v=self.embed_velocity(u)" 

        #------------------GEOMETRIE --------------
        T= repeat(self.latents, "n d -> b n d", b=b)
        assert not torch.isnan(T).any(), "NaN dected in T"

        #print("geometrie shape:",T.shape)
        if self.enco_geo:

            context_x,_=self.cross_attention_x( T,key=gamma_x,value=gamma_x) #shape (batch,M,d)
            assert not torch.isnan(context_x).any()," Nan dectected in context_x"

            context_x=T+context_x
            assert not torch.isnan(T).any(), "Nan detected in T=T+context_x"

            #T_geo=context_x+self.ff_x_after_cross_att(context_x ) #shape (batch,M,d)
            T_geo=context_x+self.ff_x_after_cross_att(context_x) #shape (batch,M,d)
            assert not torch.isnan(T_geo).any(), "Nan detected in T_geo"

        else: 
            T_geo=T
            assert not torch.isnan(T_geo).any(), "Nan detected in T_geo"

        #-------- construction of T_ob-------------------------------
        context_u,_=self.cross_attention_u(T_geo,key=gamma_x+v if self.include_pos_in_value else gamma_x ,value=v)  #shape (batch,M,d)
        assert not torch.isnan(context_u).any(), "NaN detected in context_u"

        context_u=context_u+T_geo
        assert not torch.isnan(context_u).any(), "NaN decteced in context_u=context_u+T_geo"

        T_ob=context_u+self.ff_u_after_cross_att(context_u)  #shape (batch,M,d)
        assert not torch.isnan(T_ob).any()," NaN observed in T_ob"

        #---------------------- Reduction of dimension __________________ 
        means=self.linear_mu(T_ob)
        assert not torch.isnan(means).any(), "Nan observed in means"

        logvar_square= self.linear_log_sigma(T_ob)
        assert not torch.isnan(logvar_square).any(), "NaN observed in logvar_square"

        sample=self.reparametrization_trick(means,logvar_square)
        assert not torch.isnan(sample).any(), "NaN detected in sample"

        log_z_x= self.log_z_x(sample,logvar_square,means)
        
        return sample,means,logvar_square,log_z_x
    
    
    def recover_latent(self):
        return self.latents, torch.norm(self.latents, p=2)



class Decoder(nn.Module):
    def __init__(self,h,d,k,num_self_attn_deco,mult_dim_deco,x_dim,out_dim,hidden_dim_deco,log_scale_min,log_scale_max,device,
                use_gelu_deco=False,num_heads_deco=1, 
                att_dropout_deco=0,fourrier_feature_type:str="base2",num_fourier_feature_deco=1,depth_deco=3,
                use_pi=True,log_sampling=True,include_input=True,same_self_block=True):
        super().__init__()

        self.num_fourier_feature_deco=num_fourier_feature_deco
        self.same_self_block=same_self_block

        #-------------- Lifter -------------------------
        self.lifter=nn.Linear(h,d)

        #--------------- Self attention block -------------------------------------
        self.self_attention_blocks=nn.ModuleList()
        if same_self_block:
            get_self_block=cache_fn(lambda: Transformer_enco_block(d,num_heads_deco,mult_dim_deco,att_dropout_deco,use_gelu=use_gelu_deco))
            for _ in range(num_self_attn_deco):
                self.self_attention_blocks.append(get_self_block())
        else:
            for _ in range(num_self_attn_deco):
                self.self_attention_blocks.append( Transformer_enco_block(d,num_heads_deco,mult_dim_deco,att_dropout_deco,use_gelu=use_gelu_deco) )
                
            

        #------------------- Cross attention block -------------------------------
        
        self.cross_attention= Transformer_enco_block(d,num_heads_deco,mult_dim_deco,att_dropout_deco,use_gelu=use_gelu_deco)
        #---------- decoder-----------------------------------------------------
        mlp=[]
        #Input layer
        mlp.append(nn.Linear(num_fourier_feature_deco*d,hidden_dim_deco))
        mlp.append(nn.ReLU())

        ## Add intermediate layers based on depth
        for _ in range(depth_deco-1):
            mlp.append(nn.Linear(hidden_dim_deco,hidden_dim_deco))
            mlp.append(nn.ReLU())

        # Output layer
        mlp.append(nn.Linear(hidden_dim_deco,out_dim))

        self.decode=nn.Sequential(*mlp)
        
        #--------------------- Fourier feature embedding ------------------------
        self.fourrier_feature_decoder=nn.ModuleList()
        for _ in range(num_fourier_feature_deco):
            if fourrier_feature_type=="base2":
                self.fourrier_feature_decoder.append(FourierFeaturesBase2(log_scale_min,log_scale_max,k,use_pi,log_sampling,include_input) )
            elif fourrier_feature_type=="random":
                self.fourrier_feature_decoder.append(FourrierFeatures(d, k,include_input) )
            else:
                raise ValueError(f"please choose fourroer_feature_type between {'base2'} and {'random'}")
        x=torch.zeros(x_dim,device=device).view(1,-1)
        y= self.fourrier_feature_decoder[0](x)
        try:
            shapes=y.shape
        except AttributeError:
            shapes=y[0].shape
        dim_x_embed=shapes[-1]

        # ----------------- Linear after ff
        self.linear_x_after_ff=nn.ModuleList()
        for _ in range(num_fourier_feature_deco):
            self.linear_x_after_ff.append(nn.Linear(dim_x_embed,d))



            
    def forward(self,Z,x):
        Z=self.lifter(Z)

        for block in self.self_attention_blocks:
            Z=block(Z,Z,Z)
        
        # ----- For treating gamma_x
        gamma=[]
        for ind in range(self.num_fourier_feature_deco):
            linear=self.linear_x_after_ff[ind]
            ff=self.fourrier_feature_decoder[ind]
            gamma.append( linear(ff(x)[0]) )

        f_q=[]
        for gamma_x in gamma:
            f_q.append( self.cross_attention(gamma_x,Z,Z)  )

        f_q=torch.cat(f_q,dim=-1)
        return self.decode(f_q)
    
class LocalityAwareINRDecoder(nn.Module):
    def __init__(self, output_dim=1, embed_dim=16, num_scales=3, dim=128, depth=3):
        super().__init__()
        self.dim = dim
        # Define Fourier transformation, linear layers, and other components
        self.depth = depth
        layers = [nn.Linear(embed_dim * num_scales, dim), nn.ReLU()]  # Input layer

        # Add intermediate layers based on depth
        for _ in range(depth - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dim, output_dim))  # Output layer
        self.mlp = nn.Sequential(*layers)

    def forward(self, localized_latents):
        # we stack the different scales
        localized_latents = einops.rearrange(localized_latents, "b n s c -> b n (s c)")
        return self.mlp(localized_latents)
