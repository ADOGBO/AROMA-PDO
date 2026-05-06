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

from einops import rearrange, repeat
from torch import einsum

import os
from torch.cuda.amp import autocast, GradScaler

from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers.schedulers import DDPMScheduler
from torch.nn.utils import clip_grad_norm_

""" 
Cette method me permert d'eviter d'avoir plusieurs dans le meme dossier
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from models.model import MyModel

ou encore


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
"""

""" 
$$$$$$$$$$$$ A MODIFIER $$$$$$$$$$$$$$$$
- modifier losses_train en losses_train latent
-changer la place de la derniere checpoint model 
-prendre la derniere version de module sur Jean-Zay:
 """

"""
Voir si on peut pas ameliorer la main surtout la boucle dans DiT
  """

#from DiT import DiT
#from utilis import Encoder,Decoder
from metrics import *


def add_args(parser):

    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=582838)

    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=2000) 
    parser.add_argument("--log_interval", type=int, default=2)


    parser.add_argument("--x_dim",type=int,default=1,help="dimension of x")
    parser.add_argument("--u_dim",type=int,default=1,help="dimension of u")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--global_batch_size", type=int, default=32,help="Taille batch globale")
    parser.add_argument("--mini_batch_size", type=int, default=4,help="Taille mini-batch pour éviter dépassement mémoire")

    parser.add_argument("--num_sample", type=int, default=10,help="number of samples of trajectories")
    #parser.add_argument("--num_point", type=int, default=1,help="output dim for metric")

    parser.add_argument("--mass_weight", type=float, default=100.,help="the mass weight in metric")
    parser.add_argument("--grad_weight", type=float, default=10.,help="the gradient weight in metric")
    parser.add_argument("--energy_weight", type=float, default=1.,help="the energy weight in metric")
    parser.add_argument("--boundary_weight", type=float, default=0.1,help="the boundary condition weight in metric")
    parser.add_argument("--eps", type=float, default=1e-32,help="epsilon in the log of the metric")
    #parser.add_argument("--pertu", type=float, default=0.1,help="the standard deviation of the pertubation ")

    parser.add_argument("--KL_weight", type=float,default=0.0001,help="the weight in the enco_decoder loss")
    parser.add_argument("--DPO_weight", type=float,default=10,help="the weight in the DPO loss") #or 0.5

    #parser.add_argument("--pertu", type=float, default=0.1,help="the standard deviation of the pertubation ")
    parser.add_argument("--pertu_deviation_set", type=list, default=[0.1,0.2,0.3,0.4,0.5],help="the set standard deviation of the pertubation ")
    parser.add_argument("--ratio_target", type=float, default=0.1,help="the ratio of target in the winner")

    parser.add_argument("--save_dir", type=str, default="checkpoints")  #checkpoint



def loss_model(mean,log_var_square,x,x_hat,beta,return_KL=False):
    """
         mean,log_var_square,x,x_hat : (bach,seq_length,d)
    
    """
    
    KL_divergence_per_batch=0.5*torch.mean( -1-log_var_square+mean**2+ torch.exp(log_var_square),dim=[1,2] )
    
    if return_KL==False:

        return beta*torch.mean(KL_divergence_per_batch)+F.mse_loss(x_hat,x)
    else:
        return KL_divergence_per_batch, beta*torch.mean(KL_divergence_per_batch)+F.mse_loss(x_hat,x)
    

def cycle(iterable):
    while True:
        for x in iterable: 
            yield x



def save_checkpoint(state,tmp_path, final_path):
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
    print(f"$$$$$$$$$ Checkpoint sauvegardé avec succès : {final_path}")


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



    

class Dataset_Burgers(Dataset):
    def __init__(self,data,x_dim,x_min=0,x_max=16,t_min=0,t_max=4):
        super().__init__()
        """
        for Burgers equation :
        data (numpy): Dataset values, with shape (N T Dx). Where N is the
            number of trajectories, Dx the size of the first spatial dimension, and T the
            number of timestamps.
        
        """
        self.data=torch.tensor(data,dtype=torch.float32).unsqueeze(-1) #shape (N T Dx C) where C channel (always 1)
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
    

if __name__ == "__main__":
    
    

    parser=argparse.ArgumentParser()
    add_args(parser)


    cfg=parser.parse_args()

    #cfg.hidden_size=int(4*cfg.num_heads)
    
    cfg.accumulation_steps = cfg.global_batch_size // cfg.mini_batch_size
    cfg.dx=0.16

    pertu_dev_set=cfg.pertu_deviation_set
    

    # Paramètres HPC optimisés
    scaler = GradScaler() #Réduit la mémoire par 2 à 4× et Accélère le calcul sur A100
    torch.backends.cudnn.benchmark = True  # optimise les convolutions

    #device
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    except Exception as e:
        # Fallback to CPU if device selection fails
        print(f"Warning: Device selection failed ({e}), using CPU")
        device = torch.device("cpu")

    print(f"Using device: {device}")

    torch.manual_seed=(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)


    print("-------------- Loading train data-------------")
    current_dir = Path(__file__).resolve().parent 
    parent_dir=Path(__file__).resolve().parent.parent 
    fname = parent_dir/ "dataset" / "CE_train_E1.h5"
    with h5py.File(fname, "r") as f:
        data = f["train/pde_250-100"][:]
    
    train_data=Dataset_Burgers(data,cfg.x_dim,x_min=0,x_max=16,t_min=0,t_max=4)

    print("-------------- Loading test data-------------")
    fname = parent_dir/"dataset" / "CE_test_E1.h5"
    with h5py.File(fname, "r") as f:
        data = f["test/pde_250-100"][:] #shape (B, T, N )
    
    cfg.num_point=int( data.shape[2] )
    cfg.dx=0.16
    test_data=Dataset_Burgers(data,cfg.x_dim,x_min=0,x_max=16,t_min=0,t_max=4)

    kwargs = {"num_workers": 4, "pin_memory": True} if cfg.use_gpu else {}
    train_loader=DataLoader(dataset=train_data, batch_size=cfg.mini_batch_size, shuffle=False,**kwargs)
    test_loader=DataLoader(dataset=test_data, batch_size=cfg.test_batch_size, shuffle=False,**kwargs)
    test_loader_cycle=cycle(test_loader)


    # For download existing model from checkpoint
    checkpoint_dir=Path(parent_dir/ cfg.save_dir)
    enco_deco_path= checkpoint_dir/"checkpoint.pt"
    
    metrcis_path = checkpoint_dir / "checkpoint_metrics_encoDeco.pt"

    


    assert enco_deco_path.exists() and enco_deco_path.is_file(), "No encoder decoder Burgers saved"
    
    assert metrcis_path.exists() and metrcis_path.is_file(), "No Metrics Burger saved"

    

    enco_deco_DPO_path= checkpoint_dir/"checkpoint_enco_deco_DPO.pt"
    tmp_path= checkpoint_dir / "checkpoint_metrics.tmp"
    last_enco_deco_DPO_path=checkpoint_dir / "checkpoint_enco_deco_DPO_lastModel.pt"

    train_losses_enco_deco_DPO_path=checkpoint_dir / "losses_train_enco_deco_DPO.npy"

    test_losses_enco_deco_DPO_path=checkpoint_dir / "losses_test_enco_deco_DPO.npy"


    
    
    

    #f = Path(ckpt_path)
    #print("resolve==============",f.resolve())

    ##-------------------------LOAD Metrics Model -----------------------------


    state_metrics=torch.load(metrcis_path,weights_only=False)
    #state_metrics=torch.load(metrcis_path,weights_only=False,map_location=torch.device("cpu"))
    
    

    if enco_deco_DPO_path.exists() and enco_deco_DPO_path.is_file():
        with enco_deco_DPO_path.open("rb") as fp:
            state=torch.load(fp, weights_only=False) #on recommence depui s l e modele sauvegarde
            #state=torch.load(fp, weights_only=False,map_location=torch.device("cpu")) #on recommence depui s l e modele sauvegarde
            #print(state)
            print("-------------------telechargemnt model actuel enco_deco reussi----------------------")

    else:
        state=torch.load(enco_deco_path, weights_only=False)
        #state=torch.load(enco_deco_path, weights_only=False,map_location=torch.device("cpu"))
        # We initialize the rest of attribut

        state.optim_enco=torch.optim.Adam(state.model_enco.parameters() ,lr=cfg.learning_rate  )
        state.optim_deco=torch.optim.Adam(state.model_deco.parameters() ,lr=cfg.learning_rate  )


        state.scheduler_enco = CosineAnnealingLR(state.optim_enco, T_max=cfg.max_iterations, eta_min=1e-5)
        state.scheduler_deco = CosineAnnealingLR(state.optim_deco, T_max=cfg.max_iterations, eta_min=1e-5)
        
        state.epoch=0 #epoch intitial
        state.best_valid_loss=np.inf 
        
    # _____________LOAD losses-------------------------------
    if  train_losses_enco_deco_DPO_path.exists() and train_losses_enco_deco_DPO_path.is_file(): #It is enough to consider just one
        
        train_losses_last=np.load(train_losses_enco_deco_DPO_path) #
        test_losses_last=np.load(test_losses_enco_deco_DPO_path)
        
        train_losses_last=train_losses_last.tolist()
        test_losses_last=test_losses_last.tolist()
        
        

        print("-------------------telechargemnt loss enco_deco_DPO last  reussi----------------------")

    else:    
        train_losses_last=[]
        test_losses_last=[]
        
    


    
    counter=0
    patience=100
    
    
    
    sigmoid=nn.Sigmoid()
    num_grad_update=0
    print("Starting of optimizing")

    

    for step in range(state.epoch,cfg.max_iterations): 
        t0=time.time()
        total_train_loss=0
        state.epoch=step
        #for step in range(1):

        #Réinitialisation des gradients
        state.optim_enco.zero_grad()
        state.optim_deco.zero_grad()
        

        print(f"---------{step}/{cfg.max_iterations}--------------")
        for step_train_loader,batch in enumerate(train_loader):

            state.model_enco.train()
            state.model_deco.train()

            

            x=batch[0] #shape (Batch,T,N,1)
            u=batch[1]  #shape (Batch,T,N,1)
            x=x.to(device)
            u=u.to(device)

            target=u
            #x_out=x
            batch_size_tr,time_step,space_step,_=x.shape #x.shape

            x=rearrange(x, "b T N d -> (b T) N d")
            u=rearrange(u, "b T N d -> (b T) N d")
            
            
            
            

            with autocast(enabled=False): # mixed precision
                sample,means,log_var_square,_=state.model_enco(x,u)
                out=state.model_deco(sample,x)

                out=rearrange(out,"(b T) N d -> b T N d",b=batch_size_tr)
                loss_mse=loss_model(means,log_var_square,target,out,cfg.KL_weight)


                # start of DPO

                u=rearrange(u,"(b T) N d -> b T N d",b=batch_size_tr)
                x=rearrange(x,"(b T) N d -> b T N d",b=batch_size_tr)

                # We select radomly the deviation
                pertu=np.random.choice(pertu_dev_set)
                pertubation=pertu*torch.randn( cfg.num_sample,*u.shape,device=device) #shape (E, b, T, N, d)


                u=u.unsqueeze(0)+ pertubation              #shape (E, b, T, N, d)
                u=rearrange(u,"E b T N d -> (E b T) N d")

                x=x.expand(cfg.num_sample, *x.shape)            #shape (E, b, T, N, d)
                x=rearrange(x,"E b T N d -> (E b T) N d")

                sample,_,_,_=state.model_enco(x,u) #shape ((E, batch,T),M,h)
                out=state.model_deco(sample,x)   # shape ((E, batch,T),N,d)

                out=rearrange(out,"(E b T) N d -> E b T N d",E=cfg.num_sample,b=batch_size_tr)
                winner,loser=winLos(out,target.unsqueeze(0),cfg.dx,cfg.mass_weight,cfg.energy_weight,cfg.grad_weight,cfg.boundary_weight) # shape ( b, T,N,d)

                # We remplace ratio of winner with target
                ratio_target=int(cfg.ratio_target*batch_size_tr)
                target_ind=torch.randint(batch_size_tr, (ratio_target,), device=device)

                winner[target_ind]=target[target_ind]
                winner[:,0,:,:]=target[:,0,:,:]

                with torch.no_grad():
                    winner_score_a=state_metrics.model_attn(winner) #(b,T)

                    loser_score_a=state_metrics.model_attn(loser)

                loss_DPO=-torch.log(cfg.eps+ sigmoid(winner_score_a-loser_score_a))
                loss_DPO=loss_DPO.mean()

                #FINAL LOSS
                loss=loss_mse+cfg.DPO_weight*loss_DPO

                scaler.scale(loss).backward()

                total_train_loss+=loss.detach().cpu().numpy()/len(train_loader)

                if ((step_train_loader+1)% (cfg.accumulation_steps)==0) or ((step_train_loader+1)==len(train_loader)):

                    num_grad_update+=1
                    print(f"Update gradient {num_grad_update} time")
                    #Déscaler les gradients avant clipping
                    scaler.unscale_(state.optim_enco)
                    scaler.unscale_(state.optim_deco)

                    #Gradient clipping
                    torch.nn.utils.clip_grad_norm_(state.model_enco.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(state.model_deco.parameters(), 1.0)

                    #Mise à jour des paramètres
                    scaler.step(state.optim_enco)
                    scaler.step(state.optim_deco)
                    scaler.update()

                    
                    #Réinitialisation des gradients
                    state.optim_enco.zero_grad()
                    state.optim_deco.zero_grad()
            
            #state.optim_deco.step()
            #state.optim_enco.step()
            t1 = time.time()
            print(f"---step={step:<10d} ---step_train_loader={step_train_loader:<10d} ---- time={t1-t0} ----train ElBO={loss_mse.detach().cpu().numpy():.4f}\
                   ----train DPO={loss_DPO.detach().cpu().numpy():.4f} ----train loss={loss.detach().cpu().numpy():.4f} --- LRE = {state.scheduler_enco.get_last_lr()[0]:.4f}\
                    -- LRD = {state.scheduler_deco.get_last_lr()[0]:.4f} ")
            #break
            
            

        ## Step the scheduler
        state.scheduler_enco.step()
        state.scheduler_deco.step()


        #-------checkpoint--------------------------------------------------
        if total_train_loss < state.best_valid_loss:
            state.best_valid_loss = total_train_loss
            counter=0
            save_checkpoint(state,tmp_path, enco_deco_DPO_path)
            print(f"******* step_train_loader={step_train_loader}  total_train_loss_per_loader={total_train_loss}")

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping déclenché !")
                break
            
                
        #------------------- Eval --------------
        if step % cfg.log_interval == 0:

            train_losses_last.append(total_train_loss.item())

            
    
            with torch.no_grad():

                batch=next(test_loader_cycle)
                state.model_enco.eval()
                state.model_deco.eval()

                x=batch[0]
                u=batch[1]

                x=x.to(device)
                u=u.to(device)
                target=u
            
                batch_size_test,time_step,space_step,_=x.shape #u.shape
                
                x=rearrange(x, "b T N d -> (b T) N d")
                u=rearrange(u, "b T N d -> (b T) N d")

                sample,means,log_var_square,_=state.model_enco(x,u)
                out=state.model_deco(sample,x)

                out=rearrange(out,"(b T) N d -> b T N d",b=batch_size_test)
                loss_mse=loss_model(means,log_var_square,target,out,cfg.KL_weight)


                # start of DPO

                u=rearrange(u,"(b T) N d -> b T N d",b=batch_size_test)
                x=rearrange(x,"(b T) N d -> b T N d",b=batch_size_test)

                pertu=np.random.choice(pertu_dev_set)
                pertubation=pertu*torch.randn( cfg.num_sample,*u.shape,device=device) #shape (E, b, T, N, d)


                u=u.unsqueeze(0)+ pertubation              #shape (E, b, T, N, d)
                u=rearrange(u,"E b T N d -> (E b T) N d")

                x=x.expand(cfg.num_sample, *x.shape)            #shape (E, b, T, N, d)
                x=rearrange(x,"E b T N d -> (E b T) N d")

                sample,_,_,_=state.model_enco(x,u) #shape ((E, batch,T),M,h)
                out=state.model_deco(sample,x)   # shape ((E, batch,T),N,d)

                out=rearrange(out,"(E b T) N d -> E b T N d",E=cfg.num_sample,b=batch_size_test)
                winner,loser=winLos(out,target.unsqueeze(0),cfg.dx,cfg.mass_weight,cfg.energy_weight,cfg.grad_weight,cfg.boundary_weight) # shape ( b, (T-1),N,d)
                        
                winner_score_a=state_metrics.model_attn(winner) #(b,T)

                loser_score_a=state_metrics.model_attn(loser)

                loss_DPO=-torch.log(cfg.eps+ sigmoid(winner_score_a-loser_score_a))
                loss_DPO=loss_DPO.mean()

                #FINAL LOSS
                loss=loss_mse+cfg.DPO_weight*loss_DPO

                test_losses_last.append(loss.item() )

                # ----------- Sauvegarde Loosses (On peut de passer de ca)
                np.save( train_losses_enco_deco_DPO_path, np.array(train_losses_last))
                np.save(test_losses_enco_deco_DPO_path, np.array(test_losses_last))

                # -------------------- Sauvegarder le dernier model -----------------
                save_checkpoint(state,tmp_path, last_enco_deco_DPO_path)

        
        print(f"test mse loss= {loss_mse.item()} --test DPO loss={loss_DPO.item()} ---- test loss={loss.item()}")

        #assert False

    print("End optimization")