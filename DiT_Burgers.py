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
from torch import einsum, nn

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

from DiT import *
from utilis import *


def add_args(parser):

    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=582838)

    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=2000) 
    parser.add_argument("--log_interval", type=int, default=2)


    parser.add_argument("--x_dim",type=int,default=1,help="dimension of x")
    parser.add_argument("--u_dim",type=int,default=1,help="dimension of u")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--global_batch_size", type=int, default=512,help="Taille batch globale")
    parser.add_argument("--mini_batch_size", type=int, default=128,help="Taille mini-batch pour éviter dépassement mémoire")


    parser.add_argument("--h", type=int, default=8,help="reduced dimension for bottleneck")
    parser.add_argument("--M", type=int, default=32,help="Reduction of x_dimension")
    parser.add_argument("--num_refinement_steps", type=int, default=3)
    parser.add_argument("--min_noise_std",type=float,default=1e-2)


    parser.add_argument("--save_dir", type=str, default="checkpoints")  #checkpoint




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
    print(f"Checkpoint sauvegardé avec succès : {final_path}")


class State_DiT(object):
    def __init__(self,model,optim,scheduler):
        self.model=model
        self.optim=optim
        self.scheduler=scheduler
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
    start_time=time.time()
    

    parser=argparse.ArgumentParser()
    add_args(parser)


    cfg=parser.parse_args()

    cfg.accumulation_steps = cfg.global_batch_size // cfg.mini_batch_size

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
        data = f["test/pde_250-100"][:]
    
    test_data=Dataset_Burgers(data,cfg.x_dim,x_min=0,x_max=16,t_min=0,t_max=4)

    kwargs = {"num_workers": 4, "pin_memory": True} if cfg.use_gpu else {}
    train_loader=DataLoader(dataset=train_data, batch_size=cfg.mini_batch_size, shuffle=False,**kwargs)
    test_loader=DataLoader(dataset=test_data, batch_size=cfg.test_batch_size, shuffle=False,**kwargs)
    test_loader_cycle=cycle(test_loader)


    # For checkpoint
    checkpoint_dir=Path(parent_dir/ cfg.save_dir)
    enco_deco_path= checkpoint_dir/"checkpoint.pt"
    assert enco_deco_path.exists() and enco_deco_path.is_file(), "No encoder decoder saved"

    

    ckpt_path = checkpoint_dir / "checkpoint_DiT.pt"
    tmp_path= checkpoint_dir / "checkpoint_DiT.tmp"
    last_model_path=checkpoint_dir / "checkpoint_DiT_lastModel.pt"

    train_losses_path=checkpoint_dir / "losses_train_DiT.npy"
    test_losses_path=checkpoint_dir / "losses_test_DiT.npy"
    test_latent_losses_path=checkpoint_dir / "losses_test_latent_DiT.npy"

    
    

    f = Path(ckpt_path)
    print("resolve==============",f.resolve())

    ##-------------------------LOAD DiT Model -----------------------------


    state_enco_deco=torch.load(enco_deco_path,weights_only=False)


    if ckpt_path.exists() and ckpt_path.is_file():
        with ckpt_path.open("rb") as fp:
            state=torch.load(fp, weights_only=False) #on recommence depui s l e modele sauvegarde
            #print(state)
            print("-------------------telechargemnt model DiT reussi----------------------")

    else:
        model=DiT(
        input_size=cfg.h,
        num_tokens=cfg.M,
        in_channels=4,
        hidden_size=128,  # 192,#1152,
        depth=4,  # 4
        num_heads=4,  # 6
        mlp_ratio=4.0,  # 4.0
        learn_sigma=False,
        )

        model.to(device)
        optimizer_DiT=torch.optim.Adam(model.parameters() ,lr=cfg.learning_rate  )

        epoch_init=0

        scheduler_DiT = CosineAnnealingLR(optimizer_DiT, T_max=cfg.max_iterations, eta_min=1e-5)
        
        epoch_init=0
        state=State_DiT(model,optimizer_DiT,scheduler_DiT) #best_valid_loss=np.inf and epoch_init=0

        

    # _____________LOAD losses-------------------------------
    if  train_losses_path.exists() and train_losses_path.is_file(): #It is enough to consider just one
        
        train_losses=np.load(train_losses_path) #
        test_losses=np.load(test_losses_path)
        test_losses_latent=np.load(test_latent_losses_path)

        train_losses=train_losses.tolist()
        test_losses=test_losses.tolist()
        test_losses_latent=test_losses_latent.tolist()


        print("-------------------telechargemnt loss reussi----------------------")

    else:    
        train_losses=[]
        test_losses=[]
        test_losses_latent=[]
    


    # cfg.num_refinement_steps =3
    min_noise_std = cfg.min_noise_std  # 2e-6
    betas = [
        min_noise_std ** (k / cfg.num_refinement_steps)
        for k in reversed(range(cfg.num_refinement_steps + 1))
    ]

    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_refinement_steps + 1,
        trained_betas=betas,
        prediction_type="v_prediction",
        clip_sample=False,
    )
    time_multiplier = 1000 / cfg.num_refinement_steps


    
    counter=0
    patience=200
    
    t0=time.time()
    
    
    num_grad_update=0
    print("Starting of optimizing")

    for step in range(state.epoch,cfg.max_iterations): 
        total_train_loss=0
        state.epoch=step
        #for step in range(1):
        

        print(f"---------{step}/{cfg.max_iterations}--------------")
        for step_train_loader,batch in enumerate(train_loader):

            state.model.train()

            

            

            x=batch[0] #shape (Batch,T,N,1)
            u=batch[1]  #shape (Batch,T,N,1)
            x=x.to(device)
            u=u.to(device)
            

            batch_size_tr,time_step,space_step,_=x.shape #u.shape
            x=rearrange(x, "b T N d -> (b T) N d")
            u=rearrange(u, "b T N d -> (b T) N d")

            with autocast(enabled=False): # mixed precision
                with torch.no_grad():
                    state_enco_deco.model_enco.eval()
                    sample,_,_,_=state_enco_deco.model_enco(x,u) #shape ((batch,T),M,h)

                    sammple_amenaged=rearrange(sample,"(b T) N d -> b T N d",b=batch_size_tr)

                total_train_loss_per_time=0    
                for t in range(time_step-1):
                    sample_prev=sammple_amenaged[:,t,:,:]  #sample_prev shape (b,N,d)
                    sample_cur=sammple_amenaged[:,t+1,:,:] #sample_cur

                    k = torch.randint(
                        0,
                        scheduler.config.num_train_timesteps,
                        (sample_prev.shape[0],),
                        device=sample_prev.device,
                    ) #shape (batch,)

                    # scheduler.alphas_cumprod: tensor([1-2e-6, 0.9998, 0.9872, 0.0000])

                    noise_factor = scheduler.alphas_cumprod.to(sample_prev.device)[k]
                    noise_factor = noise_factor.view(-1, *[1 for _ in range(sample_prev.ndim - 1)]) #shape: (batch,1,1)
                    signal_factor = 1 - noise_factor
                    noise = torch.randn_like(sample_cur)

                    sample_noised = scheduler.add_noise(sample_cur, noise, k)


                    pred = state.model(torch.cat([sample_prev, sample_noised], dim=1), k * time_multiplier)
                    target = (noise_factor**0.5) * noise - (signal_factor**0.5) * sample_cur
                    # print('pred', pred.shape, target.shape, x_in.shape)
                    loss = F.mse_loss(pred ,target)  # self.train_criterion(pred, target

                    #state.optim_DiT.zero_grad()
                    

                    total_train_loss_per_time+= loss/(time_step-1)

                    if t%50==0:
                        print(f"---step={step:<10d}--- t={t} ---step_train_loader={step_train_loader:<10d}---- loss_t={loss.item()}  ----loss_per_time={total_train_loss_per_time.item()} ---- LRE = {state.scheduler.get_last_lr()[0]:.6f}")
                    
                 #backward   
                scaler.scale(total_train_loss_per_time).backward() 

                total_train_loss+=total_train_loss_per_time.item()/len(train_loader)

            

            if ((step_train_loader+1)% (cfg.accumulation_steps)==0) or ((step_train_loader+1)==len(train_loader)):
                num_grad_update+=1
                print(f"Update gradient {num_grad_update} time")
                #Déscaler les gradients avant clipping
                scaler.unscale_(state.optim)
                

                #Gradient clipping
                clip_grad_norm_(state.model.parameters(), 1.0)
                

                #Mise à jour des paramètres
                scaler.step(state.optim)
                scaler.update()

                

                #Réinitialisation des gradients
                state.optim.zero_grad()
                
            
            
            print(f"++++++++++step={step:<10d} ---step_train_loader={step_train_loader:<10d}  ----loss_per_time={total_train_loss_per_time.item()}--- train_loss={total_train_loss}---- LRE = {state.scheduler.get_last_lr()[0]:.6f}")
            

        
        ## Step the scheduler
        state.scheduler.step()


        #-------checkpoint--------------------------------------------------
        if total_train_loss < state.best_valid_loss:
            state.best_valid_loss = total_train_loss
            counter=0
            save_checkpoint(state,tmp_path, ckpt_path)
            print(f"******* step_train_loader={step_train_loader}  total_train_loss_per_loader={total_train_loss}")

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping déclenché !")
                break


    
        #------------------- Eval --------------
        if step % cfg.log_interval == 0:
            train_losses.append(total_train_loss)

            
    
            with torch.no_grad():
                
             

                batch=next(test_loader_cycle)
                state_enco_deco.model_enco.eval()
                state_enco_deco.model_deco.eval()
                state.model.eval()

                x=batch[0]
                u=batch[1]
                x=x.to(device)
                u=u.to(device)
                target=u #batch (b T N d)

                batch_size_test,time_step_test,space_step,_=x.shape
                x=rearrange(x, "b T N d -> (b T) N d")
                u=rearrange(u, "b T N d -> (b T) N d")

                
                sample,_,_,_=state_enco_deco.model_enco(x,u) #shape ((batch,T),M,h)

                sammple_amenaged=rearrange(sample,"(b T) N d -> b T N d",b=batch_size_test)

                total_test_loss_per_time=0  
                y=[]  
                for t in range(time_step_test - 1 ):
                    
                    sample_prev=sammple_amenaged[:,t,:,:]  #sample_prev shape (b,N,d)
                    sample_cur=sammple_amenaged[:,t+1,:,:] #sample_cur
                    y_noised = torch.randn_like(sample_cur)  # , dtype=sample_cur.dtype, device=sample_cur.device


                    for k in scheduler.timesteps:
                        timess = (
                            torch.zeros(
                                size=(sample_cur.shape[0],), dtype=sample_cur.dtype, device=sample_cur.device
                            )
                            + k
                        )
                        pred = state.model(
                            torch.cat([sample_prev, y_noised], dim=1), timess * time_multiplier
                        )
                        y_noised = scheduler.step(pred, k, y_noised).prev_sample # shape (b,M,h)
                        
                    loss_test=F.mse_loss(y_noised,sample_cur)
                    total_test_loss_per_time+=loss_test/(time_step_test - 1 )
                    y.append( y_noised.unsqueeze(1)) 
                    #print("len de y", len(y))

                test_losses_latent.append(total_test_loss_per_time.item())
                
                y=torch.cat(y,dim=1)  # # shape (b,T-1,M,h)
                #print("y before cat ",y.shape)
                y=rearrange(y, "b T M h -> (b T) M h")

                x_out=rearrange(x, "(b T) M h -> b T M h",b=batch_size_test)
                x_out=rearrange(x_out[:,1:,:,:], "b T M h -> (b T) M h")
                #print("y.shape",y.shape)
                #print("x_out.shape",x_out.shape)
                out=state_enco_deco.model_deco(y,x_out) # shape (b,T-1,N,d)
                out=rearrange(out, "(b T) M h -> b T M h",b=batch_size_test)
                loss_gen= F.mse_loss(target[:,1:,:,:],out)
                test_losses.append(loss_gen.item())

                # ----------- Sauvegarde Loosses (On peut de passer de ca)
                np.save(train_losses_path, np.array(train_losses))
                np.save(test_losses_path, np.array(test_losses))
                np.save(test_latent_losses_path, np.array(test_losses_latent))
    
    

                save_checkpoint(state,tmp_path, last_model_path) # We save the last DiT_model

    print("End optimization")







                
                



