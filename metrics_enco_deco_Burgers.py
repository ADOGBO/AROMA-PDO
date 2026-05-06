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


from utilis import Encoder,Decoder
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
    parser.add_argument("--global_batch_size", type=int, default=128,help="Taille batch globale")
    parser.add_argument("--mini_batch_size", type=int, default=16,help="Taille mini-batch pour éviter dépassement mémoire")


    parser.add_argument("--depth", type=int, default=2,help="depth of tranformer: number of transformer block")
    parser.add_argument("--input_size", type=int, default=1,help="input_dim")
    parser.add_argument("--num_heads", type=int, default=4,help="number of heads in attention")
    parser.add_argument("--output_size", type=int, default=1,help="output dim for metric")
    parser.add_argument("--frequency_embedding_size", type=int, default=256,help="frequency_embedding_size")
    parser.add_argument("--x_space", type=str, choices=["regular","irregular"],default="regular",help="whether the space is regular or not")

    parser.add_argument("--num_sample", type=int, default=10,help="number of samples of trajectories")
    #parser.add_argument("--num_point", type=int, default=1,help="output dim for metric")


    parser.add_argument("--mass_weight", type=float, default=100.,help="the mass weight in metric")
    parser.add_argument("--grad_weight", type=float, default=10.,help="the gradient weight in metric")
    parser.add_argument("--energy_weight", type=float, default=1.,help="the energy weight in metric")
    parser.add_argument("--boundary_weight", type=float, default=0.1,help="the boundary condition weight in metric")
    parser.add_argument("--eps", type=float, default=1e-32,help="epsilon in the log of the metric")

    #parser.add_argument("--pertu", type=float, default=0.1,help="the standard deviation of the pertubation ")
    parser.add_argument("--pertu_deviation_set", type=list, default=[0.1,0.2,0.3,0.4,0.5],help="the set standard deviation of the pertubation ")
    parser.add_argument("--ratio_target", type=float, default=0.1,help="the ratio of target in the winner")

    

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

    cfg.hidden_size=int(4*cfg.num_heads)
    
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

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    #pertu_dev_set=torch.tensor(cfg.pertu_deviation_set, device=device,dtype=torch.float32)
    pertu_dev_set=cfg.pertu_deviation_set
    




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


    # For checkpoint
    checkpoint_dir=Path(parent_dir/ cfg.save_dir)
    enco_deco_path= checkpoint_dir/"checkpoint.pt"

    assert enco_deco_path.exists() and enco_deco_path.is_file(), "No encoder decoder saved"
    

    

    ckpt_path = checkpoint_dir / "checkpoint_metrics_encoDeco.pt"
    tmp_path= checkpoint_dir / "checkpoint_metrics.tmp"
    last_model_path=checkpoint_dir / "checkpoint_metrics_encoDecoLastModel.pt"

    train_losses_last_path=checkpoint_dir / "losses_train_metrics_enoDeco_l.npy"
    train_losses_mean_path=checkpoint_dir / "losses_train_metrics_encoDeco_m.npy"
    train_losses_attn_path=checkpoint_dir / "losses_train_metrics_encoDeco_a.npy"

    test_losses_last_path=checkpoint_dir / "losses_test_metrics_encoDeco_l.npy"
    test_losses_mean_path=checkpoint_dir / "losses_test_metrics_encoDeco_m.npy"
    test_losses_attn_path=checkpoint_dir / "losses_test_metrics_encoDeco_a.npy"

    
    
    

    f = Path(ckpt_path)
    print("resolve==============",f.resolve())

    ##-------------------------LOAD DiT Model -----------------------------


    state_enco_deco=torch.load(enco_deco_path,weights_only=False)


    if ckpt_path.exists() and ckpt_path.is_file():
        with ckpt_path.open("rb") as fp:
            state=torch.load(fp, weights_only=False) #on recommence depui s l e modele sauvegarde
            #print(state)
            print("-------------------telechargemnt model metrics reussi----------------------")

    else:
        model_mean=RewardSignal(depth=cfg.depth,
        input_size=cfg.input_size,
        num_heads=cfg.num_heads,
        hidden_size=cfg.hidden_size,
        output_size=cfg.output_size,
        num_point=cfg.num_point,
        frequency_embedding_size=cfg.frequency_embedding_size,
        reduction="mean",
        x_space=cfg.x_space
        )

        model_last=RewardSignal(depth=cfg.depth,
        input_size=cfg.input_size,
        num_heads=cfg.num_heads,
        hidden_size=cfg.hidden_size,
        output_size=cfg.output_size,
        num_point=cfg.num_point,
        frequency_embedding_size=cfg.frequency_embedding_size,
        reduction="last",
        x_space=cfg.x_space
        )

        model_attn=RewardSignal(cfg.depth,
        cfg.input_size,
        cfg.num_heads,
        cfg.hidden_size,
        output_size=cfg.output_size,
        num_point=cfg.num_point,
        frequency_embedding_size=cfg.frequency_embedding_size,
        reduction="attention",
        x_space=cfg.x_space
        )

        model_mean.to(device)
        model_last.to(device)
        model_attn.to(device)

        optimizer_mean=torch.optim.Adam(model_mean.parameters() ,lr=cfg.learning_rate  )
        optimizer_last=torch.optim.Adam(model_last.parameters() ,lr=cfg.learning_rate  )
        optimizer_attn=torch.optim.Adam(model_attn.parameters() ,lr=cfg.learning_rate  )

        

        scheduler_mean = CosineAnnealingLR(optimizer_mean, T_max=cfg.max_iterations, eta_min=1e-5)
        scheduler_last = CosineAnnealingLR(optimizer_last, T_max=cfg.max_iterations, eta_min=1e-5)
        scheduler_attn= CosineAnnealingLR(optimizer_last, T_max=cfg.max_iterations, eta_min=1e-5)
        
        
        
        epoch_init=0
        state=StateMetrics(model_mean,model_l=model_last,model_a=model_attn,
        optim_m=optimizer_mean,optim_l=optimizer_last,optim_a=optimizer_attn,
        scheduler_m=scheduler_mean,scheduler_l=scheduler_last,scheduler_a=scheduler_attn) 

        #best_valid_loss=np.inf and epoch_init=0

        

    # _____________LOAD losses-------------------------------
    if  train_losses_last_path.exists() and train_losses_last_path.is_file(): #It is enough to consider just one
        
        train_losses_last=np.load(train_losses_last_path) #
        test_losses_last=np.load(test_losses_last_path)
        
        train_losses_last=train_losses_last.tolist()
        test_losses_last=test_losses_last.tolist()
        

        print("-------------------telechargemnt loss metric last  reussi----------------------")

    else:    
        train_losses_last=[]
        test_losses_last=[]

    # _____________LOAD losses-------------------------------
    if  train_losses_mean_path.exists() and train_losses_mean_path.is_file(): #It is enough to consider just one
        
        train_losses_mean=np.load(train_losses_mean_path) #
        test_losses_mean=np.load(test_losses_mean_path)
        
        train_losses_mean=train_losses_mean.tolist()
        test_losses_mean=test_losses_mean.tolist()
        

        print("-------------------telechargemnt loss metric mean reussi----------------------")

    else:    
        train_losses_mean=[]
        test_losses_mean=[]

    # _____________LOAD losses-------------------------------
    if  train_losses_attn_path.exists() and train_losses_attn_path.is_file(): #It is enough to consider just one
        
        train_losses_attn=np.load(train_losses_attn_path) #
        test_losses_attn=np.load(test_losses_attn_path)
        
        train_losses_attn=train_losses_attn.tolist()
        test_losses_attn=test_losses_attn.tolist()
        

        print("-------------------telechargemnt loss metric attn  reussi----------------------")

    else:    
        train_losses_attn=[]
        test_losses_attn=[]
    
    counter=0
    patience=200
    
    t0=time.time()
    
    sigmoid=nn.Sigmoid()
    num_grad_update=0
    print("Starting of optimizing")

    

    for step in range(state.epoch,cfg.max_iterations): 
        total_train_loss_m=0
        total_train_loss_l=0
        total_train_loss_a=0
        state.epoch=step
        #for step in range(1):

        #Réinitialisation des gradients
        state.optim_mean.zero_grad()
        state.optim_last.zero_grad()
        state.optim_attn.zero_grad()
        

        print(f"---------{step}/{cfg.max_iterations}--------------")
        for step_train_loader,batch in enumerate(train_loader):

            state.model_mean.train()
            state.model_last.train()
            state.model_attn.train()

            x=batch[0] #shape (Batch,T,N,1)
            u=batch[1]  #shape (Batch,T,N,1)
            x=x.to(device)
            u=u.to(device)
            target=u
            x_out=x

            batch_size_tr,time_step,space_step,_=x.shape #x.shape

            # We select radomly the deviation
            pertu=np.random.choice(pertu_dev_set)
            pertubation=pertu*torch.randn( cfg.num_sample,*u.shape,device=device) #shape (E, b, T, N, d)
            
            u=u.unsqueeze(0)+ pertubation              #shape (E, b, T, N, d)
            u=rearrange(u,"E b T N d -> (E b T) N d")

            x=x.expand(cfg.num_sample, *x.shape)            #shape (E, b, T, N, d)
            x=rearrange(x,"E b T N d -> (E b T) N d")

            
            

            with autocast(enabled=False): # mixed precision
                with torch.no_grad():
                    state_enco_deco.model_enco.eval()
                    #samples=[]
                    

                    sample,_,_,_=state_enco_deco.model_enco(x,u) #shape ((E batch T),M,h)

                    x_out=x_out.expand(cfg.num_sample,*x_out.shape)
                    x_out=rearrange(x_out,"E b T M h -> (E b T) M h")

                    out=state_enco_deco.model_deco(sample,x_out) # shape (EbT,N,d)
                    out=rearrange(out, "(E b T) M h ->E b T M h",E=cfg.num_sample,b=batch_size_tr) # shape (E, b,,N,d)



                    winner,loser=winLos(out,target.unsqueeze(0),cfg.dx,cfg.mass_weight,cfg.energy_weight,cfg.grad_weight,cfg.boundary_weight) # shape ( b, T,N,d)

                    # We remplace ratio of winner with target
                    ratio_target=int(cfg.ratio_target*batch_size_tr)
                    target_ind=torch.randint(batch_size_tr, (ratio_target,), device=device)
                    winner[target_ind]=target[target_ind]

     
                winner_score_m=state.model_mean(winner) #(b,T-1)
                winner_score_l=state.model_last(winner)
                winner_score_a=state.model_attn(winner)

                loser_score_m=state.model_mean(loser)
                loser_score_l=state.model_last(loser)
                loser_score_a=state.model_attn(loser)

                loss_m=-torch.log(cfg.eps+ sigmoid(winner_score_m-loser_score_m))
                loss_m=loss_m.mean()

                loss_l=-torch.log(cfg.eps+ sigmoid(winner_score_l-loser_score_l))
                loss_l=loss_l.mean()

                loss_a=-torch.log(cfg.eps+ sigmoid(winner_score_a-loser_score_a))
                loss_a=loss_a.mean()
                

                scaler.scale(loss_m).backward()
                scaler.scale(loss_l).backward()
                scaler.scale(loss_a).backward()

                total_train_loss_m+=loss_m.detach().cpu().numpy()/len(train_loader)
                total_train_loss_l+=loss_l.detach().cpu().numpy()/len(train_loader)
                total_train_loss_a+=loss_a.detach().cpu().numpy()/len(train_loader)

            print(f"++++++++++step={step:<10d} ---pertu={pertu} ---loss_mean={loss_m.detach().cpu().numpy():.4f} ---loss_last={loss_l.detach().cpu().numpy():.4f} \
                  ---loss_attn={loss_a.detach().cpu().numpy():.4f}  LRE = {state.scheduler_attn.get_last_lr()[0]:.6f}")


            

            if ((step_train_loader+1)% (cfg.accumulation_steps)==0) or ((step_train_loader+1)==len(train_loader)):
                num_grad_update+=1
                print(f"Update gradient {num_grad_update} time")

                #Déscaler les gradients avant clipping
                scaler.unscale_(state.optim_mean)
                scaler.unscale_(state.optim_last)
                scaler.unscale_(state.optim_attn)
                

                #Gradient clipping
                clip_grad_norm_(state.model_mean.parameters(), 1.0)
                clip_grad_norm_(state.model_last.parameters(), 1.0)
                clip_grad_norm_(state.model_attn.parameters(), 1.0)
                

                #Mise à jour des paramètres
                scaler.step(state.optim_mean)
                scaler.step(state.optim_last)
                scaler.step(state.optim_attn)
                scaler.update()

                

                #Réinitialisation des gradients
                state.optim_mean.zero_grad()
                state.optim_last.zero_grad()
                state.optim_attn.zero_grad()
            
        
        
        ## Step the scheduler
        state.scheduler_mean.step()
        state.scheduler_last.step()
        state.scheduler_attn.step()


        #-------checkpoint--------------------------------------------------
        if (total_train_loss_l+total_train_loss_m+total_train_loss_a)/3 < state.best_valid_loss:
            state.best_valid_loss = (total_train_loss_l+total_train_loss_m+total_train_loss_a)/3
            counter=0
            save_checkpoint(state,tmp_path, ckpt_path)
            print(f"******* step_train_loader={step_train_loader}  total_train_loss_per_loader={(total_train_loss_l+total_train_loss_m+total_train_loss_a)/3}")

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping déclenché !")
                break


    
        #------------------- Eval --------------
        if step % cfg.log_interval == 0:
            train_losses_mean.append(total_train_loss_m)
            train_losses_last.append(total_train_loss_l)
            train_losses_attn.append(total_train_loss_a)

            
    
            with torch.no_grad():
                
             

                batch=next(test_loader_cycle)
                state_enco_deco.model_enco.eval()
                state_enco_deco.model_deco.eval()

                state.model_mean.eval()
                state.model_last.eval()
                state.model_attn.eval()

                x=batch[0]
                u=batch[1]

                x=x.to(device)
                u=u.to(device)

                x_out=x
                target=u #batch (b T N d)

                batch_size_test,time_step_test,space_step,_=x.shape

                

                pertu=np.random.choice(pertu_dev_set)
                pertubation=pertu*torch.randn( cfg.num_sample,*u.shape,device=device) #shape (E, b, T, N, d)

                u=u.unsqueeze(0)+ pertubation              #shape (E, b, T, N, d)
                u=rearrange(u,"E b T N d -> (E b T) N d")

                x=x.expand(cfg.num_sample, *x.shape)            #shape (E, b, T, N, d)
                x=rearrange(x,"E b T N d -> (E b T) N d")

                with autocast(enabled=False): # mixed precision
                    #state_enco_deco.model_enco.eval()
                    #samples=[]
                    

                    sample,_,_,_=state_enco_deco.model_enco(x,u) #shape ((E, batch,T),M,h)

                    x_out=x_out.expand(cfg.num_sample, *x_out.shape)
                    x_out=rearrange(x_out,"E b T M h -> (E b T) M h")

                    out=state_enco_deco.model_deco(sample,x_out) # shape (Eb(T),N,d)
                    out=rearrange(out, "(E b T) M h ->E b T M h",E=cfg.num_sample,b=batch_size_test) # shape (E, b, (T),N,d)

                    winner,loser=winLos(out,target.unsqueeze(0),cfg.dx,cfg.mass_weight,cfg.energy_weight,cfg.grad_weight,cfg.boundary_weight) # shape ( b, (T-1),N,d)

                    # We remplace ratio of winner with target
                    #ratio_target=int(cfg.ratio_target*batch_size_tr)
                    #target_ind=torch.randint(batch_size_tr,ratio_target,device=device)
                    #winner[target_ind]=target[target_ind]
                        
                    winner_score_m=state.model_mean(winner) #(b,T-1)
                    winner_score_l=state.model_last(winner)
                    winner_score_a=state.model_attn(winner)

                    loser_score_m=state.model_mean(loser)
                    loser_score_l=state.model_last(loser)
                    loser_score_a=state.model_attn(loser)

                    loss_m=-torch.log(cfg.eps+ sigmoid(winner_score_m-loser_score_m))
                    loss_m=loss_m.mean()

                    loss_l=-torch.log(cfg.eps+ sigmoid(winner_score_l-loser_score_l))
                    loss_l=loss_l.mean()

                    loss_a=-torch.log(cfg.eps+ sigmoid(winner_score_a-loser_score_a))
                    loss_a=loss_a.mean()

                    test_losses_attn.append(loss_a.cpu().numpy())
                    test_losses_mean.append(loss_m.cpu().numpy())
                    test_losses_last.append(loss_l.cpu().numpy())


                    # ----------- Sauvegarde Loosses (On peut de passer de ca)
                    np.save(train_losses_last_path, np.array(train_losses_last))
                    np.save(train_losses_mean_path, np.array(train_losses_mean))
                    np.save(train_losses_attn_path, np.array(train_losses_attn))

                    np.save(test_losses_last_path, np.array(test_losses_last))
                    np.save(test_losses_mean_path, np.array(test_losses_mean))
                    np.save(test_losses_attn_path, np.array(test_losses_attn))
                    
        
        

                    save_checkpoint(state,tmp_path, last_model_path) # We save the last DiT_model

                    
        

            
        

    print("End optimization")







                
                



