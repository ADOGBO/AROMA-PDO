
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
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from utilis import *



def add_args(parser):

    parser.add_argument("--x_dim",type=int,default=1,help="dimension of x")
    parser.add_argument("--u_dim",type=int,default=1,help="dimension of u")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--global_batch_size", type=int, default=128,help="Taille batch globale")
    parser.add_argument("--mini_batch_size", type=int, default=8,help="Taille mini-batch pour éviter dépassement mémoire")
    

    
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=2000) 
    parser.add_argument("--save_dir", type=str, default="checkpoints")  #checkpoint
    parser.add_argument("--log_interval", type=int, default=2)
    
    parser.add_argument("--h", type=int, default=8,help="reduced dimension for bottleneck")
    parser.add_argument("--M", type=int, default=32,help="Reduction of x_dimension")
    parser.add_argument("--k", type=int, default=16,help="lenghts of fourirer feature")
    parser.add_argument("--log_scale_min", type=int, default=0,help="log_scale_min")
    parser.add_argument("--log_scale_max", type=int, default=0,help="log_scale_max")
    parser.add_argument("--num_enco_head", type=int, default=4,help="number Attention head in encoder")
    #parser.add_argument("--d", type=int, default=8*parser.num_enco_head,help="dimension after embedding")

    #parser.add_argument("--num_heads_deco", type=int, default=parser.num_enco_head,help="number Attention head in deco")
    parser.add_argument("--attn_dropout", type=int, default=0,help="attn_dropout")
    parser.add_argument("--enco_dropout", type=int, default=0,help="enco_dropout")
    parser.add_argument("--att_dropout_deco", type=int, default=0,help="deco_dropout in attention")
    parser.add_argument("--mult_dim_ff", type=int, default=4,help="mult_dim_ff")
    parser.add_argument("--num_self_attn_deco", type=int, default=2,help="num_self_attn_deco")

    #parser.add_argument("--out_dim", type=int, default=parser.u_dim,help="num_self_attn_deco")

    parser.add_argument("--mult_dim_deco", type=int, default=4,help="mult_dim_deco")
    parser.add_argument("--hidden_dim_deco", type=int, default=128,help="mult_dim_deco for ff nn")
    parser.add_argument("--use_pi", type=bool, default=True,help="Use pi")
    parser.add_argument("--log_sampling", type=bool, default=True,help="log_sampling")
    parser.add_argument("--include_input", type=bool, default=True,help="include_input")
    parser.add_argument("--use_gelu", type=bool, default=True,help="use_gelu as activation in some part")

    parser.add_argument("--enco_geo", type=bool, default=False,help="Whether to encode Geometry")
    parser.add_argument("--include_pos_in_value", type=bool, default=False,help="Whether to include position coordinate in value when encoding value")
    parser.add_argument("--depth_deco", type=int, default=3,help="depth of decoder mlp")
    parser.add_argument("--same_self_block", type=bool, default=True,help="same self_transformer_block during decoding")

    parser.add_argument("--fourrier_feature_type",choices=["base2","random"],default="base2",help="variational mode")
    parser.add_argument("--num_fourier_feature_deco",type=int,default=3,help="number of fourrier for x queries in decoder")

    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=582838)
    #parser.add_argument("--resume", action="store_true", help="Reprendre depuis le dernier checkpoint")
    #parser.add_argument("--no_gpu", action="store_false", dest="use_gpu")
    #parser.set_defaults(use_gpu=True)

    #parser.add_argument("--data_dir", type=pathlib.Path, default="/tmp")

def loss_model(mean,log_var_square,x,x_hat,beta,return_KL=False):
    """
         mean,log_var_square,x,x_hat : (bach,seq_length,d)
    
    """
    
    KL_divergence_per_batch=0.5*torch.mean( -1-log_var_square+mean**2+ torch.exp(log_var_square),dim=[1,2] )
    
    if return_KL==False:

        return beta*torch.mean(KL_divergence_per_batch)+F.mse_loss(x_hat,x)
    else:
        return KL_divergence_per_batch, beta*torch.mean(KL_divergence_per_batch)+F.mse_loss(x_hat,x)




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
        
        

class Dataset_pde(Dataset):
    def __init__(self,data,x_dim,x_min=0,x_max=16,t_min=0,t_max=4,forescast=True):
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
        

        if forescast==False:
            self.data_new=self.data.reshape(-1,*data[0][0].shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,ind):
        return (self.x_space_expand[ind], self.data[ind])


# Code principal
if __name__ == "__main__":
    start_time=time.time()
    

    parser=argparse.ArgumentParser()
    add_args(parser)


    cfg=parser.parse_args()

    #complementary variables
    cfg.d=32*cfg.num_enco_head
    cfg.num_heads_deco=cfg.num_enco_head
    cfg.out_dim=cfg.u_dim
    cfg.accumulation_steps = cfg.global_batch_size // cfg.mini_batch_size

    


    # beta
    beta=0.0001

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
    
    train_data=Dataset_pde(data,cfg.x_dim,x_min=0,x_max=16,t_min=0,t_max=4,forescast=False)

    print("-------------- Loading test data-------------")
    fname = parent_dir/"dataset" / "CE_test_E1.h5"
    with h5py.File(fname, "r") as f:
        data = f["test/pde_250-100"][:]
    
    test_data=Dataset_pde(data,cfg.x_dim,x_min=0,x_max=16,t_min=0,t_max=4,forescast=False)

    kwargs = {"num_workers": 4, "pin_memory": True} if cfg.use_gpu else {}
    train_loader=DataLoader(dataset=train_data, batch_size=cfg.mini_batch_size, shuffle=False,**kwargs)
    test_loader=DataLoader(dataset=test_data, batch_size=cfg.test_batch_size, shuffle=False,**kwargs)
    test_loader_cycle=cycle(test_loader)


    # For checkpoint
    #save_dir = Path(cfg.save_dir)
    save_dir=parent_dir/cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "checkpoint.pt"
    tmp_path= save_dir / "checkpoint.tmp"
    nan_path=save_dir/"checkpoint_nan.pt"

    train_losses_path=save_dir / "losses_train_DiT.npy"
    test_losses_path=save_dir / "losses_test_DiT.npy"
    


    

    f = Path(ckpt_path)
    print("resolve==============",f.resolve())


    if ckpt_path.exists() and ckpt_path.is_file():
        with ckpt_path.open("rb") as fp:
            state=torch.load(fp, weights_only=False) #on recommence depui s l e modele sauvegarde
            #print(state)
            print("-------------------telechargemnt reussi----------------------")

    else:
        enco=Encoder(cfg.x_dim,cfg.u_dim,cfg.M,cfg.d,cfg.h,cfg.log_scale_min,cfg.log_scale_max,cfg.k,
                device,cfg.num_enco_head,attn_dropout=cfg.attn_dropout,
                enco_dropout=cfg.enco_dropout,mult_dim_ff=cfg.mult_dim_ff, use_pi=cfg.use_pi,log_sampling=cfg.log_sampling,
                include_input=cfg.include_input,use_gelu=cfg.use_gelu,enco_geo=cfg.enco_geo,include_pos_in_value=cfg.include_pos_in_value,
                fourrier_feature_type=cfg.fourrier_feature_type)
        
       
        

        deco=Decoder(cfg.h,cfg.d,cfg.k,cfg.num_self_attn_deco,cfg.mult_dim_deco,cfg.x_dim,cfg.out_dim,
                    cfg.hidden_dim_deco,cfg.log_scale_min,cfg.log_scale_max,device,
                    use_gelu_deco=cfg.use_gelu,num_heads_deco=cfg.num_enco_head, 
                    att_dropout_deco=cfg.att_dropout_deco,fourrier_feature_type=cfg.fourrier_feature_type,
                    num_fourier_feature_deco=cfg.num_fourier_feature_deco,depth_deco=cfg.depth_deco,
                    use_pi=cfg.use_pi,log_sampling=cfg.log_sampling,include_input=cfg.include_input,same_self_block=cfg.same_self_block)
        
        
        

            
        enco.to(device)
        deco.to(device)
        optimizer_enco=torch.optim.Adam(enco.parameters() ,lr=cfg.learning_rate  )
        optimizer_deco=torch.optim.Adam( deco.parameters() ,lr=cfg.learning_rate  )

        epoch_init=0

        scheduler_enco = CosineAnnealingLR(optimizer_enco, T_max=cfg.max_iterations, eta_min=1e-5)
        scheduler_deco = CosineAnnealingLR(optimizer_deco, T_max=cfg.max_iterations, eta_min=1e-5)

        
        state=State(enco,deco,optimizer_enco,optimizer_deco,scheduler_enco,scheduler_deco)


    # _____________LOAD losses-------------------------------
    if  train_losses_path.exists() and train_losses_path.is_file(): #It is enough to consider just one
        
        train_losses=np.load(train_losses_path) #
        test_losses=np.load(test_losses_path)
        

        print("-------------------telechargemnt loss reussi----------------------")

    else:    
        train_losses=[]
        test_losses=[]
        
    
    
    
    

    
    counter=0
    patience=200
    
    t0=time.time()
    
    
    num_grad_update=0
    print("Starting of optimizing")
    for step in range(state.epoch,cfg.max_iterations): 
        total_train_loss=0
        state.epoch=step
        #for step in range(1):
        state.model_enco.train()
        state.model_deco.train()

        state.optim_enco.zero_grad()
        state.optim_deco.zero_grad()

        print(f"---------{step}/{cfg.max_iterations}--------------")
        for step_train_loader,batch in enumerate(train_loader):
            #batch=next(train_ds)

            x=batch[0]
            u=batch[1]
            x=x.to(device)
            u=u.to(device)
            target=u

            batch_size_tr,time_step,space_step,_=x.shape #u.shape
            x=rearrange(x, "b T N d -> (b T) N d")
            u=rearrange(u, "b T N d -> (b T) N d")

            

            
            #state.model_enco.zero_grad()
            #state.model_deco.zero_grad()

            with autocast(enabled=False): # mixed precision
                sample,means,log_var_square,_=state.model_enco(x,u)
                out=state.model_deco(sample,x)

                out=rearrange(out,"(b T) N d -> b T N d",b=batch_size_tr)
                loss=loss_model(means,log_var_square,target,out,beta)

                ############################ CHECK NAN ####################################
                if torch.isnan(loss.detach().cpu()):
                    checkNaN=CheckNaN(state.model_enco, state.model_deco, state.optim_enco, state.optim_deco,state.scheduler_enco,state.scheduler_deco)

                    # Add les valeurs
                    checkNaN.add_attribut(x,target,sample,means,log_var_square,out,state.scheduler_enco.get_last_lr()[0],state.scheduler_deco.get_last_lr()[0])

                    
                    # Verifier nan dans les parametre 
                    for name, param  in list(state.model_enco.named_parameters())+list(state.model_deco.named_parameters()):
                        if torch.isnan(param).any():
                            print("NaN dans les poids :", name)
                            checkNaN.param_nan(name,param)
                    
                    # Verifier Nan dans les grad
                    for name, param in list(state.model_enco.named_parameters())+list(state.model_deco.named_parameters()):
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print("NaN dans le gradient :", name)
                                checkNaN.add_gradNaN(name,param.grad)
                            
                            #else:
                            #    print(f"Not Nan in {name}"")
        
                        else:
                            print(f"{name} has None gradient: Absurde")





                    torch.save(checkNaN,nan_path)
                    assert not torch.isnan(loss.detach().cpu()),"Loss nan decteted"
                ####################################################################
                
                #Backward pass avec scaling

                """ A utiliser apres pour proteger l'entrainement
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()"""
    
                scaler.scale(loss).backward()

                


                total_train_loss+=loss.detach().cpu().numpy()/len(train_loader)
            #sample,means,log_var_square,_=state.model_enco(x,u)
            #out=state.model_deco(sample,x)
            #loss=loss_model(means,log_var_square,u,out,beta)
            #loss.backward()

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
            print(f"---step={step:<10d} ---step_train_loader={step_train_loader:<10d}  ----train ElBO={loss.detach().cpu().numpy():.4f}--- LRE = {state.scheduler_enco.get_last_lr()[0]:.6f}-- LRD = {state.scheduler_deco.get_last_lr()[0]:.6f} ")

            #check Nan
            

        ## Step the scheduler
        state.scheduler_enco.step()
        state.scheduler_deco.step()


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
            
                

        """state.epoch=step+1
        if step< int(0.66*(cfg.max_iterations-state.epoch)):
            with ckpt_path.open("wb") as fp:
                torch.save(state, fp)

        else:
            if loss < best_valid_loss:
                best_valid_loss = loss
                counter=0
                with ckpt_path.open("wb") as fp:
                    torch.save(state, fp)

            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping déclenché !")
                    break"""
            

            
        

        #------------------- Eval --------------
        if step % cfg.log_interval == 0:
            train_losses.append(total_train_loss.item())

            t1 = time.time()
    
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

                loss_test = loss_model(means,log_var_square,target,out,beta)
                test_losses.append(loss_test.item() )

                # ----------- Sauvegarde Loosses (On peut de passer de ca)
                np.save(train_losses_path, np.array(train_losses))
                np.save(test_losses_path, np.array(test_losses))
                    
        


    
                
    print("End optimization")
