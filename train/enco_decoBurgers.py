import os
import sys
from pathlib import Path

# Pour guiser la recherche dans des sous dossiers
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# IMPORTANT: IL FAUT SOIT IMPOTÉ AVANT TORCH,TORCH.DISTRIBUTED....
from utilis import idr_torch # Simu2D 

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

import math
import h5py
import time
import re
import argparse

from einops import rearrange, repeat
from torch import einsum


from torch.optim.lr_scheduler import CosineAnnealingLR

from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataset
from the_well.utils.download import well_download


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from utilis.utilis import State,save_checkpoint_encoDeco,load_checkpoint_encoDeco,loss_model,cycle,compute_gradient_norm # in Simu2D
from config.paser import add_args,add_args_encoDeco
from config.preprocess import Dataset_Burgers                        
from encoder_decoder.enco_deco import Encoder,Decoder  #in Simu 2D


# Code principal
if __name__ == "__main__":

    # initialize the parallel environment
    #if False:
    dist.init_process_group(backend='nccl',\
                            init_method='env://',\
                                world_size=idr_torch.size,\
                                    rank=idr_torch.rank)
    
    # bind one GPU per process
    torch.cuda.set_device(idr_torch.local_rank)

    rank=idr_torch.rank

    parser=argparse.ArgumentParser()
    add_args(parser)
    add_args_encoDeco(parser)


    cfg=parser.parse_args()

    #complementary variables
    cfg.d=32*cfg.num_enco_head
    cfg.num_heads_deco=cfg.num_enco_head
    cfg.out_dim=cfg.u_dim
    #cfg.accumulation_steps = cfg.global_batch_size // cfg.mini_batch_size

    # beta
    beta=0.0001

    # Paramètres HPC optimisés
    torch.backends.cudnn.benchmark = True  # optimise les convolutions

    #device
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            #device=torch.device(f"cuda:{idr_torch.local_rank}")

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

    
    print("-------------- Loading train data-------------")
    current_dir = Path(__file__).resolve().parent 
    parent_dir=Path(__file__).resolve().parent.parent 
    fname = parent_dir/ "datasets" / "CE_train_E1.h5"
    with h5py.File(fname, "r") as f:
        data = f["train/pde_250-100"][:]
    
    data_train=Dataset_Burgers(data,cfg.x_dim,x_min=0,x_max=16,t_min=0,t_max=4,t_train_max=cfg.t_train_max)

    print("-------------- Loading test data-------------")
    fname = parent_dir/"datasets" / "CE_test_E1.h5"
    with h5py.File(fname, "r") as f:
        data = f["test/pde_250-100"][:]
    
    data_test=Dataset_Burgers(data,cfg.x_dim,x_min=0,x_max=16,t_min=0,t_max=4,t_train_max=cfg.t_train_max)

    #----------------------- DDP----------------------------------------------------
    #if False:
    sampler_train = DistributedSampler(data_train, shuffle=True,num_replicas=idr_torch.size, rank=idr_torch.rank)
    sampler_test = DistributedSampler(data_test, shuffle=True,num_replicas=idr_torch.size, rank=idr_torch.rank)

    kwargs = {"num_workers": 4, "pin_memory": True} if cfg.use_gpu else {}
    train_loader=DataLoader(dataset=data_train,sampler=sampler_train, batch_size=cfg.mini_batch_size, shuffle=False,**kwargs)
    test_loader=DataLoader(dataset=data_test,sampler=sampler_test, batch_size=cfg.test_batch_size, shuffle=False,**kwargs)
    test_loader_cycle=cycle(test_loader)
    #-------------------------------------------------------------------------------------


    # For checkpoint
    save_dir=parent_dir/"checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / "checkpointED.pt"
    tmp_path= save_dir / "checkpoint.tmp"
    last_path=save_dir/"checkpointED_last.pt"
    nan_path=save_dir/"checkpoint_nan.pt"

    train_losses_path=save_dir / "losses_train.npy"
    test_losses_path=save_dir / "losses_test.npy"

    grad_enco_path=save_dir / "grad_enco.npy"
    grad_deco_path=save_dir / "grad_deco.npy"

    f = Path(ckpt_path)
    print("resolve==============",f.resolve())

    #----------------------------- Definition of the model-------------------------------------------
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
    
    #--------------- Resume-------------------------------------------
    if last_path.exists() and last_path.is_file():
        state=load_checkpoint_encoDeco(state, last_path, device)  #on recommence depuis le  last modele sauvegarde
        #state.best_valid_loss=np.inf                                # We initialize the best loss
       
        print("-------------------telechargemnt model enco-deco reussi, epoch={state.epoch}----------")
    # _____________LOAD losses-------------------------------
    if  train_losses_path.exists() and train_losses_path.is_file(): #It is enough to consider just one
        
        train_losses=np.load(train_losses_path) #
        test_losses=np.load(test_losses_path)
        grad_enco=np.load(grad_enco_path)
        grad_deco=np.load(grad_deco_path)

        train_losses=train_losses.tolist()
        test_losses=test_losses.tolist()
        grad_enco=grad_enco.tolist()
        grad_deco=grad_deco.tolist()

        print("-------------------telechargemnt loss reussi----------------------")

    else:    
        train_losses=[]
        test_losses=[]
        grad_enco=[]
        grad_deco=[]
    #--------------------------------DDp---------------------------------------------------
    #if False:
    # duplicate the model
    state.model_enco = DistributedDataParallel(state.model_enco, device_ids=[idr_torch.local_rank])
    state.model_deco = DistributedDataParallel(state.model_deco, device_ids=[idr_torch.local_rank])

    #-----------------------------------------------------------------------------------------
    counter=0
    patience=2000
    num_grad_update=0
    # ----------------x_space --------------------
    # ATTENTION This x is global: vie
    
    print("Starting of optimizing")
    for step in range(state.epoch,cfg.max_iterations): 
        sampler_train.set_epoch(step)
        total_train_loss=0
        
        #for step in range(1):
        state.model_enco.train()
        state.model_deco.train()

        state.optim_enco.zero_grad()
        state.optim_deco.zero_grad()

        print(f"---------{step}/{cfg.max_iterations}--------------")
        for step_train_loader,batch in enumerate(train_loader):

            x=batch[0] #shape (Batch,T,N,1)
            u=batch[1]  #shape (Batch,T,N,1)
            x=x.to(device)
            u=u.to(device)
            target=u
            

            batch_size_tr,time_step,space_step,_=x.shape #u.shape
            x=rearrange(x, "b T N d -> (b T) N d")
            u=rearrange(u, "b T N d -> (b T) N d")
            #if rank==0:
            #    print(f"-- arrived before encoder---step_train_loader={step_train_loader:<10d}")
            sample,means,log_var_square,_=state.model_enco(x,u)
            #if rank==0:
            #    print(f"-- arrived before decoder---step_train_loader={step_train_loader:<10d}")
            out=state.model_deco(sample,x)
            out=rearrange(out,"(b T) N d -> b T N d",b=batch_size_tr)
            #if rank==0:
            #    print(f"-- arrived before loss--step_train_loader={step_train_loader:<10d}")
            loss=loss_model(means,log_var_square,target,out,beta)
            loss=loss/cfg.accumulation_steps

            #Backward pass avec scaling
            
            #print(f"-- arrived before bacward---step_train_loader={step_train_loader:<10d} --loss={loss.item()} ---rank: {rank}")
            #t0=time.time()
            loss.backward()

            """t1=time.time()
            print(f"YES rank: {rank} time={t1-t0}" )
            state.optim_enco.step()
            state.optim_deco.step()

            #Réinitialisation des gradients
            state.optim_enco.zero_grad()
            state.optim_deco.zero_grad()

            if rank==0:
                print(f"---step={step:<10d} ---step_train_loader={step_train_loader:<10d}  ----train ElBO={(loss.item()*cfg.accumulation_steps):.4f}--- LRE = {state.scheduler_enco.get_last_lr()[0]:.6f}-- LRD = {state.scheduler_deco.get_last_lr()[0]:.6f} ")
            """

            """norm_enco=compute_gradient_norm(state.model_enco)
            norm_deco=compute_gradient_norm(state.model_deco)
            if rank==0:
                print(f"-- arrived after backward---step_train_loader={step_train_loader:<10d} --grad_E={norm_enco} --grad_D={norm_deco}")"""

                
            total_train_loss+=loss.item()/len(train_loader)
            #if rank==0:
            #    print(f"---step={step:<10d} ---step_train_loader={step_train_loader:<10d}")

            if rank==0:
                print(f"---step={step:<10d} ---step_train_loader={step_train_loader:<10d}  ----train ElBO={(loss.item()*cfg.accumulation_steps):.4f}--- LRE = {state.scheduler_enco.get_last_lr()[0]:.6f}-- LRD = {state.scheduler_deco.get_last_lr()[0]:.6f} ")
            

            if ((step_train_loader+1)% (cfg.accumulation_steps)==0) or ((step_train_loader+1)==len(train_loader)):
                num_grad_update+=1
                if rank==0:
                    print(f"Update gradient {num_grad_update} time")
                state.optim_enco.step()
                state.optim_deco.step()

                #Réinitialisation des gradients
                state.optim_enco.zero_grad()
                state.optim_deco.zero_grad()
            
            

            #if step_train_loader==10:
            #    break

            #if rank==0:
            #    print(f"-- arrived after  break---step_train_loader={step_train_loader:<10d}")
            
            

        ## Step the scheduler
        state.scheduler_enco.step()
        state.scheduler_deco.step()


        #-------checkpoint--------------------------------------------------
        
        if total_train_loss < state.best_valid_loss:
            state.best_valid_loss = total_train_loss
            counter=0

            state.epoch=step
            if rank==0:
                save_checkpoint_encoDeco(state,tmp_path, ckpt_path)

            print(f"******* step_train_loader={step_train_loader}  total_train_loss_per_loader={total_train_loss}")

        else:
            counter += 1
            if counter >= patience:
                if rank==0:
                    print("Early stopping déclenché !")

                break
        

        #------------------- Eval --------------
        if step % cfg.log_interval == 0:
            # Compute the norm of the models
            with torch.no_grad():
                norm_enco=compute_gradient_norm(state.model_enco)
                norm_deco=compute_gradient_norm(state.model_deco)

            if rank==0:
                train_losses.append(total_train_loss) 
                grad_enco.append(norm_enco)
                grad_deco.append(norm_deco)
    
            with torch.no_grad():

                sampler_test.set_epoch(step)
                batch=next(test_loader_cycle)

                state.model_enco.eval()
                state.model_deco.eval()

                x=batch[0] #shape (Batch,T,N,1)
                u=batch[1]  #shape (Batch,T,N,1)
                x=x.to(device)
                u=u.to(device)
                target=u
                

                batch_size_test,time_step,space_step,_=x.shape #u.shape
                x=rearrange(x, "b T N d -> (b T) N d")
                u=rearrange(u, "b T N d -> (b T) N d")
                sample,means,log_var_square,_=state.model_enco(x,u)
                out=state.model_deco(sample,x)
                out=rearrange(out,"(b T) N d -> b T N d",b=batch_size_test)

                loss_test=loss_model(means,log_var_square,target,out,beta)

                dist.all_reduce(loss_test, op=dist.ReduceOp.SUM)        # DDP reduction
                loss_test=loss_test/ idr_torch.size

                if rank==0:
                    test_losses.append(loss_test.item() )

                # ----------- Sauvegarde Loosses (On peut de passer de ca)
                if rank==0:
                    np.save(train_losses_path, np.array(train_losses))
                    np.save(test_losses_path, np.array(test_losses))
                    np.save(grad_enco_path, np.array(grad_enco))
                    np.save(grad_deco_path, np.array(grad_deco))

                    save_checkpoint_encoDeco(state,tmp_path,last_path ) # We save the last DiT_model
                    
        #assert False
                
    print("End optimization")