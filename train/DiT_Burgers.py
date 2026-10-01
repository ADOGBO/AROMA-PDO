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
from pathlib import Path
import math
import h5py
import time
import re
import argparse

from einops import rearrange, repeat
from torch import einsum, nn

import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers.schedulers import DDPMScheduler
from torch.nn.utils import clip_grad_norm_

from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataset
from the_well.utils.download import well_download

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from DiT.DiT import DiT     
from utilis.utilis import State,save_checkpoint_encoDeco,load_checkpoint_encoDeco 
from utilis.utilis import State_DiT,save_checkpoint_DiT,load_checkpoint_DiT,cycle,compute_gradient_norm 

from config.paser import add_args,add_args_encoDeco, add_args_DiT
from config.preprocess import  Dataset_Burgers 
from encoder_decoder.enco_deco import Encoder,Decoder  

    
if __name__ == "__main__":


    dist.init_process_group(backend='nccl',\
                            init_method='env://',\
                                world_size=idr_torch.size,\
                                    rank=idr_torch.rank)
    
    
    torch.cuda.set_device(idr_torch.local_rank) 

    rank=idr_torch.rank
    #rank=0
    
    parser=argparse.ArgumentParser()
    add_args(parser)
    add_args_DiT(parser)
    cfg=parser.parse_args()

    #complementary variables
    cfg.d=32*cfg.num_enco_head
    cfg.num_heads_deco=cfg.num_enco_head
    cfg.out_dim=cfg.u_dim

    # Base_File
    base_path=Path(__file__).resolve().parent.parent.parent # base


    torch.backends.cudnn.benchmark = True  

    #device
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    except Exception as e:
        
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
    
    save_dir=parent_dir/"checkpoints" 

    checkpoint_dir=Path(save_dir)
    enco_deco_path= checkpoint_dir/"checkpointED.pt"
    assert enco_deco_path.exists() and enco_deco_path.is_file(), "No encoder decoder saved"

    DiT_path = save_dir / "checkpoint_DiT.pt"
    tmp_path= save_dir / "checkpoint.tmp"
    last_path=save_dir/"checkpoint_DiTLast.pt"
    

    train_losses_path=save_dir / "losses_train_DiT.npy"
    test_losses_path=save_dir / "losses_test_DiT.npy"
    test_latent_losses_path=save_dir / "losses_test_latent_DiT.npy"

    grad_DiT_path=save_dir /"grad_DiT.npy"
    l2_rela_path=save_dir /"l2_relative_norm.npy"

    f = Path(DiT_path)
    print("resolve==============",f.resolve())

    ##-------------------------LOAD Encoder-Decoder Model -----------------------------
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
    
    state_enco_deco=State(enco,deco,optim_enco=None,optim_deco=None,scheduler_enco=None,scheduler_deco=None)
    
    state_enco_deco=load_checkpoint_encoDeco(state_enco_deco, enco_deco_path, device,mode="eval") 
    print("-------------------telechargemnt model enco-deco reussi----------")


    ##-------------------------LOAD/ CREATION DiT Model -----------------------------

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


    if last_path.exists() and last_path.is_file():
        
        state=load_checkpoint_DiT(state, last_path, device) #on recommence depuis le modele sauvegarde
        #print(state)
        print("-------------------telechargemnt last model DiT reussi----------------------")

    # _____________LOAD losses-------------------------------
    if  train_losses_path.exists() and train_losses_path.is_file(): #It is enough to consider just one
        
        train_losses=np.load(train_losses_path) #
        test_losses=np.load(test_losses_path)
        test_losses_latent=np.load(test_latent_losses_path)
        grad_DiT_norm=np.load(grad_DiT_path)
        l2_rela_norm=np.load(l2_rela_path)

        train_losses=train_losses.tolist()
        test_losses=test_losses.tolist()
        test_losses_latent=test_losses_latent.tolist()
        grad_DiT_norm=grad_DiT_norm.tolist()
        l2_rela_norm=l2_rela_norm.tolist()

        print("-------------------telechargemnt loss reussi----------------------")

    else:    
        train_losses=[]
        test_losses=[]
        test_losses_latent=[]
        grad_DiT_norm=[]
        l2_rela_norm=[]
    
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

    #--------------------------------DDp---------------------------------------------------
    state_enco_deco.model_enco = DistributedDataParallel(state_enco_deco.model_enco, device_ids=[idr_torch.local_rank])
    state_enco_deco.model_deco = DistributedDataParallel(state_enco_deco.model_deco, device_ids=[idr_torch.local_rank])
    state.model = DistributedDataParallel(state.model, device_ids=[idr_torch.local_rank])


    #-----------------------------------------------------------------------------------------
    counter=0
    patience=1000
    
    num_grad_update=0

    print("Starting of optimizing")

    for step in range(state.epoch,cfg.max_iterations): 
        total_train_loss=0
        state.optim.zero_grad()

        print(f"---------{step}/{cfg.max_iterations}--------------")
        for step_train_loader,batch in enumerate(train_loader):
            t0=time.time()
            sampler_train.set_epoch(step)

            state.model.train()
            
            x=batch[0] 
            u=batch[1]  
            x=x.to(device)
            u=u.to(device)
            target=u

            batch_size_tr,time_step,space_step,_=x.shape 
            x=rearrange(x, "b T N d -> (b T) N d")
            u=rearrange(u, "b T N d -> (b T) N d")

            with torch.no_grad():
                state_enco_deco.model_enco.eval()
                sample,_,_,_=state_enco_deco.model_enco(x,u) 

                sammple_amenaged=rearrange(sample,"(b T) M d -> b T M d",b=batch_size_tr)

            sample_prev=sammple_amenaged[:,:-1,:,:]  
            sample_cur=sammple_amenaged[:,1:,:,:] 
            sample_prev=rearrange(sample_prev,"b T N d -> (b T) N d")       
            sample_cur=rearrange(sample_cur,"b T N d -> (b T) N d")

            k = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (sample_prev.shape[0],),
                device=sample_prev.device,
            ) 

            noise_factor = scheduler.alphas_cumprod.to(sample_prev.device)[k]
            noise_factor = noise_factor.view(-1, *[1 for _ in range(sample_prev.ndim - 1)]) 
            signal_factor = 1 - noise_factor
            noise = torch.randn_like(sample_cur)

            sample_noised = scheduler.add_noise(sample_cur, noise, k)
            
            pred = state.model(torch.cat([sample_prev, sample_noised], dim=1), k * time_multiplier)
            target = (noise_factor**0.5) * noise - (signal_factor**0.5) * sample_cur
            loss = F.mse_loss(pred ,target)  
            loss=loss/cfg.accumulation_steps

            t1=time.time()
            if rank==0:
                print(f"---step={step:<10d} ---time={t1-t0} ---step_train_loader={step_train_loader:<10d}---- loss_t={loss.item()*cfg.accumulation_steps}  ---- LRE = {state.scheduler.get_last_lr()[0]:.6f}")
                
            loss.backward() 
            

            if ((step_train_loader+1)% (cfg.accumulation_steps)==0) or ((step_train_loader+1)==len(train_loader)):
                num_grad_update+=1
                if rank==0:
                    print(f"Update gradient {num_grad_update} time")
                
                state.optim.step()

                
                state.optim.zero_grad()

            total_train_loss+=loss.item()/len(train_loader)     
            
        state.scheduler.step()


        #-------checkpoint--------------------------------------------------
        
        if total_train_loss < state.best_valid_loss:
            state.best_valid_loss = total_train_loss
            state.epoch=step
            counter=0

            if rank==0:
                save_checkpoint_DiT(state,tmp_path, DiT_path)
                print(f"******* step_train_loader={step_train_loader}  total_train_loss_per_loader={total_train_loss}")

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping déclenché !")
                break


    
        #------------------- Eval --------------
        if step % cfg.log_interval == 0:
            
            
            with torch.no_grad():
                norm_DiT=compute_gradient_norm(state.model)

            if rank==0:
                train_losses.append(total_train_loss) 
                grad_DiT_norm.append(norm_DiT)

            with torch.no_grad():

                sampler_test.set_epoch(step)
                batch=next(test_loader_cycle)
                x=batch[0] 
                u=batch[1]  
                x=x.to(device)
                u=u.to(device)
                target=u[:,1:,...]
                x_out=x[:,1:,...]  
                
                state_enco_deco.model_enco.eval()
                state_enco_deco.model_deco.eval()
                state.model.eval()

                batch_size_test,time_step,space_step,_=x.shape 
                x=rearrange(x, "b T N d -> (b T) N d")
                u=rearrange(u, "b T N d -> (b T) N d")
                
                sample,_,_,_=state_enco_deco.model_enco(x,u) 

                sammple_amenaged=rearrange(sample,"(b T) N d -> b T N d",b=batch_size_test)

                total_test_loss_per_time=0  
                y=[]  
                sample_generated=sammple_amenaged[:,0,...]
                for t in range(time_step - 1): 
                    
                    sample_prev=sammple_amenaged[:,t,:,:]  
                    sample_cur=sammple_amenaged[:,t+1,:,:] 
                    y_noised = torch.randn_like(sample_cur)  
                
                    for k in scheduler.timesteps:
                        timess = (
                            torch.zeros(
                                size=(sample_cur.shape[0],), dtype=sample_cur.dtype, device=sample_cur.device
                            )
                            + k
                        )
                        

                        pred = state.model(
                            torch.cat([sample_generated, y_noised], dim=1), timess * time_multiplier
                        )
                        y_noised = scheduler.step(pred, k, y_noised).prev_sample 
                        
                    loss_test=F.mse_loss(y_noised,sample_cur)
                    total_test_loss_per_time+=loss_test/((time_step - 1 ))
                    
                    sample_generated=y_noised
                    y.append( y_noised.unsqueeze(1)) 

                    

                dist.all_reduce(total_test_loss_per_time, op=dist.ReduceOp.SUM)        

                test_losses_latent.append(total_test_loss_per_time.item())
                
                y=torch.cat(y,dim=1)  
                y=rearrange(y, "b T M h -> (b T) M h")
                x_out=rearrange(x_out,"b T M h -> (b T) M h") 
                out=state_enco_deco.model_deco(y,x_out) 

                out=rearrange(out, "(b T) N d -> b T N d",b=batch_size_test)
                loss_gen= F.mse_loss(target,out)

                dist.all_reduce(loss_gen, op=dist.ReduceOp.SUM)        
                test_losses.append(loss_gen.item())
                l2_relative_norm=( (torch.norm((target[0] - out[0]),p=2,dim=(-3,-2,-1)) ) /(torch.norm( target[0],p=2,dim=(-3,-2,-1)) ) )*100
                dist.all_reduce(l2_relative_norm, op=dist.ReduceOp.SUM)
                l2_rela_norm.append(l2_relative_norm.item())

                

                if rank==0:
                    save_checkpoint_DiT(state,tmp_path, last_path) 
                    np.save(train_losses_path, np.array(train_losses))
                    np.save(test_losses_path, np.array(test_losses))
                    np.save(test_latent_losses_path, np.array(test_losses_latent))
                    np.save(l2_rela_path, np.array(l2_rela_norm))
                    np.save(grad_DiT_path, np.array(grad_DiT_norm))

        
    print("End optimization")