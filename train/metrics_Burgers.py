import os
import sys
from pathlib import Path

# Pour guiser la recherche dans des sous dossiers
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# IMPORTANT: IL FAUT SOIT IMPOTÉ AVANT TORCH,TORCH.DISTRIBUTED....
from utilis import idr_torch # Simu1D 

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

from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataset
from the_well.utils.download import well_download

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from DiT.DiT import DiT     
from utilis.utilis import State,save_checkpoint_encoDeco,load_checkpoint_encoDeco 
from utilis.utilis import State_DiT,save_checkpoint_DiT,load_checkpoint_DiT,cycle,compute_gradient_norm 

from config.paser import add_args,add_args_rewards
from config.preprocess import  Dataset_Burgers 
from encoder_decoder.enco_deco import Encoder,Decoder  

from rewards.rewards import PhysicsRewardSignal                                   
from metrics.metrics import PhysicalPatternsBurgers                  



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
    start_time=time.time()
    parser=argparse.ArgumentParser()
    add_args(parser)
    add_args_rewards(parser)
    cfg=parser.parse_args()

    #complementary variables
    cfg.start_latent_value_size=cfg.latent_dim
    cfg.slice_attn_heads=cfg.reward_attn_heads
    cfg.slice_attn_dim_head=cfg.reward_attn_dim_head

    cfg.d=32*cfg.num_enco_head
    cfg.num_heads_deco=cfg.num_enco_head
    cfg.out_dim=cfg.u_dim
    max_norm=0.1

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

    # For checkpoint Enco-Deco
    save_dir=parent_dir/"checkpoints" 

    checkpoint_dir=Path(save_dir)
    enco_deco_path= checkpoint_dir/"checkpointED.pt"
    DiT_path=checkpoint_dir/"checkpoint_DiTLast.pt"
    assert enco_deco_path.exists() and enco_deco_path.is_file(), "No encoder decoder saved"
    assert DiT_path.exists() and DiT_path.is_file(), "No DiT saved"


    # For checkpoint Rewards Model
    rewards_path = save_dir / "rewards.pt"
    tmp_path= save_dir / "checkpoint.tmp"
    last_path=save_dir/"checkpoint_rewLast.pt"
    

    train_losses_path=save_dir / "losses_train_rew.npy"
    test_losses_path=save_dir / "losses_test_rew.npy"
    winnerScorePath=save_dir/ "winnerScores.npy"
    loserScorePath=save_dir/ "loserScores.npy"
    trueScorePath=save_dir/ "trueScores.npy"
    

    f = Path(rewards_path)
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
    
    ##-------------------------LOAD/ CREATION DiT Model -----------------------------

    process_DiT=DiT(
        input_size=cfg.h,
        num_tokens=cfg.M,
        in_channels=4,
        hidden_size=128,  # 192,#1152,
        depth=4,  # 4
        num_heads=4,  # 6
        mlp_ratio=4.0,  # 4.0
        learn_sigma=False,
        )

    enco.to(device)
    process_DiT.to(device)
    deco.to(device)

    state_enco_deco=State(enco,deco,optim_enco=None,optim_deco=None,scheduler_enco=None,scheduler_deco=None)
    state_enco_deco=load_checkpoint_encoDeco(state_enco_deco, enco_deco_path, device,mode="eval")  
    print("-------------------telechargemnt model enco-deco reussi----------")

    state_DiT=State_DiT(process_DiT,optim=None,scheduler=None) #best_valid_loss=np.inf and epoch_init=0
    state_DiT=load_checkpoint_DiT(state_DiT, DiT_path, device,mode="eval")
    print("-------------------telechargemnt model DiTreussi----------")

    
    physics_pattern=PhysicalPatternsBurgers(cfg.dx) 
    
    #.REwards
    rewards=PhysicsRewardSignal(cfg.x_dim,cfg.u_dim,cfg.number_query_tokens,cfg.score_query_size,cfg.latent_dim,cfg.start_latent_value_size,
                                cfg.start_latent_query_size,cfg.num_physics_layers,cfg.num_temp_layers,cfg.length_x_coords,cfg.slice_num,
                                cross_attn_heads=cfg.reward_attn_heads,cross_attn_dim_head=cfg.reward_attn_dim_head,
                                slice_attn_heads=cfg.reward_attn_heads,slice_attn_dim_head=cfg.reward_attn_dim_head,
                                cross_attn_drop=cfg.cross_attn_drop,cross_attn_alibi_heads=cfg.cross_attn_alibi_heads,
                                slice_alibi_heads=cfg.slice_alibi_heads,mult_latent_dim=cfg.mult_latent_dim,drop=cfg.drop,same_block=cfg.same_block)
    print("in the main same_block=",cfg.same_block) 
    
    rewards.to(device)
    optimizer=torch.optim.Adam(rewards.parameters() ,lr=cfg.learning_rate  )

    epoch_init=0

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_iterations, eta_min=1e-6)
    
    epoch_init=0
    state=State_DiT(rewards,optimizer,cosine_scheduler) 


    if last_path.exists() and last_path.is_file():
        
        state=load_checkpoint_DiT(state, last_path, device) 
        
        state.best_valid_loss=np.inf
        #-------------------------------------

        
        print(f"-------------------telechargemnt rewardsreussi, epoch={state.epoch}----------------------")

    # _____________LOAD losses-------------------------------
    if  train_losses_path.exists() and train_losses_path.is_file(): 
        
        train_losses=np.load(train_losses_path) #
        test_losses=np.load(test_losses_path)
        winnerScoreList=np.load(winnerScorePath)
        loserScoreList=np.load(loserScorePath)
        trueScoreList=np.load(trueScorePath)

        train_losses=train_losses.tolist()
        test_losses=test_losses.tolist()
        winnerScoreList=winnerScoreList.tolist()
        loserScoreList=loserScoreList.tolist()
        trueScoreList=trueScoreList.tolist()

        print("-------------------telechargemnt loss reussi----------------------")

    else:    
        train_losses=[]
        test_losses=[]
        winnerScoreList=[]
        loserScoreList=[]
        trueScoreList=[]


   
    min_noise_std = cfg.min_noise_std  
    betas = [
        min_noise_std ** (k / cfg.num_refinement_steps)
        for k in reversed(range(cfg.num_refinement_steps + 1))
    ]

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_refinement_steps + 1,
        trained_betas=betas,
        prediction_type="v_prediction",
        clip_sample=False,
    )
    time_multiplier = 1000 / cfg.num_refinement_steps
    #--------------------------------DDp---------------------------------------------------
    
    # duplicate the model
    state_enco_deco.model_enco = DistributedDataParallel(state_enco_deco.model_enco, device_ids=[idr_torch.local_rank])
    state_enco_deco.model_deco = DistributedDataParallel(state_enco_deco.model_deco, device_ids=[idr_torch.local_rank])
    state_DiT.model=DistributedDataParallel(state_DiT.model, device_ids=[idr_torch.local_rank])
    state.model = DistributedDataParallel(state.model, device_ids=[idr_torch.local_rank],find_unused_parameters=True)


    #-----------------------------------------------------------------------------------------
    counter=0
    patience=1000
    
    sigmoid=nn.Sigmoid()
    num_grad_update=0

    print("Starting of optimizing")
    state_enco_deco.model_enco.eval()
    state_enco_deco.model_deco.eval()

    for step in range(state.epoch,cfg.max_iterations): 
        total_train_loss=0
        state.epoch=step

        #Réinitialisation des gradients
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

            batch_size_tr,time_step,space_step,u_dim=u.shape 
            if step<=5:
                pertu=np.random.choice(cfg.pertu_deviation_set, size=cfg.num_sample, replace=False)
            elif step>5:
                pertu=np.random.choice(cfg.pertu_deviation_set, size=cfg.num_sample, replace=True)
            if step > 50: 
                pertu=np.random.choice(cfg.pertu_deviation_set2, size=cfg.num_sample, replace=True)

            if rank==0:
                print(f"-------pertu_deviation {pertu}")

            pertu=torch.tensor(pertu,dtype=u.dtype,device=device)
            pertubation=pertu.reshape(-1,1,1,1)*torch.randn(cfg.num_sample,batch_size_tr,space_step,u_dim,device=device ) 

            u=u[:,0,...].unsqueeze(0)+ pertubation
            
            x_out=x.expand(cfg.num_sample,batch_size_tr,time_step,space_step,cfg.x_dim)  
            x_out=x_out[:,:,1:,...]
            x_in=x[0,0].expand(cfg.num_sample,batch_size_tr,space_step,cfg.x_dim)  

            x_in=rearrange(x_in, "E b N d -> (E b) N d")
            u=rearrange(u, "E b N d -> (E b) N d")

            with torch.no_grad():
                
                sample,_,_,_=state_enco_deco.model_enco(x_in,u) 

                y=[]  
                sample_prev=sample  
                
                for t in range(time_step - 1 ):
                    y_noised = torch.randn_like(sample_prev)  

                    for k in ddpm_scheduler.timesteps:
                        timess = (
                            torch.zeros(
                                size=(sample_prev.shape[0],), dtype=sample_prev.dtype, device=sample_prev.device
                            )
                            + k
                        )
                        pred = state_DiT.model(
                            torch.cat([sample_prev, y_noised], dim=1), timess * time_multiplier
                        )
                        y_noised = ddpm_scheduler.step(pred, k, y_noised).prev_sample 
                    sample_prev= y_noised
                    y.append( y_noised.unsqueeze(1)) 

                y=torch.cat(y,dim=1)  
                    
                y=rearrange(y, "b T M h -> (b T) M h") 

                
                x_out=rearrange(x_out,"E b T N d -> (E b T) N d ")
                out=state_enco_deco.model_deco(y,x_out) 

            out=rearrange(out, "(E b T) N h ->E b T N h",E=cfg.num_sample,b=batch_size_tr) 

            # Add ititial cond
            out=torch.cat( (target[:,0:1,...].expand(cfg.num_sample,batch_size_tr,1,space_step,u_dim),out), dim=2 )

            winner,loser=physics_pattern.winLos(out,target.unsqueeze(0),cfg.mass_weight,cfg.energy_weight,
                                                cfg.grad_weight,cfg.boundary_weight) 
            
            winnerIndx,loserIndx=physics_pattern.winLosIndx(out,target.unsqueeze(0),cfg.mass_weight,cfg.energy_weight,
                                                cfg.grad_weight,cfg.boundary_weight) 
            print(f" rank={rank} winner indexes={winnerIndx.tolist()} loser indexes={loserIndx.tolist()}")

            winner=winner.to(out.dtype)                               
            loser=loser.to(out.dtype)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                winner_score=state.model(x,winner) 

                loser_score=state.model(x,loser)
                assert winner_score.shape == loser_score.shape, f"{winner_score.shape} vs {loser_score.shape}"
                winner_score=winner_score.squeeze(-1).squeeze(-1)
                loser_score=loser_score.squeeze(-1).squeeze(-1) 

                loss=-F.logsigmoid(winner_score - loser_score)
                loss=loss.mean()

                t1=time.time()
                if rank==0:
                    print(f"+++ time={(t1-t0):.4f} ---step={step:<10d} ---loss={loss.item():.4f} LRE = {state.scheduler.get_last_lr()[0]:.6f}  win_score={winner_score.tolist()} los_score={loser_score.tolist()}")
                reg = 0.0001 * (winner_score**2 + loser_score**2).mean()
                loss=(loss+reg)/(cfg.accumulation_steps)

            loss.backward()

            total_train_loss+=loss.item()/len(train_loader)
            
            

            if ((step_train_loader+1)% (cfg.accumulation_steps)==0) or ((step_train_loader+1)==len(train_loader)):
                num_grad_update+=1
                if rank==0:
                    print(f"Update gradient {num_grad_update} time")

                
                #-------------------------------------------------- whether grad explod ---------------------------------------------------------------
                grad_norm = clip_grad_norm_(state.model.parameters(), max_norm=max_norm)
                grad_norm = grad_norm/ idr_torch.size
                #dist.broadcast(loss_tensor, src=0)  
                dist.all_reduce(grad_norm, op=dist.ReduceOp.SUM)
                if grad_norm > 1:
                    max_norm=0.1
                if grad_norm > 10:
                    max_norm=0.01
                if grad_norm > 50:
                    max_norm=0.001

                if grad_norm > 100:  
                    max_norm=0.0001

                if grad_norm > 1000:    # seuil d'alerte
                    assert False,"grad norm too grand"
                #-----------------------------------------------------------------------------------------------------------------
                
                print(f"---rank={rank} -----grad_norm={grad_norm:.4f}")

                state.optim.step()
                #Réinitialisation des gradients
                state.optim.zero_grad()

            
        state.scheduler.step()

        #-------checkpoint--------------------------------------------------
        loss_tensor = torch.tensor(total_train_loss, device=device)/ idr_torch.size
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        
        if loss_tensor.item() < state.best_valid_loss:
            state.best_valid_loss = loss_tensor.item()
            state.epoch=step
            counter=0

            if rank==0:
                save_checkpoint_DiT(state,tmp_path, rewards_path)
                print(f"******* step_train_loader={step_train_loader}  total_train_loss_per_loader={loss_tensor.item()}")

        else:
            counter += 1
            
            should_stop = torch.tensor(1 if counter >= patience else 0, device=device)
            dist.broadcast(should_stop, src=0) 

            if should_stop.item():
                if rank == 0:
                    print("Early stopping déclenché !")
                break  

        
        #------------------- Eval --------------
        if step % cfg.log_interval == 0:
            t3=time.time()
            if rank==0:
                train_losses.append(loss_tensor.item())
    
            with torch.no_grad():

                sampler_test.set_epoch(step)
                batch=next(test_loader_cycle)
                
                state.model.eval()

                x=batch[0] 
                u=batch[1]  
                x=x.to(device)
                u=u.to(device)
                target=u

                batch_size_test,time_step,space_step,u_dim=u.shape 
                
                pertu=np.random.choice(cfg.pertu_deviation_set, size=cfg.num_sample, replace=False)
                pertu=torch.tensor(pertu,dtype=u.dtype,device=device)
                pertubation=pertu.reshape(-1,1,1,1)*torch.randn(cfg.num_sample,batch_size_test,space_step,u_dim,device=device ) 
                u=u[:,0,...].unsqueeze(0)+ pertubation
                
                x_out=x.expand(cfg.num_sample,batch_size_test,time_step,space_step,cfg.x_dim) 
                x_out=x_out[:,:,1:,...]
                x_in=x[0,0].expand(cfg.num_sample,batch_size_test,space_step,cfg.x_dim)  

                x_in=rearrange(x_in, "E b N d -> (E b) N d")
                u=rearrange(u, "E b N d -> (E b) N d")

                sample,_,_,_=state_enco_deco.model_enco(x_in,u) 

                y=[]  
                sample_prev=sample 
                
                for t in range(time_step - 1 ):
                    
                    y_noised = torch.randn_like(sample_prev)  

                    for k in ddpm_scheduler.timesteps:
                        timess = (
                            torch.zeros(
                                size=(sample_prev.shape[0],), dtype=sample_prev.dtype, device=sample_prev.device
                            )
                            + k
                        )
                        pred = state_DiT.model(
                            torch.cat([sample_prev, y_noised], dim=1), timess * time_multiplier
                        )
                        y_noised = ddpm_scheduler.step(pred, k, y_noised).prev_sample 
                    sample_prev= y_noised
                    y.append( y_noised.unsqueeze(1)) 

                y=torch.cat(y,dim=1)  
                    
                y=rearrange(y, "b T M h -> (b T) M h") 

                
                x_out=rearrange(x_out,"E b T N d -> (E b T) N d ")
                out=state_enco_deco.model_deco(y,x_out) 

                out=rearrange(out, "(E b T) N h ->E b T N h",E=cfg.num_sample,b=batch_size_test) 
                # Add ititial cond
                out=torch.cat( (target[:,0:1,...].expand(cfg.num_sample,batch_size_test,1,space_step,u_dim),out), dim=2 )

                winner,loser=physics_pattern.winLos(out,target.unsqueeze(0),cfg.mass_weight,cfg.energy_weight,
                                                cfg.grad_weight,cfg.boundary_weight) 
                
                winner=winner.to(out.dtype)                               
                loser=loser.to(out.dtype)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    winner_score=state.model(x,winner)
                    loser_score=state.model(x,loser)
                    true_score=state.model(x,target)
                    lossTest=-F.logsigmoid(winner_score - loser_score)
                    lossTest=lossTest.mean()

                dist.all_reduce(lossTest, op=dist.ReduceOp.SUM)       
                test_losses.append(lossTest.item()/idr_torch.size)
                winnerScoreList.append(winner_score.tolist()[0])
                loserScoreList.append(loser_score.tolist()[0])
                trueScoreList.append(true_score.tolist()[0])

            t4=time.time()

               
                
            state.epoch=step
            if rank==0:
                print(f"----Eval --- time={(t4-t3):.4f}  ----- loss_eval={lossTest.item():.4f}")

                save_checkpoint_DiT(state,tmp_path, last_path) # We save the last DiT_model
                np.save(train_losses_path, np.array(train_losses))
                np.save(test_losses_path, np.array(test_losses))
                np.save(winnerScorePath,np.array(winnerScoreList))
                np.save(loserScorePath,np.array(loserScoreList))
                np.save(trueScorePath,np.array(trueScoreList))

       
    print("End optimization")
    