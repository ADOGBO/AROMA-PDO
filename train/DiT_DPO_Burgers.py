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
import copy
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

from DiT.DiT import DiT     #Simu1D
from utilis.utilis import State,save_checkpoint_encoDeco,load_checkpoint_encoDeco #Simu1D
from utilis.utilis import State_DiT,save_checkpoint_DiT,load_checkpoint_DiT,cycle,compute_gradient_norm #Simu1D

from config.paser import add_args,add_args_DiTDPO
from config.preprocess import  Dataset_Burgers 
from encoder_decoder.enco_deco import Encoder,Decoder  #in Simu 1D

from rewards.rewards import PhysicsRewardSignal                                    #Simu1D
from metrics.metrics import PhysicalPatternsBurgers                  #Simu1D
from utilis.unroll import unroll_samples


def grad_norm(grads):
    return torch.sqrt(
        sum(
            (g.detach() ** 2).sum()
            for g in grads
            if g is not None
        )
    )

if __name__ == "__main__":

    # initialize the parallel environment
    #if False:
    dist.init_process_group(backend='nccl',\
                            init_method='env://',\
                                world_size=idr_torch.size,\
                                    rank=idr_torch.rank)
    
    # bind one GPU per process
    """ 
        The following + "device = torch.device("cuda")" is equiv to "device = torch.device(f"cuda:{idr_torch.local_rank}"
    """
    torch.cuda.set_device(idr_torch.local_rank) 
    
    rank=idr_torch.rank
    start_time=time.time()
    parser=argparse.ArgumentParser()
    add_args(parser)
    add_args_DiTDPO(parser)
    cfg=parser.parse_args()

    #complementary variables
    cfg.start_latent_value_size=cfg.latent_dim
    cfg.slice_attn_heads=cfg.reward_attn_heads
    cfg.slice_attn_dim_head=cfg.reward_attn_dim_head

    cfg.d=32*cfg.num_enco_head
    cfg.num_heads_deco=cfg.num_enco_head
    cfg.out_dim=cfg.u_dim

    

    # Base_File
    base_path=Path(__file__).resolve().parent.parent # base

    
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
    save_dir=parent_dir/"checkpoints" # We suppose that this flder exists

    checkpoint_dir=Path(save_dir)
    enco_deco_path= checkpoint_dir/"checkpointED.pt"
    DiT_path=checkpoint_dir/"checkpoint_DiTLast.pt"
    rewards_path=checkpoint_dir/ "rewardsBest.pt"
    assert enco_deco_path.exists() and enco_deco_path.is_file(), "No encoder decoder saved"
    assert DiT_path.exists() and DiT_path.is_file(), "No DiT saved"
    assert rewards_path.exists() and rewards_path.is_file(), "No DiT saved"
    
    # For checkpoint DiT DPO Model
    DiT_DPO_path = save_dir / "DiT_DPO.pt"
    tmp_path= save_dir / "checkpoint.tmp"
    last_path=save_dir/"DiT_DPOLast.pt"
    

    train_losses_path=save_dir / "losses_trainDiT_DPO.npy"
    test_losses_path=save_dir / "losses_testDiT_DPO.npy"
    winnerScorePath=save_dir/ "winnerScoresDPO.npy"
    loserScorePath=save_dir/ "loserScoresDPO.npy"
    trueScorePath=save_dir/ "trueScoresDPO.npy"
    l2RelativeNormPath=save_dir/"l2_relativeNormDPO.npy"
    

    f = Path(DiT_DPO_path)
    print("resolve==============",f.resolve())

    # PHYSICAL Pattern
    physics_pattern=PhysicalPatternsBurgers(cfg.dx) 
    
    #---------------------------------REwards---------------------------------------------
    rewards=PhysicsRewardSignal(cfg.x_dim,cfg.u_dim,cfg.number_query_tokens,cfg.score_query_size,cfg.latent_dim,cfg.start_latent_value_size,
                                    cfg.start_latent_query_size,cfg.num_physics_layers,cfg.num_temp_layers,cfg.length_x_coords,cfg.slice_num,
                                    cross_attn_heads=cfg.reward_attn_heads,cross_attn_dim_head=cfg.reward_attn_dim_head,
                                    slice_attn_heads=cfg.reward_attn_heads,slice_attn_dim_head=cfg.reward_attn_dim_head,
                                    cross_attn_drop=cfg.cross_attn_drop,cross_attn_alibi_heads=cfg.cross_attn_alibi_heads,
                                    slice_alibi_heads=cfg.slice_alibi_heads,mult_latent_dim=cfg.mult_latent_dim,drop=cfg.drop,same_block=cfg.same_block)
         
    rewards.to(device)
    state_rewards=State_DiT(rewards,None,None) #best_valid_loss=np.inf and epoch_init=0

    state_rewards=load_checkpoint_DiT(state_rewards, rewards_path, device,mode="eval") #on recommence depui s l e modele sauvegarde
    print("-------------------telechargemnt rewardsreussi---------------------- epoch=",state_rewards.epoch)
    # Freeze rewards models
    for p in state_rewards.model.parameters():
                p.requires_grad = False

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
    
    ##-------------------------LOAD ENCO_DECO_DPO Model -----------------------------
    enco.to(device)
    deco.to(device)

    state_enco_deco=State(enco,deco,optim_enco=None,optim_deco=None,scheduler_enco=None,scheduler_deco=None)
    state_enco_deco=load_checkpoint_encoDeco(state_enco_deco, enco_deco_path, device,mode="eval")  #on recommence depui s l e modele sauvegarde
    # Freeze encoder and decoder
    for p in state_enco_deco.model_enco.parameters():
        p.requires_grad = False
    for p in state_enco_deco.model_deco.parameters():
            p.requires_grad = False
    print("-------------------telechargemnt model enco-deco reussi avec epoch=----------",state_enco_deco.epoch)

    model_DiT=DiT(
        input_size=cfg.h,
        num_tokens=cfg.M,
        in_channels=4,
        hidden_size=128,  # 192,#1152,
        depth=4,  # 4
        num_heads=4,  # 6
        mlp_ratio=4.0,  # 4.0
        learn_sigma=False,
        )

    model_DiT_DPO = copy.deepcopy(model_DiT)
    model_DiT_frozen = copy.deepcopy(model_DiT)
    model_DiT_DPO.to(device)
    model_DiT_frozen.to(device)

    optimizer_DiT=torch.optim.Adam(model_DiT.parameters() ,lr=cfg.learning_rate  )

    epoch_init=0

    scheduler_DiT = CosineAnnealingLR(optimizer_DiT, T_max=cfg.max_iterations, eta_min=1e-5)
    
    epoch_init=0
    stateDiT_DPO=State_DiT(model_DiT_DPO,optimizer_DiT,scheduler_DiT) #best_valid_loss=np.inf and epoch_init=0
    
    #--------------- Resume-------------------------------------------
    if DiT_DPO_path.exists() and DiT_DPO_path.is_file():
        stateDiT_DPO=load_checkpoint_DiT(stateDiT_DPO ,DiT_DPO_path, device)  #on recommence depuis le modele sauvegarde
        stateDiT_DPO.best_valid_loss = np.inf 
       
        print(f"-------------------telechargemnt model DiT-DPO reussi, epoch={stateDiT_DPO.epoch}----------")
    else: # BEGINING
        stateDiT_DPO=load_checkpoint_DiT(stateDiT_DPO ,DiT_path, device,mode="begining")  #on recommence depuis le modele sauvegarde
        print(f"-------------------telechargemnt model DiT reussi ---------- epoch={stateDiT_DPO.epoch}")
        stateDiT_DPO.epoch = epoch_init
        stateDiT_DPO.best_valid_loss = np.inf

    for p in stateDiT_DPO.model.parameters():
        p.requires_grad = True

    stateDiT=State_DiT(model_DiT_frozen,None,None) #best_valid_loss=np.inf and epoch_init=0
    stateDiT=load_checkpoint_DiT(stateDiT ,DiT_path, device,mode="eval")
    print(f"-------------------telechargemnt model DiT-frozen reussi ---------- epoch={stateDiT.epoch}")
    #Freeze DiT model
    for p in stateDiT.model.parameters():
        p.requires_grad = False

    #- CHECK TWO MODELS 
    print("CHECK", stateDiT.model is stateDiT_DPO.model)
    # _____________LOAD losses-------------------------------
    if  train_losses_path.exists() and train_losses_path.is_file(): #It is enough to consider just one
        
        train_losses=np.load(train_losses_path) #
        test_losses=np.load(test_losses_path)
        winnerScoreList=np.load(winnerScorePath)
        loserScoreList=np.load(loserScorePath)
        trueScoreList=np.load(trueScorePath)
        l2NormList=np.load(l2RelativeNormPath)

        train_losses=train_losses.tolist()
        test_losses=test_losses.tolist()
        winnerScoreList=winnerScoreList.tolist()
        loserScoreList=loserScoreList.tolist()
        trueScoreList=trueScoreList.tolist()
        l2NormList=l2NormList.tolist()

        print("-------------------telechargemnt loss reussi----------------------")

    else:    
        train_losses=[]
        test_losses=[]
        winnerScoreList=[]
        loserScoreList=[]
        trueScoreList=[]
        l2NormList=[]

    #--------------------------------DDp---------------------------------------------------
    
    stateDiT_DPO.model=  DistributedDataParallel(stateDiT_DPO.model, device_ids=[idr_torch.local_rank], find_unused_parameters=True)


    #-----------------------------------------------------------------------------------------
    counter=0
    patience=1000
    num_grad_update=0
    l2_relative_norm=torch.tensor(0,dtype=torch.float32,device=device)

    #------------------------------------------------------------------------------------------- 
    min_noise_std = cfg.min_noise_std  # 2e-6
    betas = [
        min_noise_std ** (k / cfg.num_refinement_steps)     #cfg.num_refinement_steps =3
        for k in reversed(range(cfg.num_refinement_steps + 1))
    ]

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_refinement_steps + 1,
        trained_betas=betas,
        prediction_type="v_prediction",
        clip_sample=False,
    )
    time_multiplier = 1000 / cfg.num_refinement_steps
    # ----------------x_space --------------------
    
    print("Starting of optimizing")
    for step in range(stateDiT_DPO.epoch,cfg.max_iterations): 
        total_train_loss=0 
        stateDiT_DPO.epoch=step
        #Réinitialisation des gradients
        stateDiT_DPO.optim.zero_grad()

        print(f"---------{step}/{cfg.max_iterations}--------------")

        if step<=40:
            pertu=np.random.choice(cfg.pertu_deviation_set, size=cfg.num_sample, replace=False)
        elif step>40 and step <60:
            pertu=np.random.choice(cfg.pertu_deviation_set, size=cfg.num_sample, replace=True)
        elif step >= 60: #100
            pertu=np.random.choice(cfg.pertu_deviation_set2, size=cfg.num_sample, replace=True)

        cfg.DPO_weight=0.1
        for step_train_loader,batch in enumerate(train_loader):
            t0=time.time()
            sampler_train.set_epoch(step)

            state_enco_deco.model_enco.eval()
            state_enco_deco.model_deco.eval()
            stateDiT_DPO.model.train()

            x=batch[0] 
            u=batch[1]  
            x=x.to(device)
            u=u.to(device)
            target=u 
            batch_size_tr,time_step,space_step,u_dim=u.shape 
            
            x=rearrange(x, "b T N d -> (b T) N d")
            u=rearrange(u, "b T N d -> (b T) N d")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): # mixed precision
                with torch.no_grad():
                                        
                    sample,_,_,_=state_enco_deco.model_enco(x,u) 

                    sammple_amenaged=rearrange(sample,"(b T) N d -> b T N d",b=batch_size_tr)
                    
                sample_prev=sammple_amenaged[:,0:-1,:,:]  
                sample_cur=sammple_amenaged[:,1:,:,:] 
                sample_prev=rearrange(sample_prev,"b T N d -> (b T) N d")
                sample_cur=rearrange(sample_cur,"b T N d -> (b T) N d")
                """
                k = torch.randint(
                    0,
                    ddpm_scheduler.config.num_train_timesteps,
                    (sample_prev.shape[0],),
                    device=sample_prev.device,
                ) #shape (batch,)"""
                loss_mse=torch.tensor(0,dtype=sample_prev.dtype, device=sample_prev.device)
                for timestep in ddpm_scheduler.timesteps:
                    k= (
                        torch.zeros(
                            size=(sample_prev.shape[0],), dtype=torch.long, device=sample_prev.device
                        )
                        + timestep
                    )

                    noise_factor = ddpm_scheduler.alphas_cumprod.to(sample_prev.device)[k]
                    noise_factor = noise_factor.view(-1, *[1 for _ in range(sample_prev.ndim - 1)]) 
                    signal_factor = 1 - noise_factor
                    noise = torch.randn_like(sample_cur)

                    sample_noised = ddpm_scheduler.add_noise(sample_cur, noise, k)


                    pred = stateDiT_DPO.model(torch.cat([sample_prev, sample_noised], dim=1), k * time_multiplier) 


                    target_latent = (noise_factor**0.5) * noise - (signal_factor**0.5) * sample_cur 


                    loss_mse += F.mse_loss(pred ,target_latent)  
                loss_mse=loss_mse/( cfg.num_refinement_steps + 1)

            #=============================== START OF DPO ============================================
            x=rearrange(x, "(b T) N d -> b T N d",b=batch_size_tr)
            u=rearrange(u, "(b T) N d -> b T N d",b=batch_size_tr)
            
            with torch.no_grad():
                winnerIndx,loserIndx,pertubation,_=unroll_samples(cfg,x,u,cfg.num_sample,pertu,state_enco_deco.model_enco,
                                                            state_enco_deco.model_deco,stateDiT.model,ddpm_scheduler,
                                                            time_multiplier,device)
            batch_idx = torch.arange(batch_size_tr)
            pertubation=torch.cat( (pertubation[winnerIndx,batch_idx].unsqueeze(0), 
                                    pertubation[loserIndx,batch_idx].unsqueeze(0)),
                                    dim=0 )

            u_init=u[:,0,...]
            u_init=u_init.unsqueeze(0)+ pertubation
            x_in=x[:,0].expand( *pertubation.shape )
            x_in=rearrange(x_in, "E b N d -> (E b) N d")
            u_init=rearrange(u_init, "E b N d -> (E b) N d")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
                with torch.no_grad():
                        
                    sample_prev,_,_,_=state_enco_deco.model_enco(x_in,u_init) 
                
            y=[]

                                    
            for t in range(time_step-1):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
                    y_noised = torch.randn_like(sample_prev) 

                    for k in ddpm_scheduler.timesteps:
                        timess = (
                            torch.zeros(
                                size=(sample_prev.shape[0],), dtype=sample_prev.dtype, device=sample_prev.device
                            )
                            + k
                        )

                        pred = stateDiT_DPO.model(
                            torch.cat([sample_prev, y_noised], dim=1), timess * time_multiplier
                        )
                        y_noised = ddpm_scheduler.step(pred, k, y_noised).prev_sample 
                
                sample_prev=y_noised
                y.append( y_noised.unsqueeze(1)) 

            y=torch.cat(y,dim=1)
            x_out=x.expand(cfg.num_win_los,batch_size_tr,time_step,space_step,1) 
            x_out=rearrange(x_out[:,:,1:,...],"E b T N d -> (E b T) N d ")
            
            y=rearrange(y,"b T N d -> (b T) N d ") 
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
                out=state_enco_deco.model_deco(y,x_out)   

            out=rearrange(out,"(E b T) N d -> E b T N d",E=cfg.num_win_los,b=batch_size_tr) 
            
            out=torch.cat( (target[:,0:1,...].expand(cfg.num_win_los,batch_size_tr,1,space_step,u_dim),out), dim=2 )
    
            #============================== END ENCO-Dit-DECO for DPO ============================
            t1=time.time()

            winner,loser=out[0],out[1]              
            t2=time.time()
            

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                
                winner_score=state_rewards.model(x,winner) #(b )
                loser_score=state_rewards.model(x,loser)
                assert winner_score.shape == loser_score.shape, f"{winner_score.shape} vs {loser_score.shape}"
                winner_score=winner_score.squeeze(-1).squeeze(-1)
                loser_score=loser_score.squeeze(-1).squeeze(-1) 
    
                loss_DPO=-F.logsigmoid(winner_score - loser_score)
                loss_DPO=loss_DPO.mean()
                
                #FINAL LOSS

                loss=(cfg.DiT_mse_weight*loss_mse+cfg.DPO_weight*loss_DPO)/cfg.accumulation_steps
                t3=time.time()
            
           

            #--------------------------------------------------------------------------------

            #backward
            loss.backward()
            t4=time.time()
            total_train_loss+=loss.item()/len(train_loader)
            
            if rank==0:
                print(f"============= step={step:<10d}--  t-E-DiT-D={(t1-t0):.4f}--- loss_mse_kl ={loss_mse.item():.4f} loss_DPO={loss_DPO.item():.4f} --- l2_rela_test={l2_relative_norm.item()}")
                print("             ----------------------------------           ")
                print(f"---- t-WinLos={(t2-t1):.4f}--- t-Rew={(t3-t2):.4f}--- t-back={(t4-t3):.4f}-- step_train_loader={step_train_loader:<10d}-- win_score={winner_score.tolist()}-- loser_score={loser_score.tolist()} ")

            

            
            if ((step_train_loader+1)% (cfg.accumulation_steps)==0) or ((step_train_loader+1)==len(train_loader)):
                num_grad_update+=1
                print(f"Update gradient {num_grad_update} time")
                # -------- Clipping grad
                DiTDPO_grad_norm = clip_grad_norm_(stateDiT_DPO.model.parameters(), max_norm=1.0)
                
                if rank == 0:
                    print(f"DiTDPO_grad_norm={DiTDPO_grad_norm :.4f} ")

                stateDiT_DPO.optim.step()

                #Réinitialisation des gradients
                stateDiT_DPO.optim.zero_grad()

        stateDiT_DPO.scheduler.step()

        #-------checkpoint--------------------------------------------------
        loss_tensor = torch.tensor(total_train_loss, device=device)/ idr_torch.size
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        
        if loss_tensor.item() < stateDiT_DPO.best_valid_loss:
            stateDiT_DPO.best_valid_loss = loss_tensor.item()
            stateDiT_DPO.epoch=step
            counter=0

            if rank==0:
                save_checkpoint_DiT(stateDiT_DPO,tmp_path, DiT_DPO_path)
                print(f"******* step_train_loader={step_train_loader}  total_train_loss_per_loader={total_train_loss}")

        else:
            counter += 1
            # Convert stop decision to a tensor and broadcast from rank 0
            should_stop = torch.tensor(1 if counter >= patience else 0, device=device)
            dist.broadcast(should_stop, src=0)  # rank 0 decision propagates to all

            if should_stop.item():
                if rank == 0:
                    print("Early stopping déclenché !")
                break  

        
        #------------------- Eval --------------
        if step % cfg.log_interval == 0:
            
            if rank==0:
                train_losses.append(total_train_loss)

            with torch.no_grad():

                sampler_test.set_epoch(step)
                batch=next(test_loader_cycle)

                x=batch[0] 
                u=batch[1]  
                x=x.to(device)
                u=u.to(device)
                target=u

                batch_size_test,time_step,space_step,u_dim=u.shape 
                
                state_enco_deco.model_enco.eval()
                state_enco_deco.model_deco.eval()
                stateDiT_DPO.model.eval()
                state_rewards.model.eval()
                
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    
                    
                    winnerIndx,loserIndx,pertubation,_=unroll_samples(cfg,x,u,cfg.num_sample,pertu,state_enco_deco.model_enco,
                                                                state_enco_deco.model_deco,stateDiT.model,ddpm_scheduler,
                                                                time_multiplier,device)
                    batch_idx = torch.arange(batch_size_test)
                    pertubation=torch.cat( (pertubation[winnerIndx,batch_idx].unsqueeze(0), 
                                            pertubation[loserIndx,batch_idx].unsqueeze(0)),
                                            dim=0 )

                    u_init=u[:,0,...]
                    u_init=u_init.unsqueeze(0)+ pertubation
                    x_in=x[:,0].expand( *pertubation.shape )
                    x_in=rearrange(x_in, "E b N d -> (E b) N d")
                    u_init=rearrange(u_init, "E b N d -> (E b) N d")

                    target=u
                    u=rearrange(u, "b T N d -> (b T) N d")
                    x=rearrange(x, "b T N d -> (b T) N d")

                            
                    sample,_,_,_=state_enco_deco.model_enco(x,u) 
                    sample_pertu,_,_,_= state_enco_deco.model_enco(x_in,u_init) 
                    sample=rearrange(sample,"(b T) N d -> b T N d",b=batch_size_test)
                    sample_pertu=rearrange(sample_pertu,"(E b) N d -> E b N d",b=batch_size_test)

                total_test_loss_per_time=0
                y=[]  
                sample_generated=torch.cat((sample[:,0,...].unsqueeze(0),sample_pertu ),dim=0)  
                sample_generated=rearrange(sample_generated,"E b N d -> (E b) N d")
                        
                for t in range(time_step-1):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
                        
                        y_noised = torch.randn_like(sample_generated)  

                        for k in ddpm_scheduler.timesteps:
                            timess = (
                                torch.zeros(
                                    size=(sample_generated.shape[0],), dtype=sample_generated.dtype, device=sample_generated.device
                                )
                                + k
                            )

                            pred = stateDiT_DPO.model(
                                torch.cat([sample_generated, y_noised], dim=1), timess * time_multiplier
                            )
                            y_noised = ddpm_scheduler.step(pred, k, y_noised).prev_sample 
                    
                    sample_generated=y_noised
                    y.append( y_noised.unsqueeze(1)) 

                y=torch.cat(y,dim=1)
                y=rearrange(y,"(E b) T M d -> E b T M d",b=batch_size_test) # T= T-1
                loss_test_mse=F.mse_loss(y[0],sample[:,1:])

                x=rearrange(x, "(b T) N d -> b T N d", b=batch_size_test)
                x_out=x.expand(1+cfg.num_win_los,batch_size_test,time_step,space_step,1) 
                x_out=rearrange(x_out[:,:,1:,...],"E b T N d -> (E b T) N d ")
                
                y=rearrange(y,"E b T N d -> (E b T) N d ") 
                out=state_enco_deco.model_deco(y,x_out) 

                out=rearrange(out, "(E b T) N h ->E b T N h",E=1+cfg.num_win_los,b=batch_size_test) 
                
                out=torch.cat( (target[:,0:1,...].expand(1+cfg.num_win_los,batch_size_test,1,space_step,u_dim),out), dim=2 )

                winner,loser=out[1],out[2]          
                

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    winner_score=state_rewards.model(x,winner) #(b)
                    loser_score=state_rewards.model(x,loser)
                    true_score=state_rewards.model(x,out[0])

                lossTestDPO=-F.logsigmoid(winner_score - loser_score)
                lossTestDPO=lossTestDPO.mean()
                lossTest=cfg.DiT_mse_weight*loss_test_mse+cfg.DPO_weight* lossTestDPO
                dist.all_reduce(lossTest, op=dist.ReduceOp.SUM)        
                l2_relative_norm=( (torch.norm((target[0] - out[0,0]),p=2,dim=(-3,-2,-1)) ) /(torch.norm( target[0],p=2,dim=(-3,-2,-1)) ) )*100
                dist.all_reduce(l2_relative_norm, op=dist.ReduceOp.SUM)
                l2NormList.append(l2_relative_norm.item()/idr_torch.size)
                
                test_losses.append(lossTest.item()/idr_torch.size)
                winnerScoreList.append(winner_score.tolist()[0])
                loserScoreList.append(loser_score.tolist()[0])
                trueScoreList.append(true_score.tolist()[0])

            stateDiT_DPO.epoch=step
            if rank==0:
                np.save(l2RelativeNormPath, np.array(l2NormList))
                np.save(test_losses_path,np.array(test_losses))
                np.save(train_losses_path,np.array(train_losses))
                np.save(winnerScorePath,np.array(winnerScoreList))
                np.save(loserScorePath,np.array(loserScoreList))
                np.save(trueScorePath, np.array(trueScoreList))
                
                
                save_checkpoint_DiT(stateDiT_DPO,tmp_path, last_path)

                print("relative_l2_norm=",l2_relative_norm.item()/idr_torch.size)

    print("End optimization")