
def add_args(parser):

    parser.add_argument("--x_dim",type=int,default=1,help="dimension of x")
    parser.add_argument("--u_dim",type=int,default=1,help="dimension of u")
    
    parser.add_argument("--save_dir", type=str, default="checkpoints")  #checkpoint
    
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

    # For DiT
    parser.add_argument("--num_refinement_steps", type=int, default=3)
    parser.add_argument("--min_noise_std",type=float,default=1e-2)

    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=582838)
    
    # FOR rewards
    parser.add_argument("--number_query_tokens",type=int,default=64,help=" the number of reduced tokens")
    parser.add_argument("--latent_dim",type=int,default=128,help="the dim of the latent variable")
    parser.add_argument("--start_latent_query_size",type=int,default=64,help="the size of the query")
    parser.add_argument("--score_query_size",type=int,default=256,help="the size of the score query")
    parser.add_argument("--num_physics_layers",type=int,default=4,help="number of the physics ")
    parser.add_argument("--num_temp_layers",type=int,default=6,help="number of the temporal layers")
    parser.add_argument("--length_x_coords",type=int,default=16,help="Lx")
    parser.add_argument("--slice_num",type=int,default=8,help="numbers of slices: Must be smaller that number_query_tokens")
    parser.add_argument("--reward_attn_heads",type=int,default=8,help="head of attention in reward")
    parser.add_argument("--reward_attn_dim_head",type=int,default=16,help="reward_attn_dim_head")
    parser.add_argument("--cross_attn_drop",type=float,default=0,help="drop in cross attention reward model")

    parser.add_argument("--cross_attn_alibi_heads",type=int,default=0,help="cross_attn_alibi_heads")
    parser.add_argument("--slice_alibi_heads",type=int,default=0,help="slice_alibi_heads")
    parser.add_argument("--mult_latent_dim",type=int,default=2,help="MLP ratio in reward model")
    parser.add_argument("--drop",type=int,default=0,help="dropout at the beginig")
    parser.add_argument("--same_block",type=bool,default=False,help="whether the physics layer share the same block/weigh")
    

    #FOR METRICS 
    parser.add_argument("--mass_weight", type=float, default=1.,help="the pressure weight in metric")
    parser.add_argument("--energy_weight", type=float, default=0.1,help="the energy weight of buoyancy in metric")
    parser.add_argument("--grad_weight", type=float, default=10.,help="thegrad weight in metric")
    parser.add_argument("--boundary_weight", type=float, default=0.1,help="the boundary weight in metric")
    parser.add_argument("--dx", type=float, default=16./100.)
    

    parser.add_argument("--eps", type=float, default=1e-32,help="epsilon in the log of the metric")
    parser.add_argument("--pertu_deviation_set", type=list, default=[0.01,0.02,0.03,0.04,0.05,0.07,0.08,0.1,0.015,0.025,0.035,0.045,0.055,0.001],help="the set standard deviation of the pertubation ")
    parser.add_argument("--pertu_deviation_set2", type=list, default=[0.01,0.02,0.03,0.01,0.02,0.03,0.01,0.02,0.03,0.01,0.02,0.03,0.01,0.02,0.03,0.001],help="the set standard deviation of the pertubation 2")


def add_args_encoDeco(parser):
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--accumulation_steps", type=int, default=32,help="number of step before to update the model:global_bath=accumulation_steps*mini_batch_size")
    parser.add_argument("--mini_batch_size", type=int, default=5,help="Taille mini-batch pour éviter dépassement mémoire")
    parser.add_argument("--test_batch_size", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=4)

    parser.add_argument("--t_train_max", type=int, default=50,help="the time horizon for training/testing NOTE THAT THE MAXIMUM=250 ")
    


def add_args_DiT(parser):

    parser.add_argument("--test_batch_size", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=2000) 
    parser.add_argument("--log_interval", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--mini_batch_size", type=int, default=5,help="Taille mini-batch pour éviter dépassement mémoire")
    parser.add_argument("--accumulation_steps", type=int, default=32,help="number of step before to update the model:global_bath=accumulation_steps*mini_batch_size")
    
    parser.add_argument("--t_train_max", type=int, default=50,help="the time horizon for training/testing NOTE THAT THE MAXIMUM=250 ")

    
def add_args_rewards(parser):
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=1000) 
    parser.add_argument("--log_interval", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--mini_batch_size", type=int, default=2,help="Taille mini-batch pour éviter dépassement mémoire")
    parser.add_argument("--accumulation_steps", type=int, default=8,help="number of step before to update the model:global_bath=accumulation_steps*mini_batch_size")

    parser.add_argument("--t_train_max", type=int, default=50,help="the time horizon for training/testing NOTE THAT THE MAXIMUM=250 ")
    parser.add_argument("--num_sample", type=int, default=8,help="number of samples of trajectories") 


def add_args_DiTDPO(parser):
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--accumulation_steps", type=int, default=8,help="number of step before to update the model:global_bath=accumulation_steps*mini_batch_size")
    parser.add_argument("--mini_batch_size", type=int, default=2,help="Taille mini-batch pour éviter dépassement mémoire")
    
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=2)

    parser.add_argument("--t_train_max", type=int, default=50,help="the time horizon for training/testing NOTE THAT THE MAXIMUM=250 ")
    parser.add_argument("--num_sample", type=int, default=8,help="number of samples of trajectories")
    parser.add_argument("--DPO_weight", type=float,default=1.,help="the weight in the DPO loss") #or 0.5