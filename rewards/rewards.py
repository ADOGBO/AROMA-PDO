import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math

from timm.models.vision_transformer import Attention, Mlp
from encoder_decoder.enco_deco import cache_fn,FourierFeaturesBase2, FeedForward, Prenorm, PreNormCross,MultiHeadAttention


class GeometryAwareAlibiPositionalBias(nn.Module):
    def __init__(self, heads, total_heads=None, learnable=False):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads if total_heads is not None else heads
        self.slopes = nn.Parameter(torch.arange(1, heads + 1).float().unsqueeze(-1).unsqueeze(-1), requires_grad=learnable)

    def forward(self, qcoords, kvcoords):
        device = self.slopes.device  # Get the actual device dynamically
        b = kvcoords.shape[0]
        x = kvcoords.shape[1]

        bias = torch.cdist(qcoords, kvcoords, p=2).clamp(min=1e-6) # B LQ LKV
        bias = bias.unsqueeze(1).repeat(1, self.heads, 1, 1) * self.slopes # B H LQ LKV
        
        bias = torch.cat((bias, torch.zeros(b, self.total_heads - self.heads, bias.shape[-2], bias.shape[-1], device=device)), dim=1)
        
        return bias

def build_block_causal_mask(L, M, device):
    """
    Builds a block-wise causal mask for L frames of M tokens each.
    Total sequence length T = L * M.

    Rule:
      - token in frame i can attend to ALL tokens in frame j <= i
      - i.e. full attention within a frame, causal across frames

    Returns a boolean mask of shape (T, T) where True = "allowed to attend".
    """
    T = L * M
    # frame_idx[i] = which frame token i belongs to
    frame_idx = torch.arange(T, device=device) // M   # (T,)

    # mask[i, j] = True if frame_idx[j] <= frame_idx[i]
    mask = frame_idx.unsqueeze(1) >= frame_idx.unsqueeze(0)  # (T, T)
    return mask  # True where attention is allowed

class CrossAttention(nn.Module):
    """
    Standard scaled dot product attention
    
    in_dim: input dimension
    out_dim: output dimension
    heads: number of head representations
    dim_head: number of dimension for each head
    dropout: dropout ratio
    """
    def __init__(
        self, query_dim, key_dim, value_dim, out_dim, heads=8, dim_head=64, dropout=0, alibi_heads=0, qk_norm = True):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, out_dim)

        self.q_norm = nn.LayerNorm(dim_head, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(dim_head, eps=1e-6) if qk_norm else nn.Identity()

        self.resid_drop = nn.Dropout(dropout)
        self.alibi_heads = alibi_heads
        if self.alibi_heads:
            self.rel_pos_bias = GeometryAwareAlibiPositionalBias(alibi_heads, heads)
        
        self.to_out = nn.Linear(inner_dim, out_dim)

        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, key=None, value=None, q_coords=None, kv_coords=None, mask=None):
        h = self.heads
        b, n, d = x.shape

        q = self.to_q(x)
        k = self.to_k(key)
        v = self.to_v(value)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = self.q_norm(q).to(q.dtype)
        k = self.k_norm(k).to(q.dtype)
        
        pos_bias = torch.zeros((b, h, q.shape[-2], k.shape[-2]), device=q.device)
        
        if self.alibi_heads:
            pos_bias = self.rel_pos_bias(q_coords, kv_coords)

        scores = F.scaled_dot_product_attention(q, k, v, attn_mask=pos_bias ,dropout_p=self.attn_drop)

        out = rearrange(scores, "b h n d -> b n (h d)", h=h)
        return self.resid_drop(self.to_out(out))

    
class BlockCausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads,heads_dim, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = heads_dim

        self.qkv_proj = nn.Linear(d_model, 3*n_heads*heads_dim)
        self.out_proj = nn.Linear(n_heads*heads_dim, d_model)
        self.dropout = dropout

    def forward(self, x, attn_mask):
        """
        x: (B, T, d_model), T = L*M
        attn_mask: (T, T) boolean, True = allowed
        """
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.split(C, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # scaled_dot_product_attention expects additive mask or bool mask
        # bool mask: True = keep, False = mask out (set to -inf)
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,     # (T, T) broadcast over B, heads
            dropout_p=self.dropout,
            is_causal=False          # we supply our own mask, not token-causal
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)
    
class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads,heads_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BlockCausalSelfAttention(d_model, n_heads,heads_dim, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class SpatioTemporalCausalTransformer(nn.Module):
    """
    Input : Z_{0:L-1} of shape (B, L, M, d)  -- past trajectory
    Output: Z^L        of shape (B, M, d)    -- predicted next latent frame
    """
    def __init__(self, d_model, n_heads=8,heads_dim=16, n_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model


        # temporal positional embedding: one per FRAME (not per token)
        self.spatial_pos_emb = SinusoidalPositionEmbeddings(d_model)

        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads,heads_dim, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final_time = nn.LayerNorm(d_model)

        # Head to project the last frame's tokens -> next frame prediction
        self.pred_head_time = nn.Linear(d_model, d_model)

    def forward(self, Z):
        """
        Z: (B, L, M, d)  -- encoded latent trajectory Z^{0:L-1}
        Returns: Z_next (B, M, d) -- predicted Z^L
        """
        B, L, M, d = Z.shape
        device = Z.device

        # 1. Add spatial positional encoding 
        spatial_pos = torch.arange(M, device=device)              # (M,)
        pos_emb = self.spatial_pos_emb(spatial_pos)               # (M, d)
        Z = Z + pos_emb.unsqueeze(0).unsqueeze(0)                              # (B, L, M, d) + (1,1,M,d)

        # 2. Flatten frames+tokens into one sequence: (B, L*M, d)
        x = Z.view(B, L * M, d)

        # 3. Build block-causal mask
        attn_mask = build_block_causal_mask(L, M, device)         # (T, T)

        # 4. Run through transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.ln_final_time(x)

        # 5. Reshape back to (B, L, M, d) and take the LAST frame's tokens
        #    (those tokens have "seen" the entire past sequence via the mask)
        x = x.view(B, L, M, d)
        last_frame_repr = x[:, -1, :, :]                          # (B, M, d)

        # 6. Project to predicted next latent frame Z^L
        Z_state = self.pred_head_time(last_frame_repr)                  # (B, M, d)
        return Z_state                  # (B, M, d)

class RectangleAlibiPositionalBias(nn.Module):
    def __init__(self, heads, total_heads=None, learnable=False):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads if total_heads is not None else heads
        self.slopes = nn.Parameter(torch.arange(1, heads + 1).float().unsqueeze(-1).unsqueeze(-1), requires_grad=learnable)

    def forward(self, n, m):
        device = self.slopes.device  # Get the actual device dynamically

        iss = - torch.arange(n, device=device).unsqueeze(1)
        jss = - torch.arange(m, device=device).unsqueeze(0)

        bias = (iss / n - jss / m) * (iss / n > jss / m) + (-iss / n + jss / m) * (iss / n < jss / m)

        bias = bias.unsqueeze(0).repeat(self.heads, 1, 1) * self.slopes
        bias = torch.cat((bias, torch.zeros(self.total_heads - self.heads, n, m, device=device)), dim=0)
        
        return bias

class Physics_Attention_Structured_Mesh_1D(nn.Module):
    def __init__(self, dim_coords,dim_im, heads=8, dim_head=64, dropout=0., slice_num=64, kernel=3, alibi_heads=0,descaling=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.slice_num = slice_num
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.dropout_p = dropout
        self.alibi_heads = alibi_heads
        self.descaling=descaling

        self.in_project_coords = nn.Conv1d(dim_coords, inner_dim, kernel_size=kernel, padding=kernel // 2)
        self.in_project_image = nn.Conv1d(dim_im, inner_dim, kernel_size=kernel, padding=kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_qkv = nn.Linear(dim_head, 3 * dim_head, bias=False)

        if self.alibi_heads:
            self.rel_pos_bias = RectangleAlibiPositionalBias(self.alibi_heads, heads)
        
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_im),
            nn.Dropout(dropout)
        )

    def forward(self, coords,image):
        """
        coords: B, N, C
        image: B, N, C
        """
        #B, H, W, C = image.shape
        B, N, C = image.shape
        Hn = self.heads
        D = self.dim_head
        M = self.slice_num

        image = image.permute(0, 2, 1)  # B  C N
        coords=coords.permute(0,2,1)

        image_mid = self.in_project_image(image)  # B Hn*D N
        coords_mid = self.in_project_coords(coords)

        image_mid = image_mid.permute(0, 2, 1).reshape(B, N, Hn, D).transpose(1, 2)  # B Hn N D
        coords_mid = coords_mid.permute(0, 2, 1).reshape(B, N, Hn, D).transpose(1, 2)    # B Hn N D

        slice_logits = self.in_project_slice(coords_mid) / torch.clamp(self.temperature, 0.1, 5.0)  # B Hn N M
        slice_weights = F.softmax(slice_logits, dim=-1)

        slice_norm = slice_weights.sum(dim=2, keepdim=True) + 1e-5 # B Hn 1 M
        slice_token = image_mid.transpose(-1, -2) @ slice_weights  # B Hn D M
        slice_token = (slice_token / slice_norm).transpose(-1, -2)  # B Hn M D

        qkv = self.to_qkv(slice_token)  # B Hn M 3*D
        q, k, v = qkv.chunk(3, dim=-1)

        pos_bias = self.rel_pos_bias(q.shape[-2], k.shape[-2]) if self.alibi_heads else None

        out_slice = F.scaled_dot_product_attention(q, k, v, attn_mask=pos_bias, dropout_p=self.dropout_p) # B Hn M D

        if self.descaling:
            out_x = out_slice.transpose(-1, -2) @ slice_weights.transpose(-1, -2)  # B Hn D N
            out_x = out_x.transpose(1, 2).contiguous().reshape(B, N, Hn * D)
            return self.to_out(out_x).reshape(B, N, -1)
        
        else:
            out_x=out_slice.transpose(-1, -2)                           # B Hn D M   
            out_x = out_x.transpose(1, 2).contiguous().reshape(B, M, Hn * D)  
            return self.to_out(out_x)               # B, M, out_dim

class PhysicsRewardSignal(nn.Module):
    def __init__(self,x_dim,u_dim,number_query_tokens,score_query_size,latent_dim,start_latent_value_size,start_latent_query_size,
            num_physics_layers,num_temp_layers,length_x_coords,slice_num,cross_attn_heads=8,cross_attn_dim_head=8, slice_attn_heads=8,
            slice_attn_dim_head=8,cross_attn_drop=0,cross_attn_alibi_heads=0,slice_alibi_heads=0,mult_latent_dim=2,drop=0.1,same_block=True):
        super().__init__()
        self.x_dim=x_dim
        self.X=length_x_coords
        self.number_query_tokens=number_query_tokens
        self.score_query_size=score_query_size
        self.start_latent_query_size=start_latent_query_size
        self.num_physics_layers=num_physics_layers
        #self.start_latent_grid_size=start_latent_grid_size


        self.fourier_feature=FourierFeaturesBase2(log_scale_min=0,log_scale_max=0,k=16)

        x=torch.zeros(x_dim).view(1,-1)
        y= self.fourier_feature(x)
        try:
            shapes=y.shape
        except AttributeError:
            shapes=y[0].shape

        start_latent_key_size=shapes[-1]
        self.linear_cross_for_query1=nn.Linear(self.start_latent_query_size,latent_dim)
        self.linear_cross_for_query2=nn.Linear(self.score_query_size,latent_dim)
        self.embedding=nn.Linear(u_dim,start_latent_value_size)
        self.dropout=nn.Dropout(drop)

        

        # Regular latent grid and coordinates
        self.reg_query = nn.Parameter(torch.randn(self.number_query_tokens, self.start_latent_query_size), requires_grad=True)
        self.reg_coords = nn.Parameter(self.make_grid1d(self.number_query_tokens,self.X), requires_grad=True)
        self.score_query = nn.Parameter(torch.randn(1, self.score_query_size), requires_grad=True)

        
        # Encoder cross-attention
        self.encoder_cross_attend = nn.ModuleList([
            PreNormCross(
                self.start_latent_query_size,
                CrossAttention(
                    query_dim=self.start_latent_query_size,
                    key_dim=start_latent_key_size,
                    value_dim=start_latent_value_size,
                    out_dim=latent_dim,
                    heads=cross_attn_heads,
                    dim_head=cross_attn_dim_head,
                    alibi_heads=cross_attn_alibi_heads,
                    qk_norm=False
                ),
                k_dim=start_latent_key_size,
                v_dim=start_latent_value_size
            ),
            Prenorm(FeedForward(latent_dim,mult_latent_dim),latent_dim)
        ])
        
        # Physics layers
        self.physics_layers=nn.ModuleList([])
        if same_block:
            get_self_physics_block=cache_fn(lambda: Physics_Attention_Structured_Mesh_1D(
                        self.start_latent_query_size,
                        latent_dim,
                        heads=slice_attn_heads,
                        dim_head=slice_attn_dim_head,
                        slice_num=slice_num,
                        alibi_heads=slice_alibi_heads,
                    ))

            get_self_physics_ff=cache_fn(lambda:Prenorm(FeedForward(latent_dim,mult_latent_dim),latent_dim))

            
            for _ in range(num_physics_layers):
                self.physics_layers.append(nn.ModuleList([
                get_self_physics_block(),
                get_self_physics_ff()
            ]))
                
        else: # The layers do'nt share the same weight
            print("--------------- same block (Je suis dans rewards)-----------",same_block)
            for _ in range(num_physics_layers):
                self.physics_layers.append(nn.ModuleList([
                    Physics_Attention_Structured_Mesh_1D(
                        self.start_latent_query_size,
                        latent_dim,
                        heads=slice_attn_heads,
                        dim_head=slice_attn_dim_head,
                        slice_num=slice_num,
                        alibi_heads=slice_alibi_heads
                    ),
                Prenorm(FeedForward(latent_dim,mult_latent_dim),latent_dim)    
                
            ]))
            
        self.compress_time=SpatioTemporalCausalTransformer(latent_dim, 
                                                           n_heads=slice_attn_heads,
                                                           heads_dim= slice_attn_dim_head,
                                                           n_layers=num_temp_layers, 
                                                            dropout=drop)
        
        # Final attention 
        self.final_attn = nn.ModuleList([
            PreNormCross(
                self.score_query_size,
                CrossAttention(
                    query_dim=self.score_query_size,
                    key_dim=self.start_latent_query_size,
                    value_dim=latent_dim,
                    out_dim=latent_dim,
                    heads=cross_attn_heads,
                    dim_head=cross_attn_dim_head,
                    alibi_heads=0.,
                    qk_norm=False 
                ),
                k_dim=self.start_latent_query_size,
                v_dim=latent_dim
            ),
            Prenorm(FeedForward(latent_dim,mult_latent_dim),latent_dim)
        ])
        self.final_ff=nn.Linear(latent_dim,1)


    def forward(self,coords,image):
        """
        coords:  B,T, Dx, Dz, x_dim
        image: B,T, Dx, Dz, u_dim
        """
        B,T,N,d=image.shape
        coords= rearrange(coords, "B T N d -> (B T) N d")
        image=rearrange(image, "B T N d -> (B T) N d")

        #query
        query=self.reg_query.expand(B*T,self.number_query_tokens,self.start_latent_query_size)

        #Embedding
        coords_embed= self.fourier_feature(coords)[0] #(BT, N d_q)
        #print(coords_embed.shape)
        #coords_embed=coords_embed.expand(B*T,*coords_embed.shape)

        kv_coords=self.reg_coords.expand(B*T,self.number_query_tokens,self.x_dim)
        #image=self.embedding(self.dropout(image))
        image=self.embedding(image)
        #print("before cross",image.mean())
        # Cross-attn
        attn,ff =self.encoder_cross_attend
        image=attn(query,key=coords_embed,value=image,q_coords=coords,kv_coords=kv_coords)
        image=image+self.linear_cross_for_query1(query)
        image=image+ff(image)          
        
        query=self.reg_query.expand(B*T,self.number_query_tokens,self.start_latent_query_size)  # (BT num_token latent_dim )
        
        #print("before physics layers",image.mean())
        # Physics layers
        for i,layer in enumerate(self.physics_layers):
            
            physics_layer,physics_ff=layer
            image= physics_layer(query,image)+image
            image=physics_ff(image)+image                      # (BT num_token latent_dim )
        
        image=rearrange(image,"(B T) M d -> B T M d", T=T)
        #print("before compress",image.mean())
        # Temporal layer
        image=self.compress_time(image)         # (B M d)

        # Final Cross-attn
        query=self.reg_query.expand(B,self.number_query_tokens,self.start_latent_query_size)  # (BT num_token latent_dim )
        score_query=self.score_query.expand(B,1,self.score_query_size)  # (BT num_token latent_dim )
        #print("before final attn",image.mean())
        attn,ff =self.final_attn
        image=attn(score_query,key=query,value=image,q_coords=None,kv_coords=None)
        #print("after cross attn ",image.mean())
        image=image+self.linear_cross_for_query2(score_query)
        #print("after image+ query ff ",image.mean())
        image=image+ff(image) 
        #print(" after cros-ff attn",image.mean())
        #final layers
        image=self.final_ff(image)
        #print(" after final  ff",image.mean())


        return image # B 1 1


    def make_grid1d(self, Nx,X):
        x= torch.linspace(0, X, Nx, device=self.reg_query .device)
            
        return x.unsqueeze(-1)
    
class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "Positional embeddings should be multiples of 2"

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim) 
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TransformersBlock(nn.Module):
    """
    A transformer block 
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        

    def forward(self, x):
        
        x = x +  self.attn( self.norm1(x) )
        x = x +  self.mlp( self.norm2(x) )
        return x

class AttentionBlock(nn.Module):
    def __init__(self,hidden_size,num_head,mlp_ratio=4.0):
        super().__init__()
        
        self.attention=PreNormCross( hidden_size, MultiHeadAttention(hidden_size,num_heads=num_head,att_dropout=0),k_dim=hidden_size,v_dim=hidden_size )
        self.ff_after_cross_att= Prenorm(
                FeedForward(hidden_size,int(mlp_ratio),use_gelu=False),
                hidden_size
            )

    def forward(self,q,k,v):
        """
        q: (batch,1,hidden_size)
        k:(batch,seq_len,hidden_size)
        v:(batch,seq_len,hidden_size)
        """
        context_,_=self.attention( q,key=k,value=v) #shape (1,seq_le,hidden_size)
        #assert not torch.isnan(context_x).any()," Nan dectected in context_x"

        q=q+context_
        #assert not torch.isnan(T).any(), "Nan detected in T=T+context_x"

        return q+self.ff_after_cross_att(q) #shape (batch,M,d)
        #assert not torch.isnan(T_geo).any(), "Nan detected in T_geo"


class AttentionRewardSignal(nn.Module):
    def __init__(self,depth,input_size,num_heads,hidden_size,x_dim=4, output_size=1,num_point=None,reduction="last",x_space="regular"):
        super().__init__()
        self.depth=depth
        self.reduction=reduction
        self.x_space=x_space

        if x_space== "variable":
            self.space_embedding=FourierFeaturesBase2(log_scale_min=0,log_scale_max=0,k=16,use_pi=True,log_sampling=True,include_input=True)
            x=torch.zeros(x_dim).view(1,-1)
            y= self.space_embedding(x)
            try:
                shapes=y.shape
            except AttributeError:
                shapes=y[0].shape
            dim_x_embed=shapes[-1]

            self.linear_x_after_ff=nn.Linear(dim_x_embed,hidden_size)
        
        elif x_space=="regular":
            pos_embeder=SinusoidalPositionEmbeddings(hidden_size)
            assert num_point,"you have to give num_point"
            timess=torch.arange(num_point)
            space_embedding=pos_embeder(timess) #(num_point,hidden_size)
            
            self.register_buffer("space_embedding", space_embedding)

        #self.reduction_query=nn.Parameter(torch.randn(reduction_dim,hidden_size))
        #self.reduction_layer= AttentionBlock(hidden_size,num_heads,mlp_ratio=4.0)

        self.u_embeder=nn.Linear(input_size,hidden_size,bias=True)

        self.layers=nn.ModuleList([])
        for _ in range(depth-1):
            self.layers.append(TransformersBlock(hidden_size, num_heads))
        
        if reduction=="last" or reduction=="mean":
            self.penultimate_layer=TransformersBlock(hidden_size, num_heads)
        
        elif reduction=="attention":
            small_std = False
            sigma = 0.02 if small_std else 1
            self.metric_query = sigma*nn.Parameter(torch.randn(1,hidden_size))
            
            self.penultimate_layer=AttentionBlock(hidden_size,num_heads,mlp_ratio=4.0)

        self.final_layer = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self,X,x=None):
        """ 
        shape of X: (batch,T,Nx,Nz,input_size)
        shape of x: (Nx,Nz,dim)
        
        """
        batch_size=X.shape[0]
        X=rearrange(X,'b T N Z c->(b T) (N Z) c')

        #----------Embedding--------------------------------
        if self.x_space== "variable" and x is not None:
            x=rearrange(x,'N Z c->(N Z) c')
            space_embe=self.linear_x_after_ff( self.space_embedding(x)[0] ) #shape (batch,N,d) # self.ff(x) is always a list
            space_embe=space_embe.unsqueeze(0) 

        elif self.x_space== "regular":
            space_embe=self.space_embedding.to( X.device )
            space_embe=space_embe.unsqueeze(0)
        
        X= self.u_embeder(X)         #((batch T),N,hidden_size)

        for layer in self.layers:
            X+space_embe
            X=layer(X)

        if self.reduction=="mean":
            X=self.penultimate_layer(X)
            X= torch.mean(X,dim=1) #((batch T),hidden_size)

        elif self.reduction=="last":
            X=self.penultimate_layer(X)
            X= X[:,-1,:] #((batch T),hidden_size)
        
        else: #self.reduction==attention
            X=self.penultimate_layer(self.metric_query,k=X,v=X) #((batch T),1,hidden_size)
            
            X=X.squeeze(1) #((batch T),hidden_size)



        X=self.final_layer(X) #((batch T), output_size=1)
        
        X=X.squeeze(-1) #(batch T)
        X=rearrange(X,"(b T)->b T",b=batch_size)
        return X

