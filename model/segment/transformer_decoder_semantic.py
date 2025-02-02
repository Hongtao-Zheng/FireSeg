
from functools import partial
import math
from typing import Iterable
from black import diff
from torch import nn, einsum
import numpy as np
import torch as th
import torch.nn as nn
import functools
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List, Callable, Dict
import math
from einops import rearrange, repeat
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
import copy
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from .position_encoding import PositionEmbeddingSine
import fvcore.nn.weight_init as weight_init





class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

def resize_fn(img, size):
    return transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img))
import math
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos = None, query_pos = None):
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output

    
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos = None, query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
#         print(tgt2.shape)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())

# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs

# Spectral normalization base class 
class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]

# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)

# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)

class SegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, con_channels,
                which_conv=nn.Conv2d, which_linear=None, activation=None, 
                upsample=None):
        super(SegBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_linear = which_conv, which_linear
        self.activation = activation
        self.upsample = upsample
        
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                            kernel_size=1, padding=0)
       
        self.register_buffer('stored_mean1', torch.zeros(in_channels))
        self.register_buffer('stored_var1',  torch.ones(in_channels)) 
        self.register_buffer('stored_mean2', torch.zeros(out_channels))
        self.register_buffer('stored_var2',  torch.ones(out_channels)) 
        
        self.upsample = upsample

    def forward(self, x, y=None):
        x = F.batch_norm(x, self.stored_mean1, self.stored_var1, None, None,
                          self.training, 0.1, 1e-4)
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = F.batch_norm(h, self.stored_mean2, self.stored_var2, None, None,
                          self.training, 0.1, 1e-4)
        
        h = self.activation(h)
        h = self.conv2(h)
        if self.learnable_sc:       
            x = self.conv_sc(x)
        return h + x

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):

        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs).double()
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:

                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x.double() * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):

    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : False,
                'input_dims' : 2,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def aggregate_attention(attention_store, res: int, from_where: List[str], is_cross: bool, select: int, prompts=None):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])
                out.append(cross_maps)

    out = torch.cat(out, dim=1)
    return out

class SegDecoderOpenWord(nn.Module):
    
    def __init__(self,
        num_queries=100,
        hidden_dim=512,
        num_classes=19,
        mask_dim=512,
        dim_feedforward=2048,
        text_emb_dim=2048,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.query_feat_mlp = nn.Linear(hidden_dim+text_emb_dim, hidden_dim, bias=False)
        self.query_embed_mlp = nn.Linear(hidden_dim+text_emb_dim, hidden_dim, bias=False)
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
        # ============================
        #  Add here: the MLP and parameters required by DFPQ.
        # ============================
        self.dynamic_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        #  allowing each query to also have a learnable bias term.
        self.B = nn.Parameter(torch.Tensor(num_queries, hidden_dim))
        nn.init.normal_(self.B, std=0.01)
        
        # ============================
        #  Also add here: HRCA-related parameters, which are used to control which layers perform
        #  Used to control which layers perform “sparse high-resolution cross-attention.”
        # ============================
        # self.hrca_layers = [11]
        self.hrca_layers = [3,7]     # The 3rd and 7th layers perform HRCA.
        self.topk_ratio = 0.1        # Only the top 10% of pixels are selected.
        self.high_res_index = 2     
        

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        
        for _ in range(self.num_feature_levels):
            self.input_proj.append(nn.Sequential())

        # output FFNs
        self.mask_classification = True
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        
        # define Transformer decoder here
        self.num_heads = 8
        self.num_layers = 10
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        pre_norm = False
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=self.num_heads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=self.num_heads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        mid_feature_channel = 512
        high_feature_channel = 512
        highest_feature_channel = 512
        
        self.mid_feature_conv = nn.Sequential(
            nn.Conv2d(101360, mid_feature_channel, kernel_size=1, bias=False),
        )
        self.mid_feature_mix_conv = SegBlock(
                                in_channels=mid_feature_channel+mid_feature_channel,
                                out_channels=mask_dim,
                                con_channels=256,
                                which_conv=functools.partial(SNConv2d,
                                    kernel_size=3, padding=1,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                which_linear=functools.partial(SNLinear,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                activation=nn.ReLU(inplace=True),
                                upsample=False,
                            )
        
        self.high_feature_conv = nn.Sequential(
            nn.Conv2d(13460, high_feature_channel, kernel_size=1, bias=False),
        )
        self.high_feature_mix_conv = SegBlock(
                                in_channels=mask_dim+high_feature_channel,
                                out_channels=mask_dim,
                                con_channels=256,
                                which_conv=functools.partial(SNConv2d,
                                    kernel_size=3, padding=1,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                which_linear=functools.partial(SNLinear,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                activation=nn.ReLU(inplace=True),
                                upsample=False,
                            )
        
        self.highest_feature_conv = nn.Sequential(
            nn.Conv2d(5120, highest_feature_channel, kernel_size=1, bias=False),
        )
        self.highest_feature_mix_conv = SegBlock(
                                in_channels=mask_dim+highest_feature_channel,
                                out_channels=mask_dim,
                                con_channels=256,
                                which_conv=functools.partial(SNConv2d,
                                    kernel_size=3, padding=1,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                which_linear=functools.partial(SNLinear,
                                    num_svs=1, num_itrs=1,
                                    eps=1e-04),
                                activation=nn.ReLU(inplace=True),
                                upsample=False,
                            )

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask
    
        
    def forward(self, diffusion_features, controller, prompts, tokenizer, text_embeddings):

        x, mask_features = self._prepare_features(diffusion_features, controller, prompts, tokenizer)
        b = mask_features.size()[0]
        
        src = []
        pos = []
        size_list = []
        

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos_enc = self.pe_layer(x[i], None).flatten(2)        # shape: [B, C, H, W] -> [B, C, HW]
            src_i   = self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]
            # flatten NxCxHxW to HWxNxC
            pos.append(pos_enc.permute(2, 0, 1))  # [HW, B, C]
            src.append(src_i.permute(2, 0, 1))    # [HW, B, C]
        
        query_embed = self.query_embed.weight               # [num_queries, hidden_dim]
        output      = self.query_feat.weight                # [num_queries, hidden_dim]
        

        text_embeddings = text_embeddings.repeat(self.num_queries, b, 1)
        text_embeddings = rearrange(text_embeddings, 'b n d -> (b n) d  ')
        
        query_embed = torch.cat([query_embed, text_embeddings], dim=1)  # [num_queries, 512+2048=2560]
        output      = torch.cat([output,     text_embeddings], dim=1)   # [num_queries, 2560]
        
        query_embed = self.query_feat_mlp(query_embed).unsqueeze(1).repeat(1, b, 1)  # [Q, B, 512]
        output      = self.query_embed_mlp(output).unsqueeze(1).repeat(1, b, 1)      # [Q, B, 512]
        
        _, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            
            attn_mask[attn_mask.sum(-1) == attn_mask.shape[-1]] = False

            if i in self.hrca_layers:
                highest_size = x[self.high_res_index].shape[-2:]
                up_mask = F.interpolate(outputs_mask, size=highest_size, mode='bilinear', align_corners=False)
                score_map = up_mask.sum(dim=1).flatten(1)  # [B, H*W]
                # 3) top-k
                k = int(self.topk_ratio * score_map.shape[1])
                topk_vals, topk_idxs = score_map.topk(k, dim=1)  # [B,k]
                
                #    src[2] : [HW2, B, 512], permute -> [B, HW2, 512]
                s2 = src[self.high_res_index].permute(1, 0, 2).contiguous()  # [B, HW2, 512]
                

                idxs_expanded = topk_idxs.unsqueeze(-1).expand(-1, -1, s2.size(2))  # [B, k, 512]
                # [B, k, 512]
                s2_selected = torch.gather(s2, 1, idxs_expanded)
                
                # [k, B, 512]
                s2_selected = s2_selected.permute(1, 0, 2)

                
                output = self.transformer_cross_attention_layers[i](
                    output, 
                    s2_selected, 
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None,
                    query_pos=query_embed
                )
            else:
                
                output = self.transformer_cross_attention_layers[i](
                    output, 
                    src[level_index], 
                    memory_mask=attn_mask, 
                    memory_key_padding_mask=None, 
                    pos=pos[level_index], 
                    query_pos=query_embed
                )
            

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](output)


            mask_mean = outputs_mask.mean(dim=(-2, -1))  # [B, Q]
            
            mask_mean = mask_mean.permute(1,0).unsqueeze(-1)  # [Q, B, 1]
            
            
            dfpq_embed = output * mask_mean  # [Q,B,512], 
            
            # flatten -> MLP -> reshape
            dfpq_embed_flat = dfpq_embed.flatten(0,1)      # [Q*B,512]
            dfpq_embed_flat = self.dynamic_mlp(dfpq_embed_flat)
            dfpq_embed = dfpq_embed_flat.view(output.shape)  # [Q,B,512]
            

            dfpq_embed = dfpq_embed + self.B.unsqueeze(1)  
            

            query_embed = 0.5 * query_embed + 0.5 * dfpq_embed  

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, 
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
        
        out = {
            'pred_logits': outputs_class,
            'pred_masks':  outputs_mask
        }
        return out
    
    def _prepare_features(self, features, attention_store, prompts, tokenizer, upsample='bilinear'):
        self.mid_feature_size = 64
        self.high_feature_size = 128
        
        mid_features = [
            F.interpolate(i, size=self.mid_feature_size, mode=upsample, align_corners=False) 
            for i in features["mid"]
        ]
        mid_features = torch.cat(mid_features, dim=1)
        
        high_features = [
            F.interpolate(i, size=self.high_feature_size, mode=upsample, align_corners=False) 
            for i in features["high"]
        ]
        high_features = torch.cat(high_features, dim=1)
        
        highest_features = torch.cat(features["highest"], dim=1)

        ## Attention map
        from_where=("up", "down")
        select = 0
        attention_maps_32s = aggregate_attention(attention_store, 32, ("up", "mid", "down"), True, select, prompts=prompts)
        attention_maps_64 = aggregate_attention(attention_store, 64, from_where, True, select, prompts=prompts)

        attention_maps_32s = rearrange(attention_maps_32s, 'b c h w d-> b (c d) h w')
        attention_maps_64  = rearrange(attention_maps_64,  'b c h w d-> b (c d) h w')
        
        attention_maps_32s = F.interpolate(attention_maps_32s, size=self.mid_feature_size, mode=upsample, align_corners=False)
        attention_maps_64  = F.interpolate(attention_maps_64,  size=self.high_feature_size, mode=upsample, align_corners=False)
        
        features_dict = {
            'mid':  torch.cat([mid_features,   attention_maps_32s], dim=1),
            'high': torch.cat([high_features,  attention_maps_64 ], dim=1),
        }
        
        mid_feat = self.mid_feature_conv(features_dict['mid'])
        mid_feat = torch.cat([mid_feat, mid_feat], dim=1)
        mid_feat = self.mid_feature_mix_conv(mid_feat, y=None)
        mid_feat = F.interpolate(mid_feat, size=self.high_feature_size, mode='bilinear', align_corners=False)
        
        high_feat = self.high_feature_conv(features_dict['high'])
        high_feat = torch.cat([mid_feat, high_feat], dim=1)
        high_feat = self.high_feature_mix_conv(high_feat, y=None)
        
        highest_feat = self.highest_feature_conv(highest_features)
        highest_feat = torch.cat([high_feat, highest_feat], dim=1)
        highest_feat = self.highest_feature_mix_conv(highest_feat, y=None)
        
        mid_feat     = F.interpolate(mid_feat,     size=80,  mode='bilinear', align_corners=False)
        high_feat    = F.interpolate(high_feat,    size=160, mode='bilinear', align_corners=False)
        highest_feat = F.interpolate(highest_feat, size=320, mode='bilinear', align_corners=False)
        
        # x[0], x[1], x[2]
        return [mid_feat, high_feat, highest_feat], high_feat