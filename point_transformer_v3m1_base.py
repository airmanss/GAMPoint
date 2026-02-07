# """
# Point Transformer - V3 Mode1

# Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
# Please cite our work if the code is helpful to you.
# """

# from functools import partial
# from addict import Dict
# import math
# import torch
# import torch.nn as nn
# import spconv.pytorch as spconv
# import torch_scatter
# from timm.models.layers import DropPath
# import copy
# try:
#     import flash_attn
# except ImportError:
#     flash_attn = None

# from pointcept.models.point_prompt_training import PDNorm
# from pointcept.models.builder import MODELS
# from pointcept.models.utils.misc import offset2bincount
# from pointcept.models.utils.structure import Point
# from pointcept.models.modules import PointModule, PointSequential


# class RPE(torch.nn.Module):
#     def __init__(self, patch_size, num_heads):
#         super().__init__()
#         self.patch_size = patch_size
#         self.num_heads = num_heads
#         self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
#         self.rpe_num = 2 * self.pos_bnd + 1
#         self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
#         torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

#     def forward(self, coord):
#         idx = (
#             coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
#             + self.pos_bnd  # relative position to positive index
#             + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
#         )
#         out = self.rpe_table.index_select(0, idx.reshape(-1))
#         out = out.view(idx.shape + (-1,)).sum(3)
#         out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
#         return out


# class SerializedAttention(PointModule):
#     def __init__(
#         self,
#         channels,
#         num_heads,
#         patch_size,
#         qkv_bias=True,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         order_index=0,
#         enable_rpe=False,
#         enable_flash=True,
#         upcast_attention=True,
#         upcast_softmax=True,
#     ):
#         super().__init__()
#         assert channels % num_heads == 0
#         self.channels = channels
#         self.num_heads = num_heads
#         self.scale = qk_scale or (channels // num_heads) ** -0.5
#         self.order_index = order_index
#         self.upcast_attention = upcast_attention
#         self.upcast_softmax = upcast_softmax
#         self.enable_rpe = enable_rpe
#         self.enable_flash = enable_flash
#         if enable_flash:
#             assert (
#                 enable_rpe is False
#             ), "Set enable_rpe to False when enable Flash Attention"
#             assert (
#                 upcast_attention is False
#             ), "Set upcast_attention to False when enable Flash Attention"
#             assert (
#                 upcast_softmax is False
#             ), "Set upcast_softmax to False when enable Flash Attention"
#             assert flash_attn is not None, "Make sure flash_attn is installed."
#             self.patch_size = patch_size
#             self.attn_drop = attn_drop
#         else:
#             # when disable flash attention, we still don't want to use mask
#             # consequently, patch size will auto set to the
#             # min number of patch_size_max and number of points
#             self.patch_size_max = patch_size
#             self.patch_size = 0
#             self.attn_drop = torch.nn.Dropout(attn_drop)

#         self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
#         self.proj = torch.nn.Linear(channels, channels)
#         self.proj_drop = torch.nn.Dropout(proj_drop)
#         self.softmax = torch.nn.Softmax(dim=-1)
#         self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

#     @torch.no_grad()
#     def get_rel_pos(self, point, order):
#         K = self.patch_size
#         rel_pos_key = f"rel_pos_{self.order_index}"
#         if rel_pos_key not in point.keys():
#             grid_coord = point.grid_coord[order]
#             grid_coord = grid_coord.reshape(-1, K, 3)
#             point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
#         return point[rel_pos_key]

#     @torch.no_grad()
#     def get_padding_and_inverse(self, point):
#         pad_key = "pad"
#         unpad_key = "unpad"
#         cu_seqlens_key = "cu_seqlens_key"
#         if (
#             pad_key not in point.keys()
#             or unpad_key not in point.keys()
#             or cu_seqlens_key not in point.keys()
#         ):
#             offset = point.offset
#             bincount = offset2bincount(offset)
#             bincount_pad = (
#                 torch.div(
#                     bincount + self.patch_size - 1,
#                     self.patch_size,
#                     rounding_mode="trunc",
#                 )
#                 * self.patch_size
#             )
#             # only pad point when num of points larger than patch_size
#             mask_pad = bincount > self.patch_size
#             bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
#             _offset = nn.functional.pad(offset, (1, 0))
#             _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
#             pad = torch.arange(_offset_pad[-1], device=offset.device)
#             unpad = torch.arange(_offset[-1], device=offset.device)
#             cu_seqlens = []
#             for i in range(len(offset)):
#                 unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
#                 if bincount[i] != bincount_pad[i]:
#                     pad[
#                         _offset_pad[i + 1]
#                         - self.patch_size
#                         + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
#                     ] = pad[
#                         _offset_pad[i + 1]
#                         - 2 * self.patch_size
#                         + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
#                         - self.patch_size
#                     ]
#                 pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
#                 cu_seqlens.append(
#                     torch.arange(
#                         _offset_pad[i],
#                         _offset_pad[i + 1],
#                         step=self.patch_size,
#                         dtype=torch.int32,
#                         device=offset.device,
#                     )
#                 )
#             point[pad_key] = pad
#             point[unpad_key] = unpad
#             point[cu_seqlens_key] = nn.functional.pad(
#                 torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
#             )
#         return point[pad_key], point[unpad_key], point[cu_seqlens_key]

#     def forward(self, point):
#         if not self.enable_flash:
#             self.patch_size = min(
#                 offset2bincount(point.offset).min().tolist(), self.patch_size_max
#             )

#         H = self.num_heads
#         K = self.patch_size
#         C = self.channels

#         pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

#         order = point.serialized_order[self.order_index][pad]
#         inverse = unpad[point.serialized_inverse[self.order_index]]

#         # padding and reshape feat and batch for serialized point patch
#         qkv = self.qkv(point.feat)[order]

#         if not self.enable_flash:
#             # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
#             q, k, v = (
#                 qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
#             )
#             # attn
#             if self.upcast_attention:
#                 q = q.float()
#                 k = k.float()
#             attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
#             if self.enable_rpe:
#                 attn = attn + self.rpe(self.get_rel_pos(point, order))
#             if self.upcast_softmax:
#                 attn = attn.float()
#             attn = self.softmax(attn)
#             attn = self.attn_drop(attn).to(qkv.dtype)
#             feat = (attn @ v).transpose(1, 2).reshape(-1, C)
#         else:
#             feat = flash_attn.flash_attn_varlen_qkvpacked_func(
#                 qkv.half().reshape(-1, 3, H, C // H),
#                 cu_seqlens,
#                 max_seqlen=self.patch_size,
#                 dropout_p=self.attn_drop if self.training else 0,
#                 softmax_scale=self.scale,
#             ).reshape(-1, C)
#             feat = feat.to(qkv.dtype)
#         feat = feat[inverse]

#         # ffn
#         feat = self.proj(feat)
#         feat = self.proj_drop(feat)
#         point.feat = feat
#         return point


# class MLP(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         hidden_channels=None,
#         out_channels=None,
#         act_layer=nn.GELU,
#         drop=0.0,
#     ):
#         super().__init__()
#         out_channels = out_channels or in_channels
#         hidden_channels = hidden_channels or in_channels
#         self.fc1 = nn.Linear(in_channels, hidden_channels)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_channels, out_channels)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# class Block(PointModule):
#     def __init__(
#         self,
#         channels,
#         num_heads,
#         patch_size=48,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         drop_path=0.0,
#         norm_layer=nn.LayerNorm,
#         act_layer=nn.GELU,
#         pre_norm=True,
#         order_index=0,
#         cpe_indice_key=None,
#         enable_rpe=False,
#         enable_flash=True,
#         upcast_attention=True,
#         upcast_softmax=True,
#     ):
#         super().__init__()
#         self.channels = channels
#         self.pre_norm = pre_norm

#         self.cpe = PointSequential(
#             spconv.SubMConv3d(
#                 channels,
#                 channels,
#                 kernel_size=3,
#                 bias=True,
#                 indice_key=cpe_indice_key,
#             ),
#             nn.Linear(channels, channels),
#             norm_layer(channels),
#         )

#         self.norm1 = PointSequential(norm_layer(channels))
#         self.attn = SerializedAttention(
#             channels=channels,
#             patch_size=patch_size,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             order_index=order_index,
#             enable_rpe=enable_rpe,
#             enable_flash=enable_flash,
#             upcast_attention=upcast_attention,
#             upcast_softmax=upcast_softmax,
#         )
#         self.norm2 = PointSequential(norm_layer(channels))
#         self.mlp = PointSequential(
#             MLP(
#                 in_channels=channels,
#                 hidden_channels=int(channels * mlp_ratio),
#                 out_channels=channels,
#                 act_layer=act_layer,
#                 drop=proj_drop,
#             )
#         )
#         self.drop_path = PointSequential(
#             DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         )

#     def forward(self, point: Point):
#         shortcut = point.feat
#         point = self.cpe(point)
#         point.feat = shortcut + point.feat
#         shortcut = point.feat
#         if self.pre_norm:
#             point = self.norm1(point)
#         point = self.drop_path(self.attn(point))
#         point.feat = shortcut + point.feat
#         if not self.pre_norm:
#             point = self.norm1(point)

#         shortcut = point.feat
#         if self.pre_norm:
#             point = self.norm2(point)
#         point = self.drop_path(self.mlp(point))
#         point.feat = shortcut + point.feat
#         if not self.pre_norm:
#             point = self.norm2(point)
#         point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
#         return point

    
    
    
    
    
# class Block2(PointModule):
#     def __init__(
#         self,
#         channels,
#         num_heads,
#         patch_size=96,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         drop_path=0.0,
#         norm_layer=nn.LayerNorm,
#         act_layer=nn.GELU,
#         pre_norm=True,
#         order_index=0,
#         cpe_indice_key=None,
#         enable_rpe=False,
#         enable_flash=True,
#         upcast_attention=True,
#         upcast_softmax=True,
#     ):
#         super().__init__()
#         self.channels = channels
#         self.pre_norm = pre_norm

#         self.cpe = PointSequential(
#             spconv.SubMConv3d(
#                 channels,
#                 channels,
#                 kernel_size=3,
#                 bias=True,
#                 indice_key=cpe_indice_key,
#             ),
#             nn.Linear(channels, channels),
#             norm_layer(channels),
#         )

#         self.norm1 = PointSequential(norm_layer(channels))
#         self.attn = SerializedAttention(
#             channels=channels,
#             patch_size=patch_size,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             order_index=order_index,
#             enable_rpe=enable_rpe,
#             enable_flash=enable_flash,
#             upcast_attention=upcast_attention,
#             upcast_softmax=upcast_softmax,
#         )
#         self.norm2 = PointSequential(norm_layer(channels))
#         self.mlp = PointSequential(
#             MLP(
#                 in_channels=channels,
#                 hidden_channels=int(channels * mlp_ratio),
#                 out_channels=channels,
#                 act_layer=act_layer,
#                 drop=proj_drop,
#             )
#         )
#         self.drop_path = PointSequential(
#             DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         )

#     def forward(self, point: Point):
#         shortcut = point.feat
#         point = self.cpe(point)
#         point.feat = shortcut + point.feat
#         shortcut = point.feat
#         if self.pre_norm:
#             point = self.norm1(point)
#         point = self.drop_path(self.attn(point))
#         point.feat = shortcut + point.feat
#         if not self.pre_norm:
#             point = self.norm1(point)

#         shortcut = point.feat
#         if self.pre_norm:
#             point = self.norm2(point)
#         point = self.drop_path(self.mlp(point))
#         point.feat = shortcut + point.feat
#         if not self.pre_norm:
#             point = self.norm2(point)
#         point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
#         return point


# class SerializedPooling(PointModule):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         stride=2,
#         norm_layer=None,
#         act_layer=None,
#         reduce="max",
#         shuffle_orders=True,
#         traceable=True,  # record parent and cluster
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
#         # TODO: add support to grid pool (any stride)
#         self.stride = stride
#         assert reduce in ["sum", "mean", "min", "max"]
#         self.reduce = reduce
#         self.shuffle_orders = shuffle_orders
#         self.traceable = traceable

#         self.proj = nn.Linear(in_channels, out_channels)
#         if norm_layer is not None:
#             self.norm = PointSequential(norm_layer(out_channels))
#         if act_layer is not None:
#             self.act = PointSequential(act_layer())

#     def forward(self, point: Point):
#         pooling_depth = (math.ceil(self.stride) - 1).bit_length()
#         if pooling_depth > point.serialized_depth:
#             pooling_depth = 0
#         assert {
#             "serialized_code",
#             "serialized_order",
#             "serialized_inverse",
#             "serialized_depth",
#         }.issubset(
#             point.keys()
#         ), "Run point.serialization() point cloud before SerializedPooling"

#         code = point.serialized_code >> pooling_depth * 3
#         code_, cluster, counts = torch.unique(
#             code[0],
#             sorted=True,
#             return_inverse=True,
#             return_counts=True,
#         )
#         # indices of point sorted by cluster, for torch_scatter.segment_csr
#         _, indices = torch.sort(cluster)
#         # index pointer for sorted point, for torch_scatter.segment_csr
#         idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
#         # head_indices of each cluster, for reduce attr e.g. code, batch
#         head_indices = indices[idx_ptr[:-1]]
#         # generate down code, order, inverse
#         code = code[:, head_indices]
#         order = torch.argsort(code)
#         inverse = torch.zeros_like(order).scatter_(
#             dim=1,
#             index=order,
#             src=torch.arange(0, code.shape[1], device=order.device).repeat(
#                 code.shape[0], 1
#             ),
#         )

#         if self.shuffle_orders:
#             perm = torch.randperm(code.shape[0])
#             code = code[perm]
#             order = order[perm]
#             inverse = inverse[perm]

#         # collect information
#         point_dict = Dict(
#             feat=torch_scatter.segment_csr(
#                 self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
#             ),
#             coord=torch_scatter.segment_csr(
#                 point.coord[indices], idx_ptr, reduce="mean"
#             ),
#             grid_coord=point.grid_coord[head_indices] >> pooling_depth,
#             serialized_code=code,
#             serialized_order=order,
#             serialized_inverse=inverse,
#             serialized_depth=point.serialized_depth - pooling_depth,
#             batch=point.batch[head_indices],
#         )

#         if "condition" in point.keys():
#             point_dict["condition"] = point.condition
#         if "context" in point.keys():
#             point_dict["context"] = point.context

#         if self.traceable:
#             point_dict["pooling_inverse"] = cluster
#             point_dict["pooling_parent"] = point
#         point = Point(point_dict)
#         if self.norm is not None:
#             point = self.norm(point)
#         if self.act is not None:
#             point = self.act(point)
#         point.sparsify()
#         return point


# class SerializedUnpooling(PointModule):
#     def __init__(
#         self,
#         in_channels,
#         skip_channels,
#         out_channels,
#         norm_layer=None,
#         act_layer=None,
#         traceable=False,  # record parent and cluster
#     ):
#         super().__init__()
#         self.proj = PointSequential(nn.Linear(in_channels, out_channels))
#         self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

#         if norm_layer is not None:
#             self.proj.add(norm_layer(out_channels))
#             self.proj_skip.add(norm_layer(out_channels))

#         if act_layer is not None:
#             self.proj.add(act_layer())
#             self.proj_skip.add(act_layer())

#         self.traceable = traceable

#     def forward(self, point):
#         assert "pooling_parent" in point.keys()
#         assert "pooling_inverse" in point.keys()
#         parent = point.pop("pooling_parent")
#         inverse = point.pop("pooling_inverse")
#         point = self.proj(point)
#         parent = self.proj_skip(parent)
#         parent.feat = parent.feat + point.feat[inverse]

#         if self.traceable:
#             parent["unpooling_parent"] = point
#         return parent


# class Embedding(PointModule):
#     def __init__(
#         self,
#         in_channels,
#         embed_channels,
#         norm_layer=None,
#         act_layer=None,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.embed_channels = embed_channels

#         # TODO: check remove spconv
#         self.stem = PointSequential(
#             conv=spconv.SubMConv3d(
#                 in_channels,
#                 embed_channels,
#                 kernel_size=5,
#                 padding=1,
#                 bias=False,
#                 indice_key="stem",
#             )
#         )
#         if norm_layer is not None:
#             self.stem.add(norm_layer(embed_channels), name="norm")
#         if act_layer is not None:
#             self.stem.add(act_layer(), name="act")

#     def forward(self, point: Point):
#         point = self.stem(point)
#         return point


# @MODELS.register_module("PT-v3m1")

# class PointTransformerV3(PointModule):
#     def __init__(
#         self,
#         in_channels=6,
#         order=("z", "z-trans"),
#         stride=(2, 2, 2, 2),
#         enc_depths=(2, 2, 2, 6, 2),
#         enc_channels=(32, 64, 128, 256, 512),
#         enc_num_head=(2, 4, 8, 16, 32),
#         enc_patch_size=(48, 48, 48, 48, 48),
#         enc_patch_size2=(96, 96, 96, 96, 96),  # 新增的 Block2 的 patch_size 参数
#         dec_depths=(2, 2, 2, 2),
#         dec_channels=(64, 64, 128, 256),
#         dec_num_head=(4, 4, 8, 16),
#         dec_patch_size=(48, 48, 48, 48),
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         drop_path=0.3,
#         pre_norm=True,
#         shuffle_orders=True,
#         enable_rpe=False,
#         enable_flash=True,
#         upcast_attention=False,
#         upcast_softmax=False,
#         cls_mode=False,
#         pdnorm_bn=False,
#         pdnorm_ln=False,
#         pdnorm_decouple=True,
#         pdnorm_adaptive=False,
#         pdnorm_affine=True,
#         pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
#         fusion_level=2  # 新增的融合层级参数
#     ):
#         super().__init__()
#         self.num_stages = len(enc_depths)
#         self.order = [order] if isinstance(order, str) else order
#         self.cls_mode = cls_mode
#         self.shuffle_orders = shuffle_orders
#         self.fusion_level = fusion_level #新增的融合参数
#         self.alpha = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)#新增一个可学习的block1和block2的比重


#         # 保存参数为实例变量
#         self.enc_depths = enc_depths
#         self.enc_channels = enc_channels
#         self.enc_num_head = enc_num_head
#         self.enc_patch_size = enc_patch_size
#         self.enc_patch_size2 = enc_patch_size2
#         self.dec_depths = dec_depths
#         self.dec_channels = dec_channels
#         self.dec_num_head = dec_num_head
#         self.dec_patch_size = dec_patch_size

#         # norm layers
#         if pdnorm_bn:
#             bn_layer = partial(
#                 PDNorm,
#                 norm_layer=partial(
#                     nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
#                 ),
#                 conditions=pdnorm_conditions,
#                 decouple=pdnorm_decouple,
#                 adaptive=pdnorm_adaptive,
#             )
#         else:
#             bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
#         if pdnorm_ln:
#             ln_layer = partial(
#                 PDNorm,
#                 norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
#                 conditions=pdnorm_conditions,
#                 decouple=pdnorm_decouple,
#                 adaptive=pdnorm_adaptive,
#             )
#         else:
#             ln_layer = nn.LayerNorm
#         # activation layers
#         act_layer = nn.GELU

#         self.embedding = Embedding(
#             in_channels=in_channels,
#             embed_channels=enc_channels[0],
#             norm_layer=bn_layer,
#             act_layer=act_layer,
#         )

#         # encoder
#         enc_drop_path = [
#             x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
#         ]
#         self.enc = PointSequential()
#         for s in range(self.num_stages):
#             enc_drop_path_ = enc_drop_path[
#                 sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
#             ]
#             enc = PointSequential()
#             if s > 0:
#                 enc.add(
#                     SerializedPooling(
#                         in_channels=enc_channels[s - 1],
#                         out_channels=enc_channels[s],
#                         stride=stride[s - 1],
#                         norm_layer=bn_layer,
#                         act_layer=act_layer,
#                     ),
#                     name="down",
#                 )
#             for i in range(enc_depths[s]):
#                 # 定义 Block1
#                 enc.add(
#                     Block(
#                         channels=enc_channels[s],
#                         num_heads=enc_num_head[s],
#                         patch_size=enc_patch_size[s],
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias,
#                         qk_scale=qk_scale,
#                         attn_drop=attn_drop,
#                         proj_drop=proj_drop,
#                         drop_path=enc_drop_path_[i],
#                         norm_layer=ln_layer,
#                         act_layer=act_layer,
#                         pre_norm=pre_norm,
#                         order_index=i % len(self.order),
#                         cpe_indice_key=f"stage{s}",
#                         enable_rpe=enable_rpe,
#                         enable_flash=enable_flash,
#                         upcast_attention=upcast_attention,
#                         upcast_softmax=upcast_softmax,
#                     ),
#                     name=f"block{i}_1",
#                 )
#                 # 定义 Block2
#                 enc.add(
#                     Block(
#                         channels=enc_channels[s],
#                         num_heads=enc_num_head[s],
#                         patch_size=enc_patch_size2[s],  # 使用不同的 patch_size
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias,
#                         qk_scale=qk_scale,
#                         attn_drop=attn_drop,
#                         proj_drop=proj_drop,
#                         drop_path=enc_drop_path_[i],
#                         norm_layer=ln_layer,
#                         act_layer=act_layer,
#                         pre_norm=pre_norm,
#                         order_index=i % len(self.order),
#                         cpe_indice_key=f"stage{s}",
#                         enable_rpe=enable_rpe,
#                         enable_flash=enable_flash,
#                         upcast_attention=upcast_attention,
#                         upcast_softmax=upcast_softmax,
#                     ),
#                     name=f"block{i}_2",
#                 )
#             if len(enc) != 0:
#                 self.enc.add(module=enc, name=f"enc{s}")

#         # decoder
#         if not self.cls_mode:
#             dec_drop_path = [
#                 x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
#             ]
#             self.dec = PointSequential()
#             dec_channels = list(dec_channels) + [enc_channels[-1]]
#             for s in reversed(range(self.num_stages - 1)):
#                 dec_drop_path_ = dec_drop_path[
#                     sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
#                 ]
#                 dec_drop_path_.reverse()
#                 dec = PointSequential()
#                 dec.add(
#                     SerializedUnpooling(
#                         in_channels=dec_channels[s + 1],
#                         skip_channels=enc_channels[s],
#                         out_channels=dec_channels[s],
#                         norm_layer=bn_layer,
#                         act_layer=act_layer,
#                     ),
#                     name="up",
#                 )
#                 for i in range(dec_depths[s]):
#                     dec.add(
#                         Block(
#                             channels=dec_channels[s],
#                             num_heads=dec_num_head[s],
#                             patch_size=dec_patch_size[s],
#                             mlp_ratio=mlp_ratio,
#                             qkv_bias=qkv_bias,
#                             qk_scale=qk_scale,
#                             attn_drop=attn_drop,
#                             proj_drop=proj_drop,
#                             drop_path=dec_drop_path_[i],
#                             norm_layer=ln_layer,
#                             act_layer=act_layer,
#                             pre_norm=pre_norm,
#                             order_index=i % len(self.order),
#                             cpe_indice_key=f"stage{s}",
#                             enable_rpe=enable_rpe,
#                             enable_flash=enable_flash,
#                             upcast_attention=upcast_attention,
#                             upcast_softmax=upcast_softmax,
#                         ),
#                         name=f"block{i}",
#                     )
#                 self.dec.add(module=dec, name=f"dec{s}")
#     def replace_feature(self, point, new_feat):
#         # 替换 point 对象中的 feat 特征
#         point.feat = new_feat
#         if hasattr(point, 'sparse_conv_feat'):
#             point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(new_feat)
#         return point

#     def clone(self, point):
#         # 创建 point 对象的深拷贝，同时保留所有已有的属性和方法
#         new_point = Point()
#         for key, value in point.items():
#             if isinstance(value, torch.Tensor):
#                 new_point[key] = value.clone()  # 对于 Tensor 使用 clone
#             else:
#                 new_point[key] = copy.deepcopy(value)  # 对于其他数据使用深拷贝
#         return new_point

#     def forward(self, data_dict):
#         point1 = self.clone(Point(data_dict))
#         point2 = self.clone(Point(data_dict))

#         point1.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
#         point2.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
#         point1.sparsify()
#         point2.sparsify()

#         point1 = self.embedding(point1)
#         point2 = self.embedding(point2)

#         for s in range(self.num_stages):
#             enc = self.enc._modules[f'enc{s}']
#             enc_depth = self.enc_depths[s]

#             if s > 0:
#                 down = enc._modules['down']
#                 point1 = down(point1)
#                 point2 = down(point2)

#             for i in range(enc_depth):
#                 block1 = enc._modules[f'block{i}_1']
#                 block2 = enc._modules[f'block{i}_2']

#                 point1 = block1(point1)
#                 point2 = block2(point2)

#             if s >= self.fusion_level:
#                 alpha = 0.5  # 可以设为可学习参数
#                 point1 = self.replace_feature(point1, self.alpha * point1.feat + (1 - self.alpha) * point2.feat)

#         point = point1
#         if not self.cls_mode:
#             point = self.dec(point)
#         return point






# """
# ###第一次修改的类
# class PointTransformerV3(PointModule):
#     def __init__(
#         self,
#         in_channels=6,
#         order=("z", "z-trans"),
#         stride=(2, 2, 2, 2),
#         enc_depths=(2, 2, 2, 6, 2),
#         enc_channels=(32, 64, 128, 256, 512),
#         enc_num_head=(2, 4, 8, 16, 32),
#         enc_patch_size=(48, 48, 48, 48, 48),
#         enc_patch_size2=(96, 96, 96, 96, 96),
#         dec_depths=(2, 2, 2, 2),
#         dec_channels=(64, 64, 128, 256),
#         dec_num_head=(4, 4, 8, 16),
#         dec_patch_size=(48, 48, 48, 48),
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         drop_path=0.3,
#         pre_norm=True,
#         shuffle_orders=True,
#         enable_rpe=False,
#         enable_flash=True,
#         upcast_attention=False,
#         upcast_softmax=False,
#         cls_mode=False,
#         pdnorm_bn=False,
#         pdnorm_ln=False,
#         pdnorm_decouple=True,
#         pdnorm_adaptive=False,
#         pdnorm_affine=True,
#         pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
#     ):
#         super().__init__()
#         self.num_stages = len(enc_depths)
#         self.order = [order] if isinstance(order, str) else order
#         self.cls_mode = cls_mode
#         self.shuffle_orders = shuffle_orders
        
#                 # 保存参数为实例变量
#         self.enc_depths = enc_depths
#         self.enc_channels = enc_channels
#         self.enc_num_head = enc_num_head
#         self.enc_patch_size = enc_patch_size
#         self.enc_patch_size2 = enc_patch_size2
#         self.dec_depths = dec_depths
#         self.dec_channels = dec_channels
#         self.dec_num_head = dec_num_head
#         self.dec_patch_size = dec_patch_size

#         assert self.num_stages == len(stride) + 1
#         assert self.num_stages == len(enc_depths)
#         assert self.num_stages == len(enc_channels)
#         assert self.num_stages == len(enc_num_head)
#         assert self.num_stages == len(enc_patch_size)
#         assert self.num_stages == len(enc_patch_size2)
#         assert self.cls_mode or self.num_stages == len(dec_depths) + 1
#         assert self.cls_mode or self.num_stages == len(dec_channels) + 1
#         assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
#         assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

#         # norm layers
#         if pdnorm_bn:
#             bn_layer = partial(
#                 PDNorm,
#                 norm_layer=partial(
#                     nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
#                 ),
#                 conditions=pdnorm_conditions,
#                 decouple=pdnorm_decouple,
#                 adaptive=pdnorm_adaptive,
#             )
#         else:
#             bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
#         if pdnorm_ln:
#             ln_layer = partial(
#                 PDNorm,
#                 norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
#                 conditions=pdnorm_conditions,
#                 decouple=pdnorm_decouple,
#                 adaptive=pdnorm_adaptive,
#             )
#         else:
#             ln_layer = nn.LayerNorm
#         # activation layers
#         act_layer = nn.GELU

#         self.embedding = Embedding(
#             in_channels=in_channels,
#             embed_channels=enc_channels[0],
#             norm_layer=bn_layer,
#             act_layer=act_layer,
#         )

#         # encoder
#         enc_drop_path = [
#             x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
#         ]
#         self.enc = PointSequential()
#         for s in range(self.num_stages):
#             enc_drop_path_ = enc_drop_path[
#                 sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
#             ]
#             enc = PointSequential()
#             if s > 0:
#                 enc.add(
#                     SerializedPooling(
#                         in_channels=enc_channels[s - 1],
#                         out_channels=enc_channels[s],
#                         stride=stride[s - 1],
#                         norm_layer=bn_layer,
#                         act_layer=act_layer,
#                     ),
#                     name="down",
#                 )
#             for i in range(enc_depths[s]):
#                 enc.add(
#                     Block(
#                         channels=enc_channels[s],
#                         num_heads=enc_num_head[s],
#                         patch_size=enc_patch_size[s],
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias,
#                         qk_scale=qk_scale,
#                         attn_drop=attn_drop,
#                         proj_drop=proj_drop,
#                         drop_path=enc_drop_path_[i],
#                         norm_layer=ln_layer,
#                         act_layer=act_layer,
#                         pre_norm=pre_norm,
#                         order_index=i % len(self.order),
#                         cpe_indice_key=f"stage{s}",
#                         enable_rpe=enable_rpe,
#                         enable_flash=enable_flash,
#                         upcast_attention=upcast_attention,
#                         upcast_softmax=upcast_softmax,
#                     ),
#                     name=f"block{i}_1",
#                 )
#                 enc.add(
#                     Block2(
#                         channels=enc_channels[s],
#                         num_heads=enc_num_head[s],
#                         patch_size=enc_patch_size2[s],
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias,
#                         qk_scale=qk_scale,
#                         attn_drop=attn_drop,
#                         proj_drop=proj_drop,
#                         drop_path=enc_drop_path_[i],
#                         norm_layer=ln_layer,
#                         act_layer=act_layer,
#                         pre_norm=pre_norm,
#                         order_index=i % len(self.order),
#                         cpe_indice_key=f"stage{s}",
#                         enable_rpe=enable_rpe,
#                         enable_flash=enable_flash,
#                         upcast_attention=upcast_attention,
#                         upcast_softmax=upcast_softmax,
#                     ),
#                     name=f"block{i}_2",
#                 )
#             if len(enc) != 0:
#                 self.enc.add(module=enc, name=f"enc{s}")

#         # decoder
#         if not self.cls_mode:
#             dec_drop_path = [
#                 x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
#             ]
#             self.dec = PointSequential()
#             dec_channels = list(dec_channels) + [enc_channels[-1]]
#             for s in reversed(range(self.num_stages - 1)):
#                 dec_drop_path_ = dec_drop_path[
#                     sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
#                 ]
#                 dec_drop_path_.reverse()
#                 dec = PointSequential()
#                 dec.add(
#                     SerializedUnpooling(
#                         in_channels=dec_channels[s + 1],
#                         skip_channels=enc_channels[s],
#                         out_channels=dec_channels[s],
#                         norm_layer=bn_layer,
#                         act_layer=act_layer,
#                     ),
#                     name="up",
#                 )
#                 for i in range(dec_depths[s]):
#                     dec.add(
#                         Block(
#                             channels=dec_channels[s],
#                             num_heads=dec_num_head[s],
#                             patch_size=dec_patch_size[s],
#                             mlp_ratio=mlp_ratio,
#                             qkv_bias=qkv_bias,
#                             qk_scale=qk_scale,
#                             attn_drop=attn_drop,
#                             proj_drop=proj_drop,
#                             drop_path=dec_drop_path_[i],
#                             norm_layer=ln_layer,
#                             act_layer=act_layer,
#                             pre_norm=pre_norm,
#                             order_index=i % len(self.order),
#                             cpe_indice_key=f"stage{s}",
#                             enable_rpe=enable_rpe,
#                             enable_flash=enable_flash,
#                             upcast_attention=upcast_attention,
#                             upcast_softmax=upcast_softmax,
#                         ),
#                         name=f"block{i}",
#                     )
#                 self.dec.add(module=dec, name=f"dec{s}")
                

#     def forward(self, data_dict):
#         point1 = Point(copy.deepcopy(data_dict))
#         point2 = Point(copy.deepcopy(data_dict))

#         point1.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
#         point2.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
#         point1.sparsify()
#         point2.sparsify()

#         point1 = self.embedding(point1)
#         point2 = self.embedding(point2)

#         for s in range(self.num_stages):
#             enc = self.enc._modules[f'enc{s}']
#             enc_depth = self.enc_depths[s]

#             if s > 0:
#                 down = enc._modules['down']
#                 point1 = down(point1)
#                 point2 = down(point2)

#             for i in range(enc_depth):
#                 block1 = enc._modules[f'block{i}_1']
#                 block2 = enc._modules[f'block{i}_2']

#                 point1 = block1(point1)

#                 # 屏蔽 Block2
#                 temp_point2 = block2(point2)

#                 # 将 block2 的所有相关属性置为零
#                 temp_point2.feat = temp_point2.feat * 0
#                 # 确保 sparse_conv_feat 的特征被清空
#                 if hasattr(temp_point2, 'sparse_conv_feat'):
#                     temp_point2.sparse_conv_feat = temp_point2.sparse_conv_feat.replace_feature(temp_point2.feat)
#                 # 如果有其他可能的属性，可以添加如下
#                 # if hasattr(temp_point2, 'other_property'):
#                 #     temp_point2.other_property = 0  # 具体属性名替换为实际的属性名

#                 point2.feat = temp_point2.feat  # 更新 point2 的特征
#                 point2.sparse_conv_feat = point2.sparse_conv_feat.replace_feature(point2.feat)

#             if s >= self.num_stages - 2:
#                 point1.feat = point1.feat + point2.feat
#                 point1.sparse_conv_feat = point1.sparse_conv_feat.replace_feature(point1.feat)
#                 point2.feat = point1.feat
#                 point2.sparse_conv_feat = point1.sparse_conv_feat

#         point = point1
#         if not self.cls_mode:
#             point = self.dec(point)
#         return point
# """              


                
# """
# ###倒数第二层开始融合
#     def forward(self, data_dict):
#         # 创建两个相同的 Point 对象
#         point1 = Point(copy.deepcopy(data_dict))
#         point2 = Point(copy.deepcopy(data_dict))

#         # 序列化和稀疏化
#         point1.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
#         point2.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
#         point1.sparsify()
#         point2.sparsify()

#         point1 = self.embedding(point1)
#         point2 = self.embedding(point2)

#         for s in range(self.num_stages):
#             enc = self.enc._modules[f'enc{s}']
#             enc_depth = self.enc_depths[s]  # __init__ 中定义 self.enc_depths

#             if s > 0:
#                 down = enc._modules['down']
#                 point1 = down(point1)
#                 point2 = down(point2)

#             for i in range(enc_depth):
#                 # 获取 Block 和 Block2
#                 block1 = enc._modules[f'block{i}_1']
#                 block2 = enc._modules[f'block{i}_2']
#                 # 分别处理
#                 point1 = block1(point1)
#                 #point2 = block2(point2)
#                 ###屏蔽Block2
#                 temp_point2 = block2(point2)
#                 temp_point2.feat = temp_point2.feat * 0
#                 point2.feat = temp_point2.feat  # 更新 point2 的特征
#                 point2.sparse_conv_feat = point2.sparse_conv_feat.replace_feature(point2.feat)

            
#             if s >= self.num_stages - 2:
#                 # 在最深的两层进行特征融合
#                 point1.feat = point1.feat + point2.feat
#                 # 更新 sparse_conv_feat 的特征
#                 point1.sparse_conv_feat = point1.sparse_conv_feat.replace_feature(point1.feat)
#                 # 同步 point2 的特征（可选）
#                 point2.feat = point1.feat
#                 point2.sparse_conv_feat = point1.sparse_conv_feat

#         # 使用融合后的 point1 作为后续输入
#         point = point1
#         if not self.cls_mode:
#             point = self.dec(point)
#         return point
# """
# """
# ###下采样层进行融合
#     def forward(self, data_dict):
#             # 创建两个相同的 Point 对象
#         # point1 = Point(data_dict)
#         # point2 = Point(data_dict)
#         point1 = Point(copy.deepcopy(data_dict))
#         point2 = Point(copy.deepcopy(data_dict))
    
#         # 序列化和稀疏化
#         point1.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
#         point2.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
#         point1.sparsify()
#         point2.sparsify()


#         point1 = self.embedding(point1)
#         point2 = self.embedding(point2)

#         # point1 = self.enc(point1)
#         # point2 = self.enc(point2)
        
#         for s in range(self.num_stages):
#             enc = self.enc._modules[f'enc{s}']
#             enc_depth = self.enc_depths[s]  #  __init__ 中定义self.enc_depths
#             if s > 0:
#                 down = enc._modules['down']
#                 point1 = down(point1)
#                 point2 = down(point2)
#             for i in range(enc_depth):
#                 # 获取 Block 和 Block2
#                 block1 = enc._modules[f'block{i}_1']
#                 block2 = enc._modules[f'block{i}_2']
#                 # 分别处理
#                 point1 = block1(point1)
#                 point2 = block2(point2)
#             # 融合特征
#             point1.feat = point1.feat + point2.feat
#             # 更新 sparse_conv_feat 的特征
#             point1.sparse_conv_feat = point1.sparse_conv_feat.replace_feature(point1.feat)

#         # 使用融合后的 point1 作为后续输入
#         point = point1
#         if not self.cls_mode:
#             point = self.dec(point)
#         # else:
#         #     point.feat = torch_scatter.segment_csr(
#         #         src=point.feat,
#         #         indptr=nn.functional.pad(point.offset, (1, 0)),
#         #         reduce="mean",
#         #     )
#         return point







"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
import sys
try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential
from scipy.spatial import cKDTree
from collections import OrderedDict
from .serialization import encode


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()
def knn(x, k):
    tree = cKDTree(x.cpu().numpy())
    _, idx = tree.query(x.cpu().numpy(), k=k)
    return torch.from_numpy(idx).to(x.device)

class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class PDNorm(PointModule):##替换新类
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point

class SpatialGate(nn.Module):
    """ Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim = -1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()

        return x1 * x2
###新类
class EnhancedRPE(torch.nn.Module):
    """增强的相对位置编码，结合多尺度和几何特征。"""
    def __init__(self, patch_sizes, num_heads, additional_dims=2):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.num_heads = num_heads
        self.additional_dims = additional_dims
        self.rpe_modules = nn.ModuleList([
            self._create_rpe(patch_size) for patch_size in patch_sizes
        ])
        # 处理几何特征的线性层
        self.geo_linear = nn.Linear(additional_dims, num_heads)
    
    def _create_rpe(self, patch_size):
        pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        rpe_num = 2 * pos_bnd + 1
        rpe_table = torch.nn.Parameter(torch.zeros(5 * rpe_num, self.num_heads))
        torch.nn.init.trunc_normal_(rpe_table, std=0.02)
        return rpe_table
    
    def forward(self, coord, normal=None, curvature=None):
        rpe_sum = 0
        for rpe_table, patch_size in zip(self.rpe_modules, self.patch_sizes):
            pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
            rpe_num = 2 * pos_bnd + 1
            idx = (
                coord.clamp(-pos_bnd, pos_bnd)
                + pos_bnd
                + torch.arange(5, device=coord.device) * rpe_num
            )
            out = rpe_table.index_select(0, idx.reshape(-1))
            out = out.view(idx.shape + (-1,)).sum(3)
            rpe_sum += out.permute(0, 3, 1, 2)  # (N, H, K, K)
        
        if normal is not None and curvature is not None:
            # 将几何特征通过线性层处理
            geo_feat = torch.cat([normal, curvature], dim=-1)  # 假设 normal 和 curvature 维度匹配
            geo_encoding = self.geo_linear(geo_feat).unsqueeze(-1).unsqueeze(-1)  # 形状调整以匹配 rpe_sum
            rpe_sum += geo_encoding
        return rpe_sum



class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out

class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        multi_scale_rpe=False,  # 新增参数
        rpe_patch_sizes=[48, 96],
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        self.multi_scale_rpe=multi_scale_rpe,  
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        # 使用增强的RPE
        if enable_rpe:
            if self.multi_scale_rpe:
                self.rpe = EnhancedRPE(rpe_patch_sizes, num_heads, additional_dims=4)  # 例如，法向量和曲率
            else:
                self.rpe = EnhancedRPE([patch_size], num_heads, additional_dims=4)
        
        # self.rpe = EnhancedRPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                normal = point.normal  # 获取法向量
                curvature = point.curvature  # 获取曲率
                rpe = self.rpe(self.get_rel_pos(point, order), normal, curvature)
                attn = attn + rpe
                # attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point

class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent
###新类
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 块，用于通道特征的自适应重标定"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        se = x.mean(dim=1)  # (B, C)
        se = self.fc1(se)    # (B, C//reduction)
        se = self.relu(se)
        se = self.fc2(se)    # (B, C)
        se = self.sigmoid(se).view(B, 1, C)  # (B, 1, C)
        return x * se.expand_as(x)  # (B, N, C)

# 修改Embedding类以接受多模态输入
class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
        multimodal_channels=0  # 新增参数，用于多模态特征的通道数
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        total_in_channels = in_channels + multimodal_channels  # 调整输入通道数

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                total_in_channels, # 增加多模态输入通道数
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        if 'multimodal_feat' in point.keys():
            point.feat = torch.cat([point.feat, point.multimodal_feat], dim=1)
        point = self.stem(point)
        return point


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        # self.se = SEBlock(channels, reduction=16)  # 添加SE块

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        # point.feat = self.se(point.feat)  # 应用SE块    
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

@MODELS.register_module("PT-v3m1") # 主要的类
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):

        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
        return point


    
    
    
    
    

# """
# Point Transformer - V3 Mode1 base_line

# Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
# Please cite our work if the code is helpful to you.
# """

# from functools import partial
# from addict import Dict
# import math
# import torch
# import torch.nn as nn
# import spconv.pytorch as spconv
# import torch_scatter
# from timm.models.layers import DropPath

# try:
#     import flash_attn
# except ImportError:
#     flash_attn = None

# from pointcept.models.point_prompt_training import PDNorm
# from pointcept.models.builder import MODELS
# from pointcept.models.utils.misc import offset2bincount
# from pointcept.models.utils.structure import Point
# from pointcept.models.modules import PointModule, PointSequential


# class RPE(torch.nn.Module):
#     def __init__(self, patch_size, num_heads):
#         super().__init__()
#         self.patch_size = patch_size
#         self.num_heads = num_heads
#         self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
#         self.rpe_num = 2 * self.pos_bnd + 1
#         self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
#         torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

#     def forward(self, coord):
#         idx = (
#             coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
#             + self.pos_bnd  # relative position to positive index
#             + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
#         )
#         out = self.rpe_table.index_select(0, idx.reshape(-1))
#         out = out.view(idx.shape + (-1,)).sum(3)
#         out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
#         return out


# class SerializedAttention(PointModule):
#     def __init__(
#         self,
#         channels,
#         num_heads,
#         patch_size,
#         qkv_bias=True,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         order_index=0,
#         enable_rpe=False,
#         enable_flash=True,
#         upcast_attention=True,
#         upcast_softmax=True,
#     ):
#         super().__init__()
#         assert channels % num_heads == 0
#         self.channels = channels
#         self.num_heads = num_heads
#         self.scale = qk_scale or (channels // num_heads) ** -0.5
#         self.order_index = order_index
#         self.upcast_attention = upcast_attention
#         self.upcast_softmax = upcast_softmax
#         self.enable_rpe = enable_rpe
#         self.enable_flash = enable_flash
#         if enable_flash:
#             assert (
#                 enable_rpe is False
#             ), "Set enable_rpe to False when enable Flash Attention"
#             assert (
#                 upcast_attention is False
#             ), "Set upcast_attention to False when enable Flash Attention"
#             assert (
#                 upcast_softmax is False
#             ), "Set upcast_softmax to False when enable Flash Attention"
#             assert flash_attn is not None, "Make sure flash_attn is installed."
#             self.patch_size = patch_size
#             self.attn_drop = attn_drop
#         else:
#             # when disable flash attention, we still don't want to use mask
#             # consequently, patch size will auto set to the
#             # min number of patch_size_max and number of points
#             self.patch_size_max = patch_size
#             self.patch_size = 0
#             self.attn_drop = torch.nn.Dropout(attn_drop)

#         self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
#         self.proj = torch.nn.Linear(channels, channels)
#         self.proj_drop = torch.nn.Dropout(proj_drop)
#         self.softmax = torch.nn.Softmax(dim=-1)
#         self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

#     @torch.no_grad()
#     def get_rel_pos(self, point, order):
#         K = self.patch_size
#         rel_pos_key = f"rel_pos_{self.order_index}"
#         if rel_pos_key not in point.keys():
#             grid_coord = point.grid_coord[order]
#             grid_coord = grid_coord.reshape(-1, K, 3)
#             point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
#         return point[rel_pos_key]

#     @torch.no_grad()
#     def get_padding_and_inverse(self, point):
#         pad_key = "pad"
#         unpad_key = "unpad"
#         cu_seqlens_key = "cu_seqlens_key"
#         if (
#             pad_key not in point.keys()
#             or unpad_key not in point.keys()
#             or cu_seqlens_key not in point.keys()
#         ):
#             offset = point.offset
#             bincount = offset2bincount(offset)
#             bincount_pad = (
#                 torch.div(
#                     bincount + self.patch_size - 1,
#                     self.patch_size,
#                     rounding_mode="trunc",
#                 )
#                 * self.patch_size
#             )
#             # only pad point when num of points larger than patch_size
#             mask_pad = bincount > self.patch_size
#             bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
#             _offset = nn.functional.pad(offset, (1, 0))
#             _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
#             pad = torch.arange(_offset_pad[-1], device=offset.device)
#             unpad = torch.arange(_offset[-1], device=offset.device)
#             cu_seqlens = []
#             for i in range(len(offset)):
#                 unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
#                 if bincount[i] != bincount_pad[i]:
#                     pad[
#                         _offset_pad[i + 1]
#                         - self.patch_size
#                         + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
#                     ] = pad[
#                         _offset_pad[i + 1]
#                         - 2 * self.patch_size
#                         + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
#                         - self.patch_size
#                     ]
#                 pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
#                 cu_seqlens.append(
#                     torch.arange(
#                         _offset_pad[i],
#                         _offset_pad[i + 1],
#                         step=self.patch_size,
#                         dtype=torch.int32,
#                         device=offset.device,
#                     )
#                 )
#             point[pad_key] = pad
#             point[unpad_key] = unpad
#             point[cu_seqlens_key] = nn.functional.pad(
#                 torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
#             )
#         return point[pad_key], point[unpad_key], point[cu_seqlens_key]

#     def forward(self, point):
#         if not self.enable_flash:
#             self.patch_size = min(
#                 offset2bincount(point.offset).min().tolist(), self.patch_size_max
#             )

#         H = self.num_heads
#         K = self.patch_size
#         C = self.channels

#         pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

#         order = point.serialized_order[self.order_index][pad]
#         inverse = unpad[point.serialized_inverse[self.order_index]]

#         # padding and reshape feat and batch for serialized point patch
#         qkv = self.qkv(point.feat)[order]

#         if not self.enable_flash:
#             # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
#             q, k, v = (
#                 qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
#             )
#             # attn
#             if self.upcast_attention:
#                 q = q.float()
#                 k = k.float()
#             attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
#             if self.enable_rpe:
#                 attn = attn + self.rpe(self.get_rel_pos(point, order))
#             if self.upcast_softmax:
#                 attn = attn.float()
#             attn = self.softmax(attn)
#             attn = self.attn_drop(attn).to(qkv.dtype)
#             feat = (attn @ v).transpose(1, 2).reshape(-1, C)
#         else:
#             feat = flash_attn.flash_attn_varlen_qkvpacked_func(
#                 qkv.half().reshape(-1, 3, H, C // H),
#                 cu_seqlens,
#                 max_seqlen=self.patch_size,
#                 dropout_p=self.attn_drop if self.training else 0,
#                 softmax_scale=self.scale,
#             ).reshape(-1, C)
#             feat = feat.to(qkv.dtype)
#         feat = feat[inverse]

#         # ffn
#         feat = self.proj(feat)
#         feat = self.proj_drop(feat)
#         point.feat = feat
#         return point


# class MLP(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         hidden_channels=None,
#         out_channels=None,
#         act_layer=nn.GELU,
#         drop=0.0,
#     ):
#         super().__init__()
#         out_channels = out_channels or in_channels
#         hidden_channels = hidden_channels or in_channels
#         self.fc1 = nn.Linear(in_channels, hidden_channels)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_channels, out_channels)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# class Block(PointModule):
#     def __init__(
#         self,
#         channels,
#         num_heads,
#         patch_size=48,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         drop_path=0.0,
#         norm_layer=nn.LayerNorm,
#         act_layer=nn.GELU,
#         pre_norm=True,
#         order_index=0,
#         cpe_indice_key=None,
#         enable_rpe=False,
#         enable_flash=True,
#         upcast_attention=True,
#         upcast_softmax=True,
#     ):
#         super().__init__()
#         self.channels = channels
#         self.pre_norm = pre_norm

#         self.cpe = PointSequential(
#             spconv.SubMConv3d(
#                 channels,
#                 channels,
#                 kernel_size=3,
#                 bias=True,
#                 indice_key=cpe_indice_key,
#             ),
#             nn.Linear(channels, channels),
#             norm_layer(channels),
#         )

#         self.norm1 = PointSequential(norm_layer(channels))
#         self.attn = SerializedAttention(
#             channels=channels,
#             patch_size=patch_size,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             order_index=order_index,
#             enable_rpe=enable_rpe,
#             enable_flash=enable_flash,
#             upcast_attention=upcast_attention,
#             upcast_softmax=upcast_softmax,
#         )
#         self.norm2 = PointSequential(norm_layer(channels))
#         self.mlp = PointSequential(
#             MLP(
#                 in_channels=channels,
#                 hidden_channels=int(channels * mlp_ratio),
#                 out_channels=channels,
#                 act_layer=act_layer,
#                 drop=proj_drop,
#             )
#         )
#         self.drop_path = PointSequential(
#             DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         )

#     def forward(self, point: Point):
#         shortcut = point.feat
#         point = self.cpe(point)
#         point.feat = shortcut + point.feat
#         shortcut = point.feat
#         if self.pre_norm:
#             point = self.norm1(point)
#         point = self.drop_path(self.attn(point))
#         point.feat = shortcut + point.feat
#         if not self.pre_norm:
#             point = self.norm1(point)

#         shortcut = point.feat
#         if self.pre_norm:
#             point = self.norm2(point)
#         point = self.drop_path(self.mlp(point))
#         point.feat = shortcut + point.feat
#         if not self.pre_norm:
#             point = self.norm2(point)
#         point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
#         return point


# class SerializedPooling(PointModule):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         stride=2,
#         norm_layer=None,
#         act_layer=None,
#         reduce="max",
#         shuffle_orders=True,
#         traceable=True,  # record parent and cluster
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
#         # TODO: add support to grid pool (any stride)
#         self.stride = stride
#         assert reduce in ["sum", "mean", "min", "max"]
#         self.reduce = reduce
#         self.shuffle_orders = shuffle_orders
#         self.traceable = traceable

#         self.proj = nn.Linear(in_channels, out_channels)
#         if norm_layer is not None:
#             self.norm = PointSequential(norm_layer(out_channels))
#         if act_layer is not None:
#             self.act = PointSequential(act_layer())

#     def forward(self, point: Point):
#         pooling_depth = (math.ceil(self.stride) - 1).bit_length()
#         if pooling_depth > point.serialized_depth:
#             pooling_depth = 0
#         assert {
#             "serialized_code",
#             "serialized_order",
#             "serialized_inverse",
#             "serialized_depth",
#         }.issubset(
#             point.keys()
#         ), "Run point.serialization() point cloud before SerializedPooling"

#         code = point.serialized_code >> pooling_depth * 3
#         code_, cluster, counts = torch.unique(
#             code[0],
#             sorted=True,
#             return_inverse=True,
#             return_counts=True,
#         )
#         # indices of point sorted by cluster, for torch_scatter.segment_csr
#         _, indices = torch.sort(cluster)
#         # index pointer for sorted point, for torch_scatter.segment_csr
#         idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
#         # head_indices of each cluster, for reduce attr e.g. code, batch
#         head_indices = indices[idx_ptr[:-1]]
#         # generate down code, order, inverse
#         code = code[:, head_indices]
#         order = torch.argsort(code)
#         inverse = torch.zeros_like(order).scatter_(
#             dim=1,
#             index=order,
#             src=torch.arange(0, code.shape[1], device=order.device).repeat(
#                 code.shape[0], 1
#             ),
#         )

#         if self.shuffle_orders:
#             perm = torch.randperm(code.shape[0])
#             code = code[perm]
#             order = order[perm]
#             inverse = inverse[perm]

#         # collect information
#         point_dict = Dict(
#             feat=torch_scatter.segment_csr(
#                 self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
#             ),
#             coord=torch_scatter.segment_csr(
#                 point.coord[indices], idx_ptr, reduce="mean"
#             ),
#             grid_coord=point.grid_coord[head_indices] >> pooling_depth,
#             serialized_code=code,
#             serialized_order=order,
#             serialized_inverse=inverse,
#             serialized_depth=point.serialized_depth - pooling_depth,
#             batch=point.batch[head_indices],
#         )

#         if "condition" in point.keys():
#             point_dict["condition"] = point.condition
#         if "context" in point.keys():
#             point_dict["context"] = point.context

#         if self.traceable:
#             point_dict["pooling_inverse"] = cluster
#             point_dict["pooling_parent"] = point
#         point = Point(point_dict)
#         if self.norm is not None:
#             point = self.norm(point)
#         if self.act is not None:
#             point = self.act(point)
#         point.sparsify()
#         return point


# class SerializedUnpooling(PointModule):
#     def __init__(
#         self,
#         in_channels,
#         skip_channels,
#         out_channels,
#         norm_layer=None,
#         act_layer=None,
#         traceable=False,  # record parent and cluster
#     ):
#         super().__init__()
#         self.proj = PointSequential(nn.Linear(in_channels, out_channels))
#         self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

#         if norm_layer is not None:
#             self.proj.add(norm_layer(out_channels))
#             self.proj_skip.add(norm_layer(out_channels))

#         if act_layer is not None:
#             self.proj.add(act_layer())
#             self.proj_skip.add(act_layer())

#         self.traceable = traceable

#     def forward(self, point):
#         assert "pooling_parent" in point.keys()
#         assert "pooling_inverse" in point.keys()
#         parent = point.pop("pooling_parent")
#         inverse = point.pop("pooling_inverse")
#         point = self.proj(point)
#         parent = self.proj_skip(parent)
#         parent.feat = parent.feat + point.feat[inverse]

#         if self.traceable:
#             parent["unpooling_parent"] = point
#         return parent


# class Embedding(PointModule):
#     def __init__(
#         self,
#         in_channels,
#         embed_channels,
#         norm_layer=None,
#         act_layer=None,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.embed_channels = embed_channels

#         # TODO: check remove spconv
#         self.stem = PointSequential(
#             conv=spconv.SubMConv3d(
#                 in_channels,
#                 embed_channels,
#                 kernel_size=5,
#                 padding=1,
#                 bias=False,
#                 indice_key="stem",
#             )
#         )
#         if norm_layer is not None:
#             self.stem.add(norm_layer(embed_channels), name="norm")
#         if act_layer is not None:
#             self.stem.add(act_layer(), name="act")

#     def forward(self, point: Point):
#         point = self.stem(point)
#         return point


# @MODELS.register_module("PT-v3m1")
# class PointTransformerV3(PointModule):
#     def __init__(
#         self,
#         in_channels=6,
#         order=("z", "z-trans"),
#         stride=(2, 2, 2, 2),
#         enc_depths=(2, 2, 2, 6, 2),
#         enc_channels=(32, 64, 128, 256, 512),
#         enc_num_head=(2, 4, 8, 16, 32),
#         enc_patch_size=(48, 48, 48, 48, 48),
#         dec_depths=(2, 2, 2, 2),
#         dec_channels=(64, 64, 128, 256),
#         dec_num_head=(4, 4, 8, 16),
#         dec_patch_size=(48, 48, 48, 48),
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         drop_path=0.3,
#         pre_norm=True,
#         shuffle_orders=True,
#         enable_rpe=False,
#         enable_flash=True,
#         upcast_attention=False,
#         upcast_softmax=False,
#         cls_mode=False,
#         pdnorm_bn=False,
#         pdnorm_ln=False,
#         pdnorm_decouple=True,
#         pdnorm_adaptive=False,
#         pdnorm_affine=True,
#         pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
#     ):
#         super().__init__()
#         self.num_stages = len(enc_depths)
#         self.order = [order] if isinstance(order, str) else order
#         self.cls_mode = cls_mode
#         self.shuffle_orders = shuffle_orders

#         assert self.num_stages == len(stride) + 1
#         assert self.num_stages == len(enc_depths)
#         assert self.num_stages == len(enc_channels)
#         assert self.num_stages == len(enc_num_head)
#         assert self.num_stages == len(enc_patch_size)
#         assert self.cls_mode or self.num_stages == len(dec_depths) + 1
#         assert self.cls_mode or self.num_stages == len(dec_channels) + 1
#         assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
#         assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

#         # norm layers
#         if pdnorm_bn:
#             bn_layer = partial(
#                 PDNorm,
#                 norm_layer=partial(
#                     nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
#                 ),
#                 conditions=pdnorm_conditions,
#                 decouple=pdnorm_decouple,
#                 adaptive=pdnorm_adaptive,
#             )
#         else:
#             bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
#         if pdnorm_ln:
#             ln_layer = partial(
#                 PDNorm,
#                 norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
#                 conditions=pdnorm_conditions,
#                 decouple=pdnorm_decouple,
#                 adaptive=pdnorm_adaptive,
#             )
#         else:
#             ln_layer = nn.LayerNorm
#         # activation layers
#         act_layer = nn.GELU

#         self.embedding = Embedding(
#             in_channels=in_channels,
#             embed_channels=enc_channels[0],
#             norm_layer=bn_layer,
#             act_layer=act_layer,
#         )

#         # encoder
#         enc_drop_path = [
#             x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
#         ]
#         self.enc = PointSequential()
#         for s in range(self.num_stages):
#             enc_drop_path_ = enc_drop_path[
#                 sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
#             ]
#             enc = PointSequential()
#             if s > 0:
#                 enc.add(
#                     SerializedPooling(
#                         in_channels=enc_channels[s - 1],
#                         out_channels=enc_channels[s],
#                         stride=stride[s - 1],
#                         norm_layer=bn_layer,
#                         act_layer=act_layer,
#                     ),
#                     name="down",
#                 )
#             for i in range(enc_depths[s]):
#                 enc.add(
#                     Block(
#                         channels=enc_channels[s],
#                         num_heads=enc_num_head[s],
#                         patch_size=enc_patch_size[s],
#                         mlp_ratio=mlp_ratio,
#                         qkv_bias=qkv_bias,
#                         qk_scale=qk_scale,
#                         attn_drop=attn_drop,
#                         proj_drop=proj_drop,
#                         drop_path=enc_drop_path_[i],
#                         norm_layer=ln_layer,
#                         act_layer=act_layer,
#                         pre_norm=pre_norm,
#                         order_index=i % len(self.order),
#                         cpe_indice_key=f"stage{s}",
#                         enable_rpe=enable_rpe,
#                         enable_flash=enable_flash,
#                         upcast_attention=upcast_attention,
#                         upcast_softmax=upcast_softmax,
#                     ),
#                     name=f"block{i}",
#                 )
#             if len(enc) != 0:
#                 self.enc.add(module=enc, name=f"enc{s}")

#         # decoder
#         if not self.cls_mode:
#             dec_drop_path = [
#                 x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
#             ]
#             self.dec = PointSequential()
#             dec_channels = list(dec_channels) + [enc_channels[-1]]
#             for s in reversed(range(self.num_stages - 1)):
#                 dec_drop_path_ = dec_drop_path[
#                     sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
#                 ]
#                 dec_drop_path_.reverse()
#                 dec = PointSequential()
#                 dec.add(
#                     SerializedUnpooling(
#                         in_channels=dec_channels[s + 1],
#                         skip_channels=enc_channels[s],
#                         out_channels=dec_channels[s],
#                         norm_layer=bn_layer,
#                         act_layer=act_layer,
#                     ),
#                     name="up",
#                 )
#                 for i in range(dec_depths[s]):
#                     dec.add(
#                         Block(
#                             channels=dec_channels[s],
#                             num_heads=dec_num_head[s],
#                             patch_size=dec_patch_size[s],
#                             mlp_ratio=mlp_ratio,
#                             qkv_bias=qkv_bias,
#                             qk_scale=qk_scale,
#                             attn_drop=attn_drop,
#                             proj_drop=proj_drop,
#                             drop_path=dec_drop_path_[i],
#                             norm_layer=ln_layer,
#                             act_layer=act_layer,
#                             pre_norm=pre_norm,
#                             order_index=i % len(self.order),
#                             cpe_indice_key=f"stage{s}",
#                             enable_rpe=enable_rpe,
#                             enable_flash=enable_flash,
#                             upcast_attention=upcast_attention,
#                             upcast_softmax=upcast_softmax,
#                         ),
#                         name=f"block{i}",
#                     )
#                 self.dec.add(module=dec, name=f"dec{s}")

#     def forward(self, data_dict):
#         point = Point(data_dict)
#         point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
#         point.sparsify()

#         point = self.embedding(point)
#         point = self.enc(point)
#         if not self.cls_mode:
#             point = self.dec(point)
#         # else:
#         #     point.feat = torch_scatter.segment_csr(
#         #         src=point.feat,
#         #         indptr=nn.functional.pad(point.offset, (1, 0)),
#         #         reduce="mean",
#         #     )
#         return point
