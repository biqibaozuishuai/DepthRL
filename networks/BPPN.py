import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .swin_transformer import SwinTransformer
########################################################################################################################

def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

class conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, norm,GN_group):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=1, stride=1,
                              padding=0)
        if norm == 'BN':
            self.norm = nn.BatchNorm2d(out_channels, momentum=0.01, affine=True, eps=1.1e-5)
        elif norm == 'GN':
            self.norm = nn.GroupNorm(num_groups = GN_group, num_channels=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out
    
class conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, norm = 'GN', GN_group = 32):
        super(conv3x3, self).__init__()
        self.conv   = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                              padding=1)
        if norm == 'BN':
            self.norm     = nn.BatchNorm2d(out_channels, momentum=0.01, affine=True, eps=1.1e-5)
        elif norm == 'GN':
            self.norm = nn.GroupNorm(num_groups = GN_group, num_channels=out_channels)
        self.relu   = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class myConv(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride=1,
                 padding=0, dilation=1, bias=True, norm='GN', act='ELU', num_groups=32):
        super(myConv, self).__init__()
        if act == 'ELU':
            act = nn.ELU()
        # elif act == 'Mish':
        #     act = Mish()
        else:
            act = nn.ReLU(True)
        module = []
        module.append(nn.Conv2d(in_channels = in_ch,out_channels=out_ch, kernel_size=kSize, stride=stride,
                           padding=padding, dilation=dilation, groups=1, bias=bias))
        # decide use GN or BN
        if norm == 'GN':
            module.append(nn.GroupNorm(
                num_groups=num_groups, num_channels=in_ch))
        else:
            module.append(nn.BatchNorm2d(out_ch, eps=0.001,
                          momentum=0.1, affine=True, track_running_stats=True))
        module.append(act)

        self.module = nn.Sequential(*module)

    def forward(self, x):
        out = self.module(x)
        return out

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(2*dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio) # sr_ratios=[8, 4, 2, 1]
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x,y, H, W):
        B, N, C = x.shape
        q  = self.q(x).reshape(B, N, self.num_heads, C //self.num_heads).permute(0,2,1,3)  # (0,2,1,3)
        k1 = self.k1(x)
        if self.sr_ratio > 1:
            y_ = y.permute(0, 2, 1).reshape(B, C, H, W)
            y_ = self.sr(y_).reshape(B, C, -1).permute(0, 2, 1)
            y_ = self.norm(y_)
            # kv = self.kv(y_).reshape(B, -1, 2, self.num_heads,C // self.num_heads).permute(2, 0, 3, 1, 4)
            k2 = self.k2(y)
            v  = self.v(y).reshape(B, -1, self.num_heads, C //self.num_heads).permute(0,2,1,3)
        else:
            k2 = self.k2(y)
            v  = self.v(y).reshape(B, -1, self.num_heads, C //self.num_heads).permute(0,2,1,3)
            
        k = torch.cat([k1,k2], dim=-1) # B,N,2C
        k = self.k(k).reshape(B, -1, self.num_heads, C //self.num_heads).permute(0,2,1,3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale # transpose交换矩阵的两个维度
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

"""Multiscale Cross Attention Feature Enhancement Module"""
class MCFEM(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.005, act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,out_channel=64):
        super().__init__()

        self.norm1  = norm_layer(dim)
        self.norm2  = norm_layer(dim)
        self.norm3  = norm_layer(dim)
        self.norm4  = norm_layer(dim)
        self.norm5  = norm_layer(dim)
        self.norm6  = norm_layer(dim)
        self.attn1  = Attention(dim,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.attn2  = Attention(dim,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1  = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2  = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim   = int(dim * mlp_ratio)
        
        self.mlp1        = Mlp(in_features=dim, out_features = dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop)
        self.mlp2        = Mlp(in_features=2*dim, out_features = dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop)
        self.mlp3        = Mlp(in_features=dim, out_features = dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop)
        
        #self.chn_down   = nn.Sequential(conv3x3(3*out_channel, 2*out_channel),conv3x3(2*out_channel, out_channel))
        self.reduce = nn.Linear(2*dim, dim, bias=qkv_bias)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x,y,z):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).view(B,-1,C) # B,H,W,C --> B,N,C
        y = y.permute(0,2,3,1).view(B,-1,C) # 
        z1 = z.permute(0,2,3,1).view(B,-1,C) # 
        
        feat_yz = self.attn1(y,z1, H, W) 
        self.drop_path1(feat_yz)
        feat_yz = self.mlp1(feat_yz, H, W)
        
        yz = torch.cat([y, feat_yz],dim=-1) # 128+128=256
        yz = self.reduce(yz)
        feat_xy = self.attn2(x,yz, H, W)
        self.drop_path2(feat_xy)
        feat_xy = self.mlp3(feat_xy, H, W)
        
        x = x.permute(0,2,1).view(B,C,H,W) # B,H,W,C
        y = y.permute(0,2,1).view(B,C,H,W) # B,H,W,C
        feat_xy = feat_xy.permute(0,2,1).view(B,C,H,W) # B,H,W,C
        feat_yz = feat_yz.permute(0,2,1).view(B,C,H,W) # B,H,W,C
        #z = z.permute(0,2,1).view(B,C,H,W) # B,H,W,C
        out = torch.cat([x,feat_xy,y,feat_yz],dim=1) # out_channel: 64*4
        return out


class ConvPatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(dim) if postnorm else norm_layer(dim)

    def forward(self, x):
        B, C,H,W = x.shape
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        if self.postnorm:
            # x = x.permute(0, 3, 1, 2)  # B C H W
            x = self.reduction(x)
            #x = x.flatten(2)  # B C H//2*W//2 
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.reduction(x)
            #x = x.flatten(2)  # B C H//2*W//2
        #x = x.view(B, C, H, W) # B,C,H,W
        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, postnorm=True):
        super().__init__()
        self.dim = dim
        self.postnorm = postnorm

        self.reduction = nn.Linear(4 * dim, 2*dim, bias=False) # 4*64-->64
        self.norm = norm_layer(2*dim) if postnorm else norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, C, H, W = x.shape
        # x = x.permute(0,2,3,1) # B,H,W,C
        # x = x.view(B, H, W, C)
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2 
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2
        x = x.permute(0,2,3,1) # B,H,W,C
        x = x.view(B, -1, 4 * C)  # B, H/2*W/2, 4*C 

        if self.postnorm:
            x = self.reduction(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.reduction(x)
        x = x.permute(0,2,1) # B,C,-1
        x = x.view(B, 2* C, H//2, W//2) # B,C,H,W
        return x

""" Symmetric Gating Attention Fusion Module """
class SGAFM(nn.Module):
    def __init__(self,norm, act, in_chn):
        super(SGAFM, self).__init__()
        self.conv1 = nn.Sequential(conv1x1(2*in_chn, in_chn,'GN',16), nn.Conv2d(in_chn, in_chn,3,1,1))
        self.conv2 = nn.Sequential(conv1x1(2*in_chn, in_chn,'GN',16), nn.Conv2d(in_chn, in_chn,3,1,1))
        
        self.ConvBlock1 = nn.Sequential(conv1x1(2*in_chn, in_chn,'GN',16))
        

    def forward(self, input1,input2 ):
        x  = torch.cat([input1, input2], dim=1)  # 64+64-->128
        
        gate1 = torch.sigmoid(self.conv1(x)) # 64
        gate2 = torch.sigmoid(self.conv2(x)) # 64
        
        feat1 = input1 + input1 *gate1  
        feat2 = input2 + input2 *gate2
        # feat2 = torch.cat([input1*gate2, input2], dim=1)  # 64+64-->128
        feat3 = torch.cat([feat1, feat2], dim=1) # 64+64-->128
        out = self.ConvBlock1(feat3) # 2*C --> C
        return out


class Pyramid_Decoder(nn.Module):
    def __init__(self,in_channels):
        super(Pyramid_Decoder,self).__init__()
        # tiny = [96, 192, 384, 768]
        in_ch1, in_ch2, in_ch3, in_ch4 = in_channels[0],in_channels[1],in_channels[2],in_channels[3]
        self.downChn1 = conv1x1(in_ch1, 64,'GN',16)
        self.downChn2 = conv1x1(in_ch2, 128,'GN',16)
        self.downChn3 = conv1x1(in_ch3, 320,'GN',32)
        self.downChn4 = conv1x1(in_ch4, 512,'GN',32)
        
        self.ConvPM = ConvPatchMerging(dim=64, norm_layer=nn.LayerNorm, postnorm=True)
        
        self.conv1_2_1 = nn.Sequential(conv3x3(128, 64,'GN',64//4))
        self.conv1_2_2 = nn.Sequential(PatchMerging(dim=64, norm_layer=nn.LayerNorm, postnorm=True), conv3x3(128, 128,'GN',128//8))
        self.conv1_3_1 = nn.Sequential(conv3x3(64*3, 64*2,'GN',128//8), conv3x3(64*2, 64,'GN',64//4))
        self.conv1_3_2 = nn.Sequential(PatchMerging(dim=64, norm_layer=nn.LayerNorm, postnorm=True), conv3x3(128, 128,'GN',128//8))

        self.conv2_2   = nn.Sequential(conv3x3(128*4, 128*2,'GN',256//32), conv3x3(128*2, 128,'GN',128//8))
        self.conv2_2_1 = nn.Sequential(nn.PixelShuffle(2),conv3x3(32, 64,'GN',64//4))
        self.conv2_2_2 = nn.Sequential(conv3x3(256, 128,'GN',128//8))
        self.conv2_2_3 = nn.Sequential(PatchMerging(dim=128, norm_layer=nn.LayerNorm, postnorm=True), conv3x3(256, 256,'GN',256//16))
        self.conv2_1_1 = nn.Sequential(nn.PixelShuffle(2), conv3x3(32, 64,'GN',64//4))
        self.conv2_1_1_1 = nn.Sequential(nn.PixelShuffle(2),conv3x3(32, 32,'GN',32//4))
        self.conv2_1_2 = nn.Sequential(conv3x3(128, 64,'GN',64//4))
        self.conv2_3   = nn.Sequential(conv3x3(128*4, 128*2,'GN',256//16), conv3x3(128*2, 128,'GN',128//8))
        self.conv2_3_1 = nn.Sequential(nn.PixelShuffle(2),conv3x3(32, 64, 'GN',64//4))
        self.conv3_1_1 = nn.Sequential(nn.PixelShuffle(2),conv3x3(80, 128,'GN',128//8))
        self.conv3_1_2 = nn.Sequential(conv3x3(320, 256,'GN',256//16))
        self.conv3_2   = nn.Sequential(conv3x3(256*4, 256*2,'GN', 512//32), conv3x3(256*2, 256,'GN',256//16))
        # self.conv3_2_1 = nn.Sequential(nn.PixelShuffle(2), conv3x3(48, 32),conv3x3(32, 16))
        self.conv3_2_1 = nn.Sequential(nn.PixelShuffle(2), conv3x3(64, 128,'GN',128//8)) 
        self.conv4_1_1 = nn.Sequential(nn.PixelShuffle(2),conv3x3(128, 256,'GN',256//16))

        self.fb1 = MCFEM(dim=128, out_channel=128, attn_drop=0.01)
        self.fb2 = MCFEM(dim=256, out_channel=256, attn_drop=0.01)
        self.fb3 = MCFEM(dim=128, out_channel=128, attn_drop=0.01, num_heads=4)
        
        self.frm1 = SGAFM('BN', 'ReLU',64)
        self.frm2 = SGAFM('BN', 'ReLU',64)
        self.frm3 = SGAFM('BN', 'ReLU',64)
        
        self.block3_2_1 = nn.Sequential(nn.PixelShuffle(2),conv3x3(256, 128,'GN',128//8)) #,lightDASP('GN', 'ReLU', 128,32//4,128//8)
        self.block3_2_2 = nn.Sequential(conv3x3(128+3, 64,'GN',64//4),conv3x3(64, 32 ,'GN',32//4),conv3x3(32, 16,'GN',16//4),conv3x3(16, 8,'GN', 8//8),nn.Conv2d(8, 1,3,1,1))
        #self.convUp3    = nn.Sequential(conv3x3(64, 128,'GN',128//16))
        
        self.block2_3_1 = nn.Sequential(nn.PixelShuffle(2), conv3x3(192, 128,'GN',128//8), conv3x3(128, 64,'GN',64//4)) # ,lightDASP('BN', 'ReLU', 64)
        self.block2_3_2 = nn.Sequential(conv3x3(64+1+3, 32,'GN',32//4),conv3x3(32, 16,'GN',16//4),conv3x3(16, 8,'GN',8//4),conv3x3(8, 4,'GN',4//4),nn.Conv2d(4, 1,3,1,1))
        
        self.block1_4_1 = nn.Sequential(nn.PixelShuffle(2), conv3x3(80, 64,'GN',64//4), conv3x3(64, 32,'GN',32//4)) # ,lightDASP('GN', 'ReLU', 32,8//4,32//4)
        self.block1_4_2 = nn.Sequential(conv3x3(32+1+3, 16,'GN', 16//4), conv3x3(16, 8,'GN', 8//4), conv3x3(8, 4,'GN', 4//4), nn.Conv2d(4, 1,3,1,1))
        
        self.convUp1    = nn.Sequential(nn.PixelShuffle(2) ) # ,lightDASP('GN', 'ReLU', 8,2//2,8//4)
        self.last       = nn.Sequential(conv3x3(8+1+3,8,'GN',8//4), conv3x3(8, 4,'GN',4//4), nn.Conv2d(4, 1,3,1,1))
        
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x_1, x_2, x_3, x_4,lap_list):
        
        rgb_lv4, rgb_lv3, rgb_lv2, rgb_lv1 = lap_list[0], lap_list[1], lap_list[2], lap_list[3]
        
        """
        x    :torch.Size([2, 3, 352, 704]), kitti dataset
        conv1:torch.Size([2, 64, 88, 176]), 1/4
        conv2:torch.Size([2, 128, 44, 88]), 1/8 
        conv3:torch.Size([2, 320, 22, 44]), 1/16
        conv4:torch.Size([2, 512, 11, 22]), 1/32
        """

        x_1 = self.downChn1(x_1)
        x_2 = self.downChn2(x_2)
        x_3 = self.downChn3(x_3)
        x_4 = self.downChn4(x_4)
        # 计算feature1_2
        feat2_1_1 = self.conv2_1_1(x_2) # 96 H/4 W/4
        feat1_2   = self.frm1(x_1, feat2_1_1)
        
        # 计算feature2_2
        feat3_1_1 = self.conv3_1_1(x_3)         
        feat1_2_2 = self.conv1_2_2(feat1_2)     
        feat2_1_2 = x_2
        feat2_2_t = self.fb1(feat3_1_1,feat2_1_2,feat1_2_2) # 128*4
        feat2_2   = self.conv2_2(feat2_2_t) # 128
        
        # 计算 feature3_2
        feat4_1_1 = self.conv4_1_1(x_4)     # size*2 channel-->128
        feat3_1_2 = self.conv3_1_2(x_3)     # 
        feat2_2_3 = self.conv2_2_3(feat2_2) # 
        feat3_2_t = self.fb2(feat4_1_1,feat3_1_2,feat2_2_3) # 256*4=192，H/8, W/8
        feat3_2   = self.conv3_2(feat3_2_t) # 256 W/8, H/8
        
        #计算 feature1_3
        feat2_2_1 = self.conv2_2_1(feat2_2) # 64
        feat1_2_1 = self.conv1_2_1(torch.cat([feat1_2, x_1],dim=1))
        feat1_3   = self.frm2(feat1_2_1, feat2_2_1)
        # 计算 feature2_3
        feat3_2_1 = self.conv3_2_1(feat3_2) # 128, H/16, W/16
        feat2_2_2 = self.conv2_2_2(torch.cat([feat2_2,x_2],dim=1)) # 32
        feat1_3_2 = self.conv1_3_2(feat1_3)
        feat2_3_t   = self.fb3(feat3_2_1, feat2_2_2, feat1_3_2) # 128*4
        feat2_3   = self.conv2_3(feat2_3_t) # 128 W/8, H/8
        # calculate feature1_4

        feat2_3_1 = self.conv2_3_1(feat2_3)
        feat1_3_1 = self.conv1_3_1(torch.cat([feat1_3,feat1_2,x_1],dim=1)) # 64*3 --> 64
        feat1_4   = self.frm3(feat1_3_1,feat2_3_1) # 8, 16
        
        # generate att3
        att3_2_up = self.block3_2_1(feat3_2_t)  # channel 256*4-->128
        att3      = self.block3_2_2(torch.cat([att3_2_up,rgb_lv4],dim=1)) # 1/8 
        att3_up   = F.interpolate(att3, scale_factor=2, mode='bilinear', align_corners=True)  # 1/4
        # generate att2

        att2_3_up = self.block2_3_1(torch.cat([x_2,feat2_3_t, att3_2_up],dim=1)) # 1/4 ,64
        att2      = self.block2_3_2(torch.cat([att2_3_up,att3_up,rgb_lv3],dim=1))
        att2_up   = F.interpolate(att2, scale_factor=2, mode='bilinear', align_corners=True)  # 1/2
        # generate att1
        att1_4    = self.block1_4_1(torch.cat([x_1,feat1_2,feat1_3,feat1_4, att2_3_up],dim=1)) # 1/2, 64*5-->32，只有一个pixel shuffle
        att1      = self.block1_4_2(torch.cat([att1_4,att2_up,rgb_lv2],dim=1))          # 1/2
        att1_up   = F.interpolate(att1, scale_factor=2, mode='bilinear', align_corners=True)  # 1/2
        # generate att
        att1_4_up = self.convUp1(att1_4)         # 32-->64
        att       = self.last(torch.cat([att1_4_up,att1_up,rgb_lv1],dim=1)) # 1/1,chn:64+64-->32
        
        return att,att1_up,att2_up,att3_up



"""The Bi-directional Pyramid Policy Network"""
class BPPN(nn.Module):

    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()
        self.max_depth = max_depth
        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False

        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

        window_size = int(version[-2:])

        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.backbone = SwinTransformer(**backbone_cfg)

        self.init_weights(pretrained=pretrained)
        self.decoder = Pyramid_Decoder(in_channels)
        
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)

        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def forward(self, imgs):
        """ [128, 256, 512, 1024] 128*88*280, 256*144*140, 512*22*70, 1024*11*35   """
        enc_feats   = self.backbone(imgs) 
        # out, att4, att3, att2 = self.decoder(enc_feats[0],enc_feats[1],enc_feats[2],enc_feats[3])
        

        rgb_down2   = F.interpolate(imgs, scale_factor=0.5, mode='bilinear', align_corners=True)
        rgb_down4   = F.interpolate(rgb_down2, scale_factor=0.5, mode='bilinear', align_corners=True)
        rgb_down8   = F.interpolate(rgb_down4, scale_factor=0.5, mode='bilinear', align_corners=True)
        rgb_down16  = F.interpolate(rgb_down8, scale_factor=0.5, mode='bilinear', align_corners=True)
        rgb_down32  = F.interpolate(rgb_down16, scale_factor=0.5, mode='bilinear', align_corners=True)

        rgb_up16    = F.interpolate(rgb_down32, scale_factor=2, mode='bilinear', align_corners=True)
        rgb_up8     = F.interpolate(rgb_down16, scale_factor=2, mode='bilinear', align_corners=True)
        rgb_up4     = F.interpolate(rgb_down8, scale_factor=2, mode='bilinear', align_corners=True)
        rgb_up2     = F.interpolate(rgb_down4, scale_factor=2, mode='bilinear', align_corners=True)
        rgb_up      = F.interpolate(rgb_down2, scale_factor=2, mode='bilinear', align_corners=True)
        
        lap1        = imgs - rgb_up
        lap2        = rgb_down2 - rgb_up2
        lap3        = rgb_down4 - rgb_up4
        lap4        = rgb_down8 - rgb_up8
        lap5        = rgb_down16 - rgb_up16
        lap_list    = [ lap4, lap3, lap2, lap1]

        att,att1_up,att2_up,att3_up = self.decoder(enc_feats[0],enc_feats[1],enc_feats[2],enc_feats[3],lap_list)
        # att5, att4, att3, att2, att_out = self.decoder(enc_feats[0],enc_feats[1],enc_feats[2],enc_feats[3],lap_list) # c=256
        
        depth3      = att3_up # 1/4
        depth2      = att2_up+ F.interpolate(depth3, scale_factor=2, mode='bilinear', align_corners=True) # 1/2
        depth1      = att1_up+ F.interpolate(depth2, scale_factor=2, mode='bilinear', align_corners=True)  # 1/1
        att_img     = att + depth1
        out_depth   = torch.sigmoid(att_img) * self.max_depth                                              # 1*352*704
        
        depth3 = F.interpolate(depth3, scale_factor=4, mode='bilinear', align_corners=True)
        depth3 = torch.sigmoid(depth3) * self.max_depth
        depth2 = F.interpolate(depth2, scale_factor=2, mode='bilinear', align_corners=True)
        depth2 = torch.sigmoid(depth2) * self.max_depth
        depth1 = torch.sigmoid(depth1) * self.max_depth

        return out_depth, depth1,depth2,depth3
    
