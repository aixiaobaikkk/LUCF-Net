from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple



def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


# 定义一个多层感知器神经网络类
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一个全连接层
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二个全连接层
        self.drop = nn.Dropout(drop)  # Dropout层

    def forward(self, x):
        x = self.fc1(x)  # 输入经过第一个全连接层
        x = self.act(x)  # 激活函数
        x = self.drop(x)  # Dropout层
        x = self.fc2(x)  # 经过第二个全连接层
        x = self.drop(x)  # Dropout层
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 定义一个卷积多层感知器神经网络类
class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)  # 输入卷积层
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)  # 输出卷积层
        self.drop = nn.Dropout(drop)  # Dropout层

    def forward(self, x):
        x = self.fc1(x)  # 输入经过第一个卷积层
        x = self.act(x)  # 激活函数
        x = self.drop(x)  # Dropout层
        x = self.fc2(x)  # 经过第二个卷积层
        x = self.drop(x)  # Dropout层
        return x


class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.upsample = nn.Upsample(scale_factor=sr_ratio, mode='nearest')
        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
            kernel_size = sr_ratio
            self.LocalProp = nn.ConvTranspose2d(dim, dim, kernel_size, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        if self.sr > 1.:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(H / self.sr), int(W / self.sr))
            x = self.LocalProp(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalAgg(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # global layer_scale
        # self.ls = layer_scale

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)



        return x


class LG_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()

        if sr_ratio > 1:
            self.LocalAgg = LocalAgg(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
                                     act_layer, norm_layer)
        else:
            self.LocalAgg = nn.Identity()

        self.SelfAttn = SelfAttn(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer,
                                 norm_layer, sr_ratio)

    def forward(self, x):
        x = self.LocalAgg(x)
        x = self.SelfAttn(x)
        return x


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        self.lg_en1 = LG_Block(dim=self.ft_chns[1], num_heads=3, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=self.dropout[1], sr_ratio=4, drop_path=0.1)
        self.lg_en2 = LG_Block(dim=self.ft_chns[2], num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=self.dropout[2], sr_ratio=2, drop_path=0.1)
        self.lg_en3 = LG_Block(dim=self.ft_chns[3], num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=self.dropout[3], sr_ratio=2, drop_path=0.1)
        self.lg_en4 = LG_Block(dim=self.ft_chns[4], num_heads=24, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=self.dropout[4], sr_ratio=1, drop_path=0.1)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x1 = self.lg_en1(x1)
        x2 = self.down2(x1)
        x2 = self.lg_en2(x2)
        x3 = self.down3(x2)
        x3 = self.lg_en3(x3)
        x4 = self.down4(x3)
        x4 = self.lg_en4(x4)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)



    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        p = []
        p.append(x)
        x = self.up2(x, x2)
        p.append(x)
        x = self.up3(x, x1)
        p.append(x)
        x = self.up4(x, x0)
        p.append((x))

        return p




class CIE_Head(nn.Module):
    def __init__(self,params):
        super(CIE_Head, self).__init__()
        self.params = params
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.out_head1 = nn.Conv2d(self.ft_chns[3],  self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.ft_chns[2],  self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.ft_chns[1],  self.n_class, 1)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
    def forward(self, feature):

        p = []
        p.append(F.interpolate(self.out_head1(feature[0]), scale_factor=8, mode='bilinear'))
        p.append(F.interpolate(self.out_head2(feature[1]), scale_factor=4, mode='bilinear'))
        p.append(F.interpolate(self.out_head3(feature[2]), scale_factor=2, mode='bilinear'))
        p.append(F.interpolate(self.out_conv(feature[3]), scale_factor=1, mode='bilinear'))

        return p




class LUCF_Net(nn.Module):
    def __init__(self, in_chns, class_num):
        super(LUCF_Net, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 48, 96,192, 384],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.cie = CIE_Head(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        output = self.cie(output)

        return output



