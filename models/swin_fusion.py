# SWinIR
from functools import partial
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torchinfo import summary
class SwinIR(nn.Module):
    r""" SwinIR
        基于 Swin Transformer 的图像恢复网络.

    输入:
        img_size (int | tuple(int)): 输入图像的大小，默认为 64*64.
        patch_size (int | tuple(int)): patch 的大小，默认为 1.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): Patch embedding 的维度，默认为 96.
        depths (tuple(int)): Swin Transformer 层的深度.
        num_heads (tuple(int)): 在不同层注意力头的个数.
        window_size (int): 窗口大小，默认为 7.
        mlp_ratio (float): MLP隐藏层特征图通道与嵌入层特征图通道的比，默认为 4.
        qkv_bias (bool): 给 query, key, value 添加可学习的偏置，默认为 True.
        qk_scale (float): 重写默认的缩放因子，默认为 None.
        drop_rate (float): 随机丢弃神经元，丢弃率默认为 0.
        attn_drop_rate (float): 注意力权重的丢弃率，默认为 0.
        drop_path_rate (float): 深度随机丢弃率，默认为 0.1.
        norm_layer (nn.Module): 归一化操作，默认为 nn.LayerNorm.
        ape (bool): patch embedding 添加绝对位置 embedding，默认为 False.
        patch_norm (bool): 在 patch embedding 后添加归一化操作，默认为 True.
        use_checkpoint (bool): 是否使用 checkpointing 来节省显存，默认为 False.
        upscale: 放大因子， 2/3/4/8 适合图像超分, 1 适合图像去噪和 JPEG 压缩去伪影
        img_range: 灰度值范围， 1 或者 255.
        upsampler: 图像重建方法的选择模块，可选择 pixelshuffle, pixelshuffledirect, nearest+conv 或 None.
        resi_connection: 残差连接之前的卷积块， 可选择 1conv 或 3conv.
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans  # 输入图片通道数
        num_out_ch = in_chans  # 输出图片通道数
        num_feat = 64  # 特征图通道数
        self.img_range = img_range  # 灰度值范围:[0, 1] or [0, 255]
        if in_chans == 3:  # 如果输入是RGB图像
            rgb_mean = (0.4488, 0.4371, 0.4040)  # 数据集RGB均值
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)  # 转为[1, 3, 1, 1]的张量
        else:  # 否则灰度图
            self.mean = torch.zeros(1, 1, 1, 1)  # 构造[1, 1, 1, 1]的张量
        self.upscale = upscale  # 图像放大倍数，超分(2/3/4/8),去噪(1)
        self.upsampler = upsampler  # 上采样方法
        self.window_size = window_size  # 注意力窗口的大小

        #######################################################################################
        ################################### 1, 浅层特征提取 ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)  # 输入卷积层

        ##########################################################################################
        ################################### 2, 深层特征提取 ######################################
        self.num_layers = len(depths)  # Swin Transformer 层的个数
        self.embed_dim = embed_dim  # 嵌入层特征图的通道数
        self.ape = ape  # patch embedding 添加绝对位置 embedding，默认为 False.
        self.patch_norm = patch_norm  # 在 patch embedding 后添加归一化操作，默认为 True.
        self.num_features = embed_dim  # 特征图的通道数
        self.mlp_ratio = mlp_ratio  # MLP隐藏层特征图通道与嵌入层特征图通道的比

        # 将图像分割成多个不重叠的patch
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches  # 分割得到patch的个数
        patches_resolution = self.patch_embed.patches_resolution  # 分割得到patch的分辨率
        self.patches_resolution = patches_resolution

        # 将多个不重叠的patch合并成图像
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # 绝对位置嵌入
        if self.ape:
            # 结构为 [1，patch个数， 嵌入层特征图的通道数] 的参数
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)  # 截断正态分布，限制标准差为0.02

        self.pos_drop = nn.Dropout(p=drop_rate)  # 以drop_rate为丢弃率随机丢弃神经元，默认不丢弃

        # 随机深度衰减规律，默认为 [0, 0.1] 进行24等分后的列表
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Residual Swin Transformer blocks (RSTB)
        # 残差 Swin Transformer 块 (RSTB)
        self.layers = nn.ModuleList()  # 创建一个ModuleList实例对象，也就是多个 RSTB
        for i_layer in range(self.num_layers):  # 循环 Swin Transformer 层的个数次
            # 实例化 RSTB
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)  # 将 RSTB 对象插入 ModuleList 中
        self.norm = norm_layer(self.num_features)  # 归一化操作，默认 LayerNorm

        # 在深层特征提取网络中加入卷积块，保持特征图通道数不变
        if resi_connection == '1conv':  # 1层卷积
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':  # 3层卷积
            # 为了减少参数使用和节约显存，采用瓶颈结构
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),  # 降维
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))  # 升维

        # 高质量图像重建模块
        if self.upsampler == 'pixelshuffle':  # pixelshuffle 上采样
            # 适合经典超分
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)  # 上采样
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)  # 输出卷积层
        elif self.upsampler == 'pixelshuffledirect':  # 一步是实现既上采样也降维
            # 适合轻量级充分，可以减少参数量(一步是实现既上采样也降维)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':  # 最近邻插值上采样
            # 适合真实图像超分
            assert self.upscale == 4, 'only support x4 now.'  # 声明目前仅支持4倍超分重建
            # 上采样之前的卷积层
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            # 第一次上采样卷积(直接对输入做最近邻插值变为2倍图像)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            # 第二次上采样卷积(直接对输入做最近邻插值变为2倍图像)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 对上采样完成的图像再做卷积
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)  # 激活层
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)  # 输出卷积层
        else:
            # 适合图像去噪和 JPEG 压缩去伪影
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)  # 初始化网络参数

    # 初始化网络参数
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):  # 判断是否为线性 Linear 层
            trunc_normal_(m.weight, std=.02)  # 截断正态分布，限制标准差为 0.02
            if m.bias is not None:  # 如果设置了偏置
                nn.init.constant_(m.bias, 0)  # 初始化偏置为 0
        elif isinstance(m, nn.LayerNorm):  # 判断是否为归一化 LayerNorm 层
            nn.init.constant_(m.bias, 0)  # 初始化偏置为 0
            nn.init.constant_(m.weight, 1.0)  # 初始化权重系数为 1

    # 检查图片(准确说是张量)的大小
    def check_image_size(self, x):
        _, _, h, w = x.size()  # 张量 x 的高和宽
        # h 维度要填充的个数
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        # w 维度要填充的个数
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # 右填充 mod_pad_w 个值，下填充 mod_pad_h 个值，模式为反射(可以理解为以 x 的维度末尾为轴对折)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    # 深层特征提取网络的前向传播
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])  # 张量 x 的高和宽
        x = self.patch_embed(x)  # 分割 x 为多个不重叠的 patch embeddings
        if self.ape:  # 绝对位置 embedding
            x = x + self.absolute_pos_embed  # x 加上对应的绝对位置 embedding
        x = self.pos_drop(x)  # 随机将x中的部分元素置 0

        for layer in self.layers:
            x = layer(x, x_size)  # x 通过多个串联的 RSTB

        x = self.norm(x)  # 对 RSTB 的输出进行归一化
        x = self.patch_unembed(x, x_size)  # 将多个不重叠的 patch 合并成图像

        return x

    # SWinIR 的前向传播
    def forward(self, x):
        H, W = x.shape[2:]  # 输入图片的高和宽
        x = self.check_image_size(x)  # 检查图片的大小，使高宽满足 window_size 的整数倍

        self.mean = self.mean.type_as(x)  # RGB 均值的类型同 x 一致
        x = (x - self.mean) * self.img_range  # x 减去 RGB 均值再乘以输入的最大灰度值

        if self.upsampler == 'pixelshuffle':  # pixelshuffle 上采样方法
            # 适合经典超分
            x = self.conv_first(x)  # 输入卷积层
            x = self.conv_after_body(self.forward_features(x)) + x  # 深度特征提取网络，引入残差
            x = self.conv_before_upsample(x)  # 上采样前进行卷积
            x = self.conv_last(self.upsample(x))  # 上采样后再通过输出卷积层
        elif self.upsampler == 'pixelshuffledirect':  # 一步是实现既上采样也降维
            # 适合轻量级超分
            x = self.conv_first(x)  # 输入卷积层
            x = self.conv_after_body(self.forward_features(x)) + x  # 深度特征提取网络，引入残差
            x = self.upsample(x)  # 上采样并降维后输出
        elif self.upsampler == 'nearest+conv':  # 最近邻插值上采样方法
            # 适合真实图像超分，只适合 4 倍超分
            x = self.conv_first(x)  # 输入卷积层
            x = self.conv_after_body(self.forward_features(x)) + x  # 深度特征提取网络，引入残差
            x = self.conv_before_upsample(x)  # 上采样前进行卷积
            # 第一次上采样 2 倍
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            # 第二次上采样 2 倍
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))  # 输出卷积层
        else:
            # 适合图像去噪和 JPEG 压缩去伪影
            x_first = self.conv_first(x)  # 输入卷积层
            res = self.conv_after_body(self.forward_features(x_first)) + x_first  # 深度特征提取网络，引入残差
            x = x + self.conv_last(res)  # 输出卷积层，引入残差

        x = x / self.img_range + self.mean  # 最后的 x 除以灰度值范围再加上 RGB 均值

        return x[:, :, :H * self.upscale, :W * self.upscale]  # 返回输出 x
