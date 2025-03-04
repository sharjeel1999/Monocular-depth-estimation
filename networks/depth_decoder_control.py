# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from collections import OrderedDict
from layers import *

class Combine_Feats(nn.Module):
    def __init__(self):
        super(Combine_Feats, self).__init__()
        
        self.context_1 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
                nn.ReLU(inplace=True)
            )
        
        self.context_2 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
                nn.ReLU(inplace=True)
            )
        
        self.comb = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
                nn.ReLU(inplace=True)
            )
        
        self.refine = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, current, prev):
        x1 = self.context_1(current)
        x2 = self.context_2(prev)
        x = torch.cat([x1, x2], dim = 1)
        
        x = self.comb(x)
        x = self.refine(x)
        return x

class Self_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        #print('in: ', x.shape)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        #print('q shape: ', q.shape)
        #print('k shape: ', k.shape)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #print('for scale shape: ', dots.shape)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.self_attention = Self_Attention(dim)

    def forward(self, x, x_prev):
        
        kv = self.to_kv(x_prev).chunk(2, dim = -1)
        q = self.to_q(x)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        sf = lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads)
        q = sf(q)
        # print('v shape: ', v.shape)
        # print('k shape: ', k.shape)
        # print('q shape: ', q.shape)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #print('for scale shape: ', dots.shape)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print('After attention: ', out.shape)
        out = self.to_out(out)
        # out = self.self_attention(out)
        
        return out

class Combine_Attention(nn.Module):
    def __init__(self, image_height, image_width):
        super().__init__()
        
        channels = 128
        dim = 1024
        
        patch_height_1 = 3
        patch_width_1 = 10
        
        num_patches = (image_height // patch_height_1) * (image_width // patch_width_1)
        patch_dim = channels * patch_height_1 * patch_width_1
        patch_dim_embed = patch_height_1 * patch_width_1
        
        self.conv1 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(channels, eps=0.001, momentum=0.01),
                nn.ReLU(inplace=True)
            )
        
        self.conv2 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(channels, eps=0.001, momentum=0.01),
                nn.ReLU(inplace=True)
            )
        
        self.pos_conv = nn.Sequential(
                nn.Conv2d(int(channels*2), channels, kernel_size = 1),
                nn.BatchNorm2d(channels, eps=0.001, momentum=0.01),
                nn.ReLU(),
                
                nn.Conv2d(channels, 2, kernel_size = 3, padding = 1),
                # nn.BatchNorm2d(channels, eps=0.001, momentum=0.01),
                nn.Softmax()
            )
        
        
        self.to_patch_pos_1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height_1, p2 = patch_width_1),
            nn.Linear(patch_dim_embed, dim),
        )
        
        self.to_patch_pos_2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height_1, p2 = patch_width_1),
            nn.Linear(patch_dim_embed, dim),
        )
        
        
        
        self.to_patch_embedding_1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height_1, p2 = patch_width_1),
            nn.Linear(patch_dim, dim),
        )
        
        self.to_patch_embedding_2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height_1, p2 = patch_width_1),
            nn.Linear(patch_dim, dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        
        self.attention = Attention(dim = dim)
        
        self.to_normal = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height_1, p2 = patch_width_1, h = int(image_height/patch_height_1), w = int(image_width/patch_width_1)),
        )
        
        self.final_conv = nn.Sequential(
                nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 1),# padding = 1),
                nn.BatchNorm2d(channels, eps=0.001, momentum=0.01),
                nn.ReLU(inplace=True)
            )
        
        
    def forward(self, x_in, x_prev):
        # print('Original input shapes: ', x_in.shape, x_prev.shape)
        # plt.imshow(x_in[1, 50, :, :].detach().cpu().numpy())
        # plt.show()
        # plt.imshow(x_prev[1, 50, :, :].detach().cpu().numpy())
        # plt.show()
        # sumed_in = torch.sum(x_in, dim = 1)
        # sumed_prev = torch.sum(x_prev, dim = 1)
        # print('Original summed shapes: ', sumed_in.shape, sumed_prev.shape)
        # plt.imshow(sumed_in[1, :, :].detach().cpu().numpy())
        # plt.show()
        # plt.imshow(sumed_prev[1, :, :].detach().cpu().numpy())
        # plt.show()
        
        comb_pos = self.pos_conv(torch.cat((x_in, x_prev), dim = 1))
        pos_in = torch.unsqueeze(comb_pos[:, 0, :, :], dim = 1)
        pos_prev = torch.unsqueeze(comb_pos[:, 1, :, :], dim = 1)
        # print('combined pos shapes: ', pos_in.shape, pos_prev.shape)
        pos_in  = self.to_patch_pos_1(pos_in)
        pos_prev = self.to_patch_pos_2(pos_prev)
        # print('resized positional: ', res.shape)
        
        x = self.conv1(x_in)
        x_prev = self.conv2(x_prev)
        
        # print('--- required before embedding shape: ', x.shape, x_prev.shape)
        x = self.to_patch_embedding_1(x)
        x_prev = self.to_patch_embedding_2(x_prev)
        b, n, _ = x.shape
        # print('--- required after embedding shape: ', x.shape, x_prev.shape)
        
        # print('positional embedding shape : ', self.pos_embedding[:, :(n+1)].shape)
        
        x += self.pos_embedding[:, :(n + 1)]
        x_prev += self.pos_embedding[:, :(n + 1)]
        
        x += pos_in
        x_prev += pos_prev
        
        attention_out = self.attention(x, x_prev)
        # print('Input: ', x_in.shape)
        # print('Norm attention: ', self.to_normal(attention_out).shape)
        # f_out = x + attention_out
        # f_out = self.to_normal(f_out)
        
        attention_out = self.to_normal(attention_out)
        comb = torch.cat((x_in, attention_out), dim = 1)
        # print('final combined: ', comb.shape)
        f_out = self.final_conv(comb)

        return f_out
        
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class Zero_Conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Zero_Conv, self).__init__()
        self.zero_conv = nn.Conv2d(input_channels, output_channels, kernel_size = 1)
        # self.zero_conv.weight.data.fill_(0.0)
        # self.zero_conv.bias.data.fill_(0.0)
        # self.zbn1 = nn.BatchNorm2d(output_channels, eps=0.001, momentum=0.01)
        
        # self.univ_act = nn.ReLU()
    
    def forward(self, x):
        # return self.univ_act(self.zbn1(self.zero_conv(x)))
        return self.zero_conv(x)
        
        
class Control_DepthDecoder(nn.Module):
    def __init__(self, scales=range(1), num_output_channels=1):
        super(Control_DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales
        
        self.univ_act = nn.ReLU()
        
        self.zero_conv1 = zero_module(Zero_Conv(256, 256))
        self.zero_conv2 = zero_module(Zero_Conv(256, 256))
        self.zero_conv3 = zero_module(Zero_Conv(256, 256))
        self.zero_conv4 = zero_module(Zero_Conv(256, 256))
        self.zero_conv5 = zero_module(Zero_Conv(256, 256))
        
        self.zero_conv_out1 = zero_module(Zero_Conv(128, 128))
        self.zero_conv_out2 = zero_module(Zero_Conv(128, 128))
        self.zero_conv_out3 = zero_module(Zero_Conv(128, 128))
        
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        self.conv6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        self.deconv1 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        self.deconv6 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        
        self.depth_pred = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, num_output_channels, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        if 1 in self.scales:
            self.depth_pred1 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(128, num_output_channels, kernel_size=3, stride=1, padding=0),
                nn.Sigmoid()
            )
        if 2 in self.scales:
            self.depth_pred2 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(128, num_output_channels, kernel_size=3, stride=1, padding=0),
                nn.Sigmoid()
            )
        if 3 in self.scales:
            self.depth_pred3 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(128, num_output_channels, kernel_size=3, stride=1, padding=0),
                nn.Sigmoid()
            )
        
        
        self.comb_1 = Combine_Attention(image_height = 6, image_width = 20)
        self.comb_2 = Combine_Attention(image_height = 12, image_width = 40)
        self.comb_3 = Combine_Attention(image_height = 24, image_width = 80)
        
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feature_maps, prev_maps = None):
        self.outputs = {}
        # print('-------------- entered')

        feats = list(reversed(feature_maps))
        
        f0 = self.zero_conv1(feats[0])
        x = self.deconv1(self.conv1(f0))
        
        self.outputs[('feats', 3)] = self.zero_conv_out3(x)
        
        # if prev_maps is not None:
        #     x = self.comb_1(x, prev_maps[('feats', 3)]) # instead of combining with x, it might be better to combine with feats[1] and so on.

        f1 = self.zero_conv2(feats[1])
        x = self.deconv2(torch.cat([self.conv2(f1), x], dim=1))
        if 3 in self.scales:
            self.outputs[("disp", 3)] = self.depth_pred3(x)
        
        
        
        self.outputs[('feats', 2)] = self.zero_conv_out2(x)
        
        if prev_maps is not None:
            x = self.comb_2(x, prev_maps[('feats', 2)])
        
        f2 = self.zero_conv3(feats[2])
        x = self.deconv3(torch.cat([self.conv3(f2), x], dim=1))
        if 2 in self.scales:
            self.outputs[("disp", 2)] = self.depth_pred2(x)
        
        
        self.outputs[('feats', 1)] = self.zero_conv_out1(x)
        
        
        if prev_maps is not None:
            x = self.comb_3(x, prev_maps[('feats', 1)])
        
        # plt.imshow(feats[3][1, 50, :, :].detach().cpu().numpy())
        # plt.show()
        
        f3 = self.zero_conv4(feats[3])
        x = self.deconv4(torch.cat([self.conv4(f3), x], dim=1))
        if 1 in self.scales:
            self.outputs[("disp", 1)] = self.depth_pred1(x)
        
        
        # print('last features shape: ', feats[4].shape)
        f4 = self.zero_conv5(feats[4])
        x = self.deconv5(torch.cat([self.conv5(f4), x], dim=1))
        
        x = self.deconv6(self.conv6(x))
        # print('final deconv shape: ', x.shape)
        x = self.depth_pred(x)
        # print('final depth out shape: ', x.shape)
        
        self.outputs[("disp", 0)] = x
        return self.outputs
