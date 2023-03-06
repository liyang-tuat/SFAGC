# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:56:55 2019

@author: 81906
"""

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from time import time
import numpy as np
from torch.autograd import Variable

class Network(nn.Module):
    def __init__(self,num_points,k,pool_n,num_part):
        super(Network,self).__init__()
        #self.batch_size = batch_size
        self.num_points = num_points
        self.k = k
        self.pool_n = pool_n
    
        self.device = torch.device("cuda")
        
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        
        channels = 128
        self.q_conv1 = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv1 = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv1.weight = self.k_conv1.weight
        self.v_conv1 = nn.Conv1d(channels, channels, 1, bias=False)
        self.trans_conv1 = nn.Conv1d(channels, channels, 1, bias=False)
        self.after_norm1 = nn.BatchNorm1d(channels)
        
        self.q_conv2 = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv2 = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv2.weight = self.k_conv2.weight
        self.v_conv2 = nn.Conv1d(channels, channels, 1, bias=False)
        self.trans_conv2 = nn.Conv1d(channels, channels, 1, bias=False)
        self.after_norm2 = nn.BatchNorm1d(channels)
        
        self.q_conv3 = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv3 = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv3.weight = self.k_conv3.weight
        self.v_conv3 = nn.Conv1d(channels, channels, 1, bias=False)
        self.trans_conv3 = nn.Conv1d(channels, channels, 1, bias=False)
        self.after_norm3 = nn.BatchNorm1d(channels)
        
        self.q_conv4 = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv4 = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv4.weight = self.k_conv4.weight
        self.v_conv4 = nn.Conv1d(channels, channels, 1, bias=False)
        self.trans_conv4 = nn.Conv1d(channels, channels, 1, bias=False)
        self.after_norm4 = nn.BatchNorm1d(channels)
        
        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2))

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.dp1 = nn.Dropout(0.4)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, num_part, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)
    
    def forward(self,points,cls_label,as_neighbor=11):
        batch_size = points.shape[0]
        
        x = F.relu(self.bn1(self.conv1(points)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x_q = self.q_conv1(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv1(x)# b, c, n        
        x_v = self.v_conv1(x)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = F.softmax(energy,dim=-1)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n 
        x_r = F.relu(self.after_norm1(self.trans_conv1(x - x_r)))
        x1 = x + x_r
        
        x_q = self.q_conv2(x1).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv2(x1)# b, c, n        
        x_v = self.v_conv2(x1)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = F.softmax(energy,dim=-1)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n 
        x_r = F.relu(self.after_norm2(self.trans_conv2(x1 - x_r)))
        x2 = x1 + x_r
        
        x_q = self.q_conv3(x2).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv3(x2)# b, c, n        
        x_v = self.v_conv3(x2)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = F.softmax(energy,dim=-1)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n 
        x_r = F.relu(self.after_norm3(self.trans_conv3(x2 - x_r)))
        x3 = x2 + x_r
        
        x_q = self.q_conv4(x3).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv4(x3)# b, c, n        
        x_v = self.v_conv4(x3)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = F.softmax(energy,dim=-1)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n 
        x_r = F.relu(self.after_norm4(self.trans_conv4(x3 - x_r)))
        x4 = x3 + x_r
        
        x = torch.cat([x1,x2,x3,x4],dim=1)
        x = self.conv_fuse(x)
        
        x_max = torch.max(x,dim=-1,keepdim=True)[0]
        x_max = x_max.repeat(1,1,self.num_points)#[B,C,N]
        x_avg = torch.mean(x,dim=-1,keepdim=True)
        x_avg = x_avg.repeat(1,1,self.num_points)#[B,C,N]
        
        # FC layers
        cls_label_one_hot = cls_label.view(batch_size,16,1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1,1,self.num_points)
        feat = torch.cat([x,x_max,x_avg,cls_label_feature],dim=1)#[B,1024+1024*2+64,N]
        
        x = F.relu(self.bns1(self.convs1(feat)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss