#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:16:17 2021

@author: liyang
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

class get_model(nn.Module):
    def __init__(self,part_num=50,normal_channel=True):
        super(get_model,self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        
        self.part_num = part_num
        self.conv1 = nn.Conv1d(channel,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,128,1)
        self.conv4 = nn.Conv1d(128,512,1)
        self.conv5 = nn.Conv1d(512,2048,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        
        self.convs1 = nn.Conv1d(2048+2048+512+128+128+64+16,256,1)
        self.convs2 = nn.Conv1d(256,256,1)
        self.convs3 = nn.Conv1d(256,128,1)
        self.convs4 = nn.Conv1d(128,part_num,1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)
        
    def forward(self,point_cloud,label):
        B,C,N = point_cloud.shape
        
        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        
        out4 = F.relu(self.bn4(self.conv4(out3)))
        out5 = self.bn5(self.conv5(out4))
        
        out_max = torch.max(out5,dim=2,keepdim=True)[0]
        out_max = out_max.view(-1,2048)
        
        out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)#[B,C,N]
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num) # [B, N, 50]
        
        return net

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        total_loss = loss
        return total_loss