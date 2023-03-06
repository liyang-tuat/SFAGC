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
from pointnet2_utils import PointNetSetAbstraction,PointNetSetAbstractionMsg,PointNetFeaturePropagation
from ASNL_utils import pointASNL,pointASNLDecoding

class Network(nn.Module):
    def __init__(self,num_points,k,pool_n,num_part):
        super(Network,self).__init__()
        #self.batch_size = batch_size
        self.num_points = num_points
        self.k = k
        self.pool_n = pool_n
    
        self.device = torch.device("cuda")
        
        self.asnl01_1 = pointASNL(npoint=1024,nsample=self.k[2],
                                             in_channel=3+3,b_c=32,b_c1=32,mlp_list=[32,3+1],l_mlp_list=[32,64])
        
        self.asnl01_2 = pointASNL(npoint=512,nsample=self.k[2],
                                             in_channel=64+3,b_c=32,b_c1=32,mlp_list=[32,64+1],l_mlp_list=[64,128])
        
        self.asnl01_3 = pointASNL(npoint=256,nsample=self.k[2],
                                             in_channel=128+3,b_c=64,b_c1=64,mlp_list=[64,128+1],l_mlp_list=[128,256])
        
        self.asnl01_4 = pointASNL(npoint=128,nsample=self.k[2],
                                             in_channel=256+3,b_c=128,b_c1=128,mlp_list=[128,256+1],l_mlp_list=[256,512])
        
        
        self.fp4 = pointASNLDecoding(nsample=16,in_channel1=256,in_channel2=512,b_c1=256,mlp_list=[512,512],l_mlp_list=[512,32])
        self.fp3 = pointASNLDecoding(nsample=16,in_channel1=128,in_channel2=512,b_c1=256,mlp_list=[512,256],l_mlp_list=[512,32])
        self.fp2 = pointASNLDecoding(nsample=16,in_channel1=64,in_channel2=256,b_c1=128,mlp_list=[256,128],l_mlp_list=[256,32])
        self.fp1 = pointASNLDecoding(nsample=16,in_channel1=3,in_channel2=128,b_c1=64,mlp_list=[128,128],l_mlp_list=[128,32])
        
        self.conv1 = nn.Conv1d(128+6+16, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.4)
        self.conv3 = nn.Conv1d(128, num_part, 1)
    
    def forward(self,points,cls_label):
        batch_size = points.shape[0]
        l0_xyz = points
        l0_points = points
        
        l1_xyz,l1_points = self.asnl01_1(l0_xyz,l0_points)
        l2_xyz,l2_points = self.asnl01_2(l1_xyz,l1_points)
        l3_xyz,l3_points = self.asnl01_3(l2_xyz,l2_points)
        l4_xyz,l4_points = self.asnl01_4(l3_xyz,l3_points)
        
        l3_points = self.fp4(l3_xyz,l4_xyz,l3_points,l4_points)
        l2_points = self.fp3(l2_xyz,l3_xyz,l2_points,l3_points)
        l1_points = self.fp2(l1_xyz,l2_xyz,l1_points,l2_points)
        l0_points = self.fp1(l0_xyz,l1_xyz,l0_points,l1_points)
        
        cls_label_one_hot = cls_label.view(batch_size,16,1).repeat(1,1,self.num_points)
        x = torch.cat((torch.cat([cls_label_one_hot,l0_xyz,points],1), l0_points),dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = self.conv3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss