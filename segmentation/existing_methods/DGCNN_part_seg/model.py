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
from knn import get_graph_feature

class Network(nn.Module):
    def __init__(self,num_points,k,pool_n,num_part):
        super(Network,self).__init__()
        #self.batch_size = batch_size
        self.num_points = num_points
        self.k = k
        self.pool_n = pool_n
    
        self.device = torch.device("cuda")
        
        self.dconv1 = nn.Conv2d(3*2,64,kernel_size=1,bias=False)
        self.dbn1 = nn.BatchNorm2d(64)
        self.dconv2 = nn.Conv2d(64,64,kernel_size=1,bias=False)
        self.dbn2 = nn.BatchNorm2d(64)
        
        self.dconv3 = nn.Conv2d(64*2,64,kernel_size=1,bias=False)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dconv4 = nn.Conv2d(64,64,kernel_size=1,bias=False)
        self.dbn4 = nn.BatchNorm2d(64)
        
        self.dconv5 = nn.Conv2d(64*2,64,kernel_size=1,bias=False)
        self.dbn5 = nn.BatchNorm2d(64)
        
        self.dconv6 = nn.Conv1d(64*3,1024,kernel_size=1,bias=False)
        self.dbn6 = nn.BatchNorm1d(1024)
        
        self.cconv = nn.Conv1d(16,64,kernel_size=1,bias=False)
        self.cbn = nn.BatchNorm1d(64)
        
        self.conv1 = nn.Conv1d(1024+64+64*3,256,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(256,256,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.conv3 = nn.Conv1d(256,128,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128,num_part,kernel_size=1,bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,points,cls_label,as_neighbor=11):
        batch_size = points.shape[0]
    
        #graph_conv1
        x = get_graph_feature(points,k=self.k[0])#[B,C,N,k]
        x = F.leaky_relu(self.dbn1(self.dconv1(x)),0.2)
        x = F.leaky_relu(self.dbn2(self.dconv2(x)),0.2)
        x1 = torch.max(x,dim=-1)[0]#[B,C,N]
        
        x = get_graph_feature(x1,k=self.k[0])#[B,C,N,k]
        x = F.leaky_relu(self.dbn3(self.dconv3(x)),0.2)
        x = F.leaky_relu(self.dbn4(self.dconv4(x)),0.2)
        x2 = torch.max(x,dim=-1)[0]#[B,C,N]
        
        x = get_graph_feature(x2,k=self.k[0])#[B,C,N,k]
        x = F.leaky_relu(self.dbn5(self.dconv5(x)),0.2)
        x3 = torch.max(x,dim=-1)[0]#[B,C,N]
        
        x = torch.cat([x1,x2,x3],dim=1)#[B,C,N]
        
        x = F.leaky_relu(self.dbn6(self.dconv6(x)),0.2)#[B,C,N]
        x = torch.max(x,dim=-1,keepdim=True)[0]#[B,C,1]
        x = x.repeat(1,1,self.num_points)
        
        cls_label_one_hot = cls_label.view(batch_size,16,1).repeat(1,1,self.num_points)
        cv = F.leaky_relu(self.cbn(self.cconv(cls_label_one_hot)),0.2)#[B,16,N]
        
        x = torch.cat([x,cv],dim=1)#[B,C,N]
        
        x = torch.cat([x,x1,x2,x3],dim=1)#[B,C,N]
        
        # FC layers
        feat = F.leaky_relu(self.bn1(self.conv1(x)),0.2)#[B,C,N]
        feat = self.drop1(feat)
        feat = F.leaky_relu(self.bn2(self.conv2(feat)),0.2)#[B,C,N]
        feat = self.drop2(feat)
        feat = F.leaky_relu(self.bn3(self.conv3(feat)),0.2)#[B,C,N]
        x = self.conv4(feat)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss