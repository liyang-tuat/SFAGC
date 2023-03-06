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
from knn import get_graph_feature,get_graph_feature_A
from pointnet2_utils import PointNetSetAbstraction,PointNetSetAbstractionMsg,PointNetFeaturePropagation
from GraphPool_utils import GraphPool

class GC(nn.Module):
    def __init__(self,fin_dim,cin_dim,fout_dim,cout_dim=None):
        super(GC,self).__init__()
        self.fin_dim = fin_dim
        self.cin_dim = cin_dim
        self.fout_dim = fout_dim
        self.cout_dim = cout_dim
        
        self.sk_conv = nn.Conv1d(self.cin_dim+self.fin_dim,self.fout_dim,kernel_size=1,bias=False)
        self.sk_bn = nn.BatchNorm1d(self.fout_dim)
        
        self.bvconv = nn.Conv2d(self.cin_dim,self.cin_dim,kernel_size=1,bias=False)
        self.bvbn = nn.BatchNorm2d(self.cin_dim)
        self.reconv = nn.Conv2d(self.cin_dim,1,kernel_size=1,bias=False)
        self.rebn = nn.BatchNorm2d(1)
        self.seconv = nn.Conv2d(self.cin_dim+1,1,1,bias=False)
        self.sebn = nn.BatchNorm2d(1)
        self.fconv = nn.Conv1d(self.cin_dim+2+self.fin_dim,self.fin_dim,1,bias=False)
        self.fbn = nn.BatchNorm1d(self.fin_dim)
        
        self.atwq_m = nn.Conv2d(self.fin_dim,self.fin_dim,1,bias=False)
        self.atwk_m = nn.Conv2d(self.fin_dim,self.fin_dim,1,bias=False)
        
        self.atwq = nn.Conv2d(self.fin_dim,self.fout_dim,1,bias=False)
        self.atwkv = nn.Conv2d(self.fin_dim,self.fout_dim*2,1,bias=False)
        self.atwa1 = nn.Conv2d(1,self.fout_dim,1,bias=False)
        self.atwa2 = nn.Conv2d(self.fout_dim,self.fout_dim,1,bias=False)
        self.bnatwa2 = nn.BatchNorm2d(self.fout_dim)
        self.atwa3 = nn.Conv2d(self.fout_dim,1,1,bias=False)
        self.atwp1 = nn.Conv2d(self.cin_dim,self.fout_dim,1,bias=False)
        self.bnatwp1 = nn.BatchNorm2d(self.fout_dim)
        self.atwp2 = nn.Conv2d(self.fout_dim,self.fout_dim,1,bias=False)
        
        self.conv = nn.Conv1d(self.fout_dim,self.fout_dim,kernel_size=1,bias=False)
        self.bn = nn.BatchNorm1d(self.fout_dim)
        self.int_conv = nn.Conv1d(self.fout_dim+self.fout_dim,self.fout_dim,kernel_size=1,bias=False)
        self.int_bn = nn.BatchNorm1d(self.fout_dim)
        
        self.conv_sc = nn.Conv1d(self.fout_dim+self.fout_dim,self.fout_dim,kernel_size=1,bias=False)
        self.bn_sc = nn.BatchNorm1d(self.fout_dim)
        self.conv_sc_h = nn.Conv1d(self.fout_dim,self.fout_dim,kernel_size=1,bias=False)
        self.bn_sc_h = nn.BatchNorm1d(self.fout_dim)
        self.bn_h = nn.BatchNorm1d(self.fout_dim)
        
        if self.cout_dim is not None:
            self.co_conv = nn.Conv1d(self.cin_dim,self.cout_dim,kernel_size=1,bias=False)
            self.co_bn = nn.BatchNorm1d(self.cout_dim)
        
    def forward(self,feature,k,co,as_neighbor=11):
        B,C,N = feature.shape
        _,Cco,_ = co.shape
        in_gc = feature.view(B,C,N,1)
        
        f_abs,diff = get_graph_feature(x=co,k=as_neighbor)
        basevactor = F.leaky_relu(self.bvbn(self.bvconv(diff.permute(0,3,1,2))),0.2)#[B,C,N,k]
        basevactor = torch.mean(basevactor,dim=3).view(B,Cco,N,1)#[B,C,N,1]
        basevactor = basevactor.permute(0,2,3,1)#[B,N,1,C]
        distent1 = torch.sum(basevactor ** 2,dim=3) 
        distent2 = torch.sum(diff ** 2,dim=3) 
        inn_product = basevactor.matmul(diff.permute(0,1,3,2)).view(B,N,-1)
        cos = (inn_product / torch.sqrt((distent1 * distent2) + 1e-10)).view(B,N,as_neighbor-1,1) #feature angle
        diff = F.leaky_relu(self.rebn(self.reconv(diff.permute(0,3,1,2))),0.2).permute(0,2,3,1)#[B,N,k,1],ralational embedding 
        sv = torch.cat((f_abs,diff),dim=3)#[B,N,k,C],structure vectors
        t = sv.permute(0,3,1,2)#[B,C,N,k]
        t = F.leaky_relu(self.sebn(self.seconv(t)),0.2).permute(0,2,3,1)#[B,N,k,1]
        sv = torch.cat((f_abs,diff,t),dim=3)#[B,N,k,C],structure vectors
        sv = (cos * sv).permute(0,3,1,2)#[B,C,N,k]
        sv = torch.sum(sv,dim=-1)#[B,C,N],local structure projection aggregation
        agg_f = torch.cat([feature,sv],dim=1)#[B,C,N]
        agg_f = F.relu(self.fbn(self.fconv(agg_f)))#[B,C,N]
        
        nS,p,diff = get_graph_feature_A(x=agg_f,xyz=co,k=k)
        new_points = torch.cat((diff.permute(0,3,1,2),in_gc.repeat(1,1,1,k)),dim=1)
        new_points = torch.max(new_points,dim=-1)[0]
        new_points = F.relu(self.sk_bn(self.sk_conv(new_points)))
        in_gc_t = agg_f.view(B,-1,N,1)#[B,C,N,1]
        node = nS.permute(0,3,1,2)#[B,C,N,k]
        Qm = self.atwq_m(in_gc_t)#[B,C,N,1]
        Km = self.atwk_m(node)#[B,C,N,k]
        Am = torch.matmul(Qm.permute(0,2,3,1),Km.permute(0,2,1,3))#[B,N,1,k]
        at = self.atwa1(Am.permute(0,2,1,3))#[B,C,N,k],multiply attention function
        Q = self.atwq(in_gc_t)#[B,C,N,1]
        KV = self.atwkv(node)#[B,C,N,k]
        K = KV[:,:self.fout_dim,:,:]
        V = KV[:,self.fout_dim:,:,:]
        QK = Q - K#[B,C,N,k],subtraction attention function
        p = p.permute(0,3,1,2)#[B,C,N,k]
        p = F.relu(self.bnatwp1(self.atwp1(p)))#[B,C,N,k]
        p = self.atwp2(p)#[B,C,N,k],position embedding
        QK = QK + p + at#[B,C,N,k]
        A = F.relu(self.bnatwa2(self.atwa2(QK)))#[B,C,N,k]
        A = self.atwa3(A)#[B,1,N,k]
        A = A/(self.fout_dim**0.5)
        A = F.softmax(A,dim=-1)
        node1 = A * (V+p)
        node1 = torch.sum(node1,dim=-1)#[B,C,N]
        node1 = F.leaky_relu(self.bn(self.conv(node1)),0.2)
        node1 = torch.cat((new_points,node1),dim=1)
        self_f = F.leaky_relu(self.int_bn(self.int_conv(node1)),0.2)
        new_points_sc_t = self.bn_sc(self.conv_sc(node1))
        new_points_t = self.bn_sc_h(self.conv_sc_h(self_f))
        new_points_t = F.leaky_relu(self.bn_h(new_points_sc_t+new_points_t),0.2)
        
        if self.cout_dim is not None:
            co = F.relu(self.co_bn(self.co_conv(co)))#coordinates updating
            return new_points_t,co
        else:
            return new_points_t

class Network(nn.Module):
    def __init__(self,num_points,k,pool_n,num_part):
        super(Network,self).__init__()
        #self.batch_size = batch_size
        self.num_points = num_points
        self.k = k
        self.pool_n = pool_n
    
        self.device = torch.device("cuda")
        
        self.gc1 = GC(fin_dim=3,cin_dim=3,fout_dim=64,cout_dim=32)
        self.gc2 = GC(fin_dim=64,cin_dim=32,fout_dim=64)
        
        self.wga01_line1 = nn.Conv1d(64*2,512,kernel_size=1,bias=False)
        self.bnga01_line1 = nn.BatchNorm1d(512)
        
        self.conv_d01a = nn.Conv1d(3,64,kernel_size=1,bias=False)
        self.bnconv_d01a = nn.BatchNorm1d(64)
        self.conv_d01b = nn.Conv1d(64,64,kernel_size=1,bias=False)
        self.bnconv_d01b = nn.BatchNorm1d(64)
        self.conv_d01c = nn.Conv1d(64,64,kernel_size=1,bias=False)
        self.bnconv_d01c = nn.BatchNorm1d(64)
        
        self.weight_d01 = nn.Parameter(torch.FloatTensor(64,64*3))
        init.kaiming_uniform_(self.weight_d01,a=0.2)
        self.bn_d01 = nn.BatchNorm1d(64)
        
        self.gp01_1 = GraphPool(npoint=512,nsample=self.k[1],
                                             in_channel=64+3,b_c1=32,l_mlp_list=[64,64],NL=False)
        
        self.co_conv01_1 = nn.Conv1d(3,32,kernel_size=1,bias=False)
        self.co_bn01_1 = nn.BatchNorm1d(32)
        
        self.gc3 = GC(fin_dim=64,cin_dim=32,fout_dim=128,cout_dim=64)
        self.gc4 = GC(fin_dim=128,cin_dim=64,fout_dim=128)
        
        self.wga01_line2 = nn.Conv1d(64+128*2,512,kernel_size=1,bias=False)
        self.bnga01_line2 = nn.BatchNorm1d(512)
        
        self.conv_d011a = nn.Conv1d(64,128,kernel_size=1,bias=False)
        self.bnconv_d011a = nn.BatchNorm1d(128)
        self.conv_d011b = nn.Conv1d(128,128,kernel_size=1,bias=False)
        self.bnconv_d011b = nn.BatchNorm1d(128)
        self.conv_d011c = nn.Conv1d(128,128,kernel_size=1,bias=False)
        self.bnconv_d011c = nn.BatchNorm1d(128)
        
        self.weight_d011 = nn.Parameter(torch.FloatTensor(128,128*3))
        init.kaiming_uniform_(self.weight_d011,a=0.2)
        self.bn_d011 = nn.BatchNorm1d(128)
        
        self.gp01_2 = GraphPool(npoint=128,nsample=self.k[2],
                                             in_channel=128+3,b_c1=64,l_mlp_list=[128,128],NL=False)
        
        self.co_conv01_2 = nn.Conv1d(3,64,kernel_size=1,bias=False)
        self.co_bn01_2 = nn.BatchNorm1d(64)
        
        self.gc5 = GC(fin_dim=128,cin_dim=64,fout_dim=256,cout_dim=128)
        self.gc6 = GC(fin_dim=256,cin_dim=128,fout_dim=256)
        
        self.wga01_line3 = nn.Conv1d(128+256*2,512,kernel_size=1,bias=False)
        self.bnga01_line3 = nn.BatchNorm1d(512)
        
        self.conv_d012a = nn.Conv1d(128,256,kernel_size=1,bias=False)
        self.bnconv_d012a = nn.BatchNorm1d(256)
        self.conv_d012b = nn.Conv1d(256,256,kernel_size=1,bias=False)
        self.bnconv_d012b = nn.BatchNorm1d(256)
        self.conv_d012c = nn.Conv1d(256,256,kernel_size=1,bias=False)
        self.bnconv_d012c = nn.BatchNorm1d(256)
        
        self.weight_d012 = nn.Parameter(torch.FloatTensor(256,256*3))
        init.kaiming_uniform_(self.weight_d012,a=0.2)
        self.bn_d012 = nn.BatchNorm1d(256)
        
        self.fp2 = PointNetFeaturePropagation(in_channel=128+512*2+256+512*2, mlp=[256,128])
        
        self.fp1 = PointNetFeaturePropagation(in_channel=64+512*2+128, mlp=[128, 128])
        
        self.conv1 = nn.Conv1d(128+3+3+16, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(128, num_part, 1)
    
    def forward(self,points,cls_label,as_neighbor=11):
        batch_size = points.shape[0]
        l0_xyz = points
        l0_points = points
        
        #phase 1
        new_points0_t,co = self.gc1(feature=points,k=self.k[0],co=points)
        new_points0_1 = self.gc2(feature=new_points0_t,k=self.k[0],co=co)
        
        new_points10 = torch.cat((new_points0_t,new_points0_1),dim=1)
        new_points10 = self.bnga01_line1(self.wga01_line1(new_points10))
        x_max00 = F.adaptive_max_pool1d(new_points10,1).view(batch_size,-1)
        x_avg00 = F.adaptive_avg_pool1d(new_points10,1).view(batch_size,-1)
        x_new10 = torch.cat((x_max00,x_avg00),dim=1)
        #phase 1
        
        #FPS-based graph pool 1
        points0 = self.bnconv_d01a(self.conv_d01a(points))
        new_points0_t = self.bnconv_d01b(self.conv_d01b(new_points0_t))
        new_points0_1 = self.bnconv_d01c(self.conv_d01c(new_points0_1))
        new_points10 = torch.cat((points0,new_points0_t,new_points0_1),dim=1)
        new_points0_pool = F.relu(self.bn_d01(self.weight_d01.matmul(new_points10)))
        l0_points = new_points0_pool
        
        l1_xyz,l1_points,pool_idx = self.gp01_1(l0_xyz,l0_points)

        new_points0_2 = l1_points
        co = F.relu(self.co_bn01_1(self.co_conv01_1(l1_xyz)))
        
        #phase 2
        new_points0,co = self.gc3(feature=new_points0_2,k=self.k[0],co=co)
        new_points0_3 = self.gc4(feature=new_points0,k=self.k[0],co=co)
        
        new_points1 = torch.cat((new_points0_2,new_points0,new_points0_3),dim=1)
        new_points1 = self.bnga01_line2(self.wga01_line2(new_points1))
        x_max01 = F.adaptive_max_pool1d(new_points1,1).view(batch_size,-1)
        x_avg01 = F.adaptive_avg_pool1d(new_points1,1).view(batch_size,-1)
        x_new11 = torch.cat((x_max01,x_avg01),dim=1)
        #phase 2
        
        #FPS-based graph pool 2
        new_points0_2 = self.bnconv_d011a(self.conv_d011a(new_points0_2))
        new_points0 = self.bnconv_d011b(self.conv_d011b(new_points0))
        new_points0_3 = self.bnconv_d011c(self.conv_d011c(new_points0_3))
        new_points1 = torch.cat((new_points0_2,new_points0,new_points0_3),dim=1)
        new_points0_pool1 = F.relu(self.bn_d011(self.weight_d011.matmul(new_points1)))
        l1_points = new_points0_pool1
        
        l2_xyz,l2_points,pool_idx = self.gp01_2(l1_xyz,l1_points)
        
        new_points0_4 = l2_points
        co = F.relu(self.co_bn01_2(self.co_conv01_2(l2_xyz)))
        
        #phase 3
        new_points0_5,co = self.gc5(feature=new_points0_4,k=self.k[0],co=co)
        new_points0_6 = self.gc6(feature=new_points0_5,k=self.k[0],co=co)
        
        new_points11 = torch.cat((new_points0_4,new_points0_5,new_points0_6),dim=1)
        new_points11 = self.bnga01_line3(self.wga01_line3(new_points11))
        x_max11 = F.adaptive_max_pool1d(new_points11,1).view(batch_size,-1)
        x_avg11 = F.adaptive_avg_pool1d(new_points11,1).view(batch_size,-1)
        x_new12 = torch.cat((x_max11,x_avg11),dim=1)
        #phase 3
        
        new_points0_4 = self.bnconv_d012a(self.conv_d012a(new_points0_4))
        new_points0_5 = self.bnconv_d012b(self.conv_d012b(new_points0_5))
        new_points0_6 = self.bnconv_d012c(self.conv_d012c(new_points0_6))
        new_points11 = torch.cat((new_points0_4,new_points0_5,new_points0_6),dim=1)
        new_points0_pool2 = F.relu(self.bn_d012(self.weight_d012.matmul(new_points11)))
        l2_points = new_points0_pool2
        
        # Feature Propagation layers
        l2_points = torch.cat((l2_points,x_new12.view(batch_size,-1,1).repeat(1,1,128)),dim=1)
        
        l1_points = torch.cat((l1_points,x_new11.view(batch_size,-1,1).repeat(1,1,512)),dim=1)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        l0_points = torch.cat((l0_points,x_new10.view(batch_size,-1,1).repeat(1,1,self.num_points)),dim=1)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        
        # FC layers
        cls_label_one_hot = cls_label.view(batch_size,16,1).repeat(1,1,self.num_points)
        l0_points = torch.cat((torch.cat([cls_label_one_hot,l0_xyz,points],1),l0_points),dim=1)
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss