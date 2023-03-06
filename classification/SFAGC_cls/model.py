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
from pointnet2_utils import PointNetSetAbstractionMsg
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
    def __init__(self,num_points,k,pool_n,num_class=2,cuda=True):
        super(Network,self).__init__()
        self.num_points = num_points
        self.k = k
        self.pool_n = pool_n
        self.cuda=cuda
        
        if not self.cuda:
            self.device = torch.device("cpu")
        else:
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
        
        self.att_w01 = nn.Conv1d(64,1,kernel_size=1,bias=False)
        self.bn_att01 = nn.BatchNorm1d(1)
        
        self.gp01_1 = GraphPool(npoint=512,nsample=self.k[1],
                                             in_channel=64,b_c1=32,l_mlp_list=[64,64],NL=False)
        
        self.fatt_conv01 = nn.Conv1d(64*3,64,kernel_size=1,bias=False)
        self.fatt_bn01 = nn.BatchNorm1d(64)
        
        self.co_conv01_1 = nn.Conv1d(64,32,kernel_size=1,bias=False)
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
        
        self.att_w011 = nn.Conv1d(128,1,kernel_size=1,bias=False)
        self.bn_att011 = nn.BatchNorm1d(1)
        
        self.gp01_2 = GraphPool(npoint=128,nsample=self.k[2],
                                             in_channel=128,b_c1=64,l_mlp_list=[128,128],NL=False)
        
        self.fatt_conv02 = nn.Conv1d(128*3,128,kernel_size=1,bias=False)
        self.fatt_bn02 = nn.BatchNorm1d(128)
        
        self.co_conv01_2 = nn.Conv1d(128,64,kernel_size=1,bias=False)
        self.co_bn01_2 = nn.BatchNorm1d(64)
        
        self.gc5 = GC(fin_dim=128,cin_dim=64,fout_dim=256,cout_dim=128)
        self.gc6 = GC(fin_dim=256,cin_dim=128,fout_dim=256)
        
        self.wga01_line3 = nn.Conv1d(128+256*2,512,kernel_size=1,bias=False)
        self.bnga01_line3 = nn.BatchNorm1d(512)
        
        self.co_conv02_1 = nn.Conv1d(3,64,kernel_size=1,bias=False)
        self.co_bn02_1 = nn.BatchNorm1d(64)
        
        self.gc7 = GC(fin_dim=128,cin_dim=64,fout_dim=128,cout_dim=64)
        self.gc8 = GC(fin_dim=128,cin_dim=64,fout_dim=128)
        
        self.wga02_line1 = nn.Conv1d(128*2,512,kernel_size=1,bias=False)
        self.bnga02_line1 = nn.BatchNorm1d(512)
        
        self.conv_d02a = nn.Conv1d(128,128,kernel_size=1,bias=False)
        self.bnconv_d02a = nn.BatchNorm1d(128)
        self.conv_d02b = nn.Conv1d(128,128,kernel_size=1,bias=False)
        self.bnconv_d02b = nn.BatchNorm1d(128)
        self.conv_d02c = nn.Conv1d(128,128,kernel_size=1,bias=False)
        self.bnconv_d02c = nn.BatchNorm1d(128)
        
        self.weight_d02 = nn.Parameter(torch.FloatTensor(128,128*3))
        init.kaiming_uniform_(self.weight_d02,a=0.2)
        self.bn_d02 = nn.BatchNorm1d(128)
        
        self.att_w02 = nn.Conv1d(128,1,kernel_size=1,bias=False)
        self.bn_att02 = nn.BatchNorm1d(1)
        
        self.gp02_1 = GraphPool(npoint=128,nsample=self.k[2],
                                             in_channel=128,b_c1=64,l_mlp_list=[128,128],NL=False)
        
        self.fatt_conv02_1 = nn.Conv1d(128*3,128,kernel_size=1,bias=False)
        self.fatt_bn02_1 = nn.BatchNorm1d(128)
        
        self.co_conv02_2 = nn.Conv1d(128,64,kernel_size=1,bias=False)
        self.co_bn02_2 = nn.BatchNorm1d(64)
        
        self.gc9 = GC(fin_dim=128,cin_dim=64,fout_dim=256,cout_dim=128)
        self.gc10 = GC(fin_dim=256,cin_dim=128,fout_dim=256)
        
        self.wga02_line2 = nn.Conv1d(128+256*2,512,kernel_size=1,bias=False)
        self.bnga02_line2 = nn.BatchNorm1d(512)
        
        self.sa1 = PointNetSetAbstractionMsg(npoint=512,radius_list=[0.2,0.4],nsample_list=[32,128],in_channel=3,mlp_list=[[32,32,64],[32,32,64]])
        self.pconv = nn.Conv1d(128,1024,kernel_size=1,bias=False)
        self.pbn = nn.BatchNorm1d(1024)
        self.fc1p = nn.Linear(1024,512)
        self.fc2p = nn.Linear(512,256)
        self.fc3p = nn.Linear(256,num_class)
        self.dropoutp = nn.Dropout(p=0.3)
        self.bnl1p = nn.BatchNorm1d(512)
        self.bnl2p = nn.BatchNorm1d(256)
        
        self.fc1a = nn.Linear(1024,512)
        self.fc2a = nn.Linear(512,256)
        self.fc3a = nn.Linear(256,num_class)
        self.dropout1 = nn.Dropout(p=0.3)
        self.bnl1a = nn.BatchNorm1d(512)
        self.bnl2a = nn.BatchNorm1d(256)
        
        self.fc10 = nn.Linear(1024,512)
        self.fc20 = nn.Linear(512,256)
        self.fc30 = nn.Linear(256,num_class)
        self.dropout0 = nn.Dropout(p=0.3)
        self.bnl10 = nn.BatchNorm1d(512)
        self.bnl20 = nn.BatchNorm1d(256)
        
        self.fc12 = nn.Linear(1024,512)
        self.fc22 = nn.Linear(512,256)
        self.fc32 = nn.Linear(256,num_class)
        self.dropout2 = nn.Dropout(p=0.3)
        self.bnl12 = nn.BatchNorm1d(512)
        self.bnl22 = nn.BatchNorm1d(256)
        
        self.fc11 = nn.Linear(1024,512)
        self.fc21 = nn.Linear(512,256)
        self.fc31 = nn.Linear(256,num_class)
        self.dropout11 = nn.Dropout(p=0.3)
        self.bnl11 = nn.BatchNorm1d(512)
        self.bnl21 = nn.BatchNorm1d(256)
        
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,num_class)
        self.dropout = nn.Dropout(p=0.3)
        self.bnl1 = nn.BatchNorm1d(512)
        self.bnl2 = nn.BatchNorm1d(256)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,points,as_neighbor=11):
        batch_size = points.shape[0]
        
        #phase1
        new_points0_t,co = self.gc1(feature=points,k=self.k[0],co=points)
        new_points0_1 = self.gc2(feature=new_points0_t,k=self.k[0],co=co)
        
        new_points10 = torch.cat((new_points0_t,new_points0_1),dim=1)
        x_g00 = self.bnga01_line1(self.wga01_line1(new_points10))
        x_max00 = F.adaptive_max_pool1d(x_g00,1).view(batch_size,-1)
        x_avg00 = F.adaptive_avg_pool1d(x_g00,1).view(batch_size,-1)
        x_new10 = torch.cat((x_max00,x_avg00),dim=1)
        
        x0 = F.leaky_relu(self.bnl10(self.fc10(x_new10)),0.2)
        x0 = self.dropout0(x0)
        x0 = F.leaky_relu(self.bnl20(self.fc20(x0)),0.2)
        x0 = self.dropout0(x0)
        x0 = self.fc30(x0)
        logits0 = F.log_softmax(x0,dim=1)
        #phase1
        
        #score-based graph pool 1
        points0 = self.bnconv_d01a(self.conv_d01a(points))
        new_points0_t = self.bnconv_d01b(self.conv_d01b(new_points0_t))
        new_points0_1 = self.bnconv_d01c(self.conv_d01c(new_points0_1))
        new_points10 = torch.cat((points0,new_points0_t,new_points0_1),dim=1)
        new_points0_pool = F.relu(self.bn_d01(self.weight_d01.matmul(new_points10)))
        #*********************attpool1******************
        t1 = new_points0_pool.contiguous()#[B,C,N]
        score1 = F.softmax(self.bn_att01(self.att_w01(t1)).transpose(2,1),dim=1)#[B,N,1]
        #find idx
        pool_idx = torch.sort(score1,dim=1,descending=True)[1][:,0:self.pool_n[0],:].view(batch_size,self.pool_n[0])
        idx_base = torch.arange(0,batch_size,device=self.device).view(-1,1,1)*self.num_points
        pool_idx1 = (pool_idx.view(batch_size,self.pool_n[0],1) + idx_base).view(-1,1)
        #pool
        l1_xyz,new_points0_2 = self.gp01_1(t1,t1,cidx=pool_idx)
        t1 = (new_points0_pool.transpose(2,1)).contiguous()#[B,N,C]
        new_points0_1o = t1.view(batch_size*self.num_points,-1)[pool_idx1,:].view(batch_size,self.pool_n[0],-1).transpose(2,1)#[B,C,N']
        new_points0_1 = (t1*score1).view(batch_size*self.num_points,-1)[pool_idx1,:].view(batch_size,self.pool_n[0],-1).transpose(2,1)#[B,C,N']
        new_points0_2 = torch.cat((new_points0_1o,new_points0_1,new_points0_2),dim=1)
        new_points0_2 = F.relu(self.fatt_bn01(self.fatt_conv01(new_points0_2)))
        #*********************attpool1******************
        
        co = F.relu(self.co_bn01_1(self.co_conv01_1(new_points0_2)))
        
        #phase2
        new_points0,co = self.gc3(feature=new_points0_2,k=self.k[0],co=co)
        new_points0_3 = self.gc4(feature=new_points0,k=self.k[0],co=co)
        
        new_points1 = torch.cat((new_points0_2,new_points0,new_points0_3),dim=1)
        x_g01 = self.bnga01_line2(self.wga01_line2(new_points1))
        x_max01 = F.adaptive_max_pool1d(x_g01,1).view(batch_size,-1)
        x_avg01 = F.adaptive_avg_pool1d(x_g01,1).view(batch_size,-1)
        x_new11 = torch.cat((x_max01,x_avg01),dim=1)
        
        xa = F.leaky_relu(self.bnl1a(self.fc1a(x_new11)),0.2)
        xa = self.dropout1(xa)
        xa = F.leaky_relu(self.bnl2a(self.fc2a(xa)),0.2)
        xa = self.dropout1(xa)
        xa = self.fc3a(xa)
        logitsa = F.log_softmax(xa,dim=1)
        #phase2
        
        #score-based graph pool 2
        new_points0_2 = self.bnconv_d011a(self.conv_d011a(new_points0_2))
        new_points0 = self.bnconv_d011b(self.conv_d011b(new_points0))
        new_points0_3 = self.bnconv_d011c(self.conv_d011c(new_points0_3))
        new_points1 = torch.cat((new_points0_2,new_points0,new_points0_3),dim=1)
        new_points0_pool1 = F.relu(self.bn_d011(self.weight_d011.matmul(new_points1)))
        #*********************attpool11******************
        t1 = new_points0_pool1.contiguous()#[B,C,N]
        score1 = F.softmax(self.bn_att011(self.att_w011(t1)).transpose(2,1),dim=1)#[B,N,1]
        #find idx
        pool_idx = torch.sort(score1,dim=1,descending=True)[1][:,0:self.pool_n[1],:].view(batch_size,self.pool_n[1])
        idx_base = torch.arange(0,batch_size,device=self.device).view(-1,1,1)*512
        pool_idx1 = (pool_idx.view(batch_size,self.pool_n[1],1) + idx_base).view(-1,1)
        #pool
        l2_xyz,new_points0_4 = self.gp01_2(t1,t1,cidx=pool_idx)
        t1 = (new_points0_pool1.transpose(2,1)).contiguous()#[B,N,C]
        new_points0_1o = t1.view(batch_size*512,-1)[pool_idx1,:].view(batch_size,self.pool_n[1],-1).transpose(2,1)#[B,C,N']
        new_points0_1 = (t1*score1).view(batch_size*512,-1)[pool_idx1,:].view(batch_size,self.pool_n[1],-1).transpose(2,1)#[B,C,N']
        new_points0_4 = torch.cat((new_points0_1o,new_points0_1,new_points0_4),dim=1)
        new_points0_4 = F.relu(self.fatt_bn02(self.fatt_conv02(new_points0_4)))
        #*********************attpool11******************
        
        co = F.relu(self.co_bn01_2(self.co_conv01_2(new_points0_4)))
        
        #phase3
        new_points0_5,co = self.gc5(feature=new_points0_4,k=self.k[0],co=co)
        new_points0_6 = self.gc6(feature=new_points0_5,k=self.k[0],co=co)
        
        new_points11 = torch.cat((new_points0_4,new_points0_5,new_points0_6),dim=1)
        x_g011 = self.bnga01_line3(self.wga01_line3(new_points11))
        x_max11 = F.adaptive_max_pool1d(x_g011,1).view(batch_size,-1)
        x_avg11 = F.adaptive_avg_pool1d(x_g011,1).view(batch_size,-1)
        x_new12 = torch.cat((x_max11,x_avg11),dim=1)
        
        xb = F.leaky_relu(self.bnl12(self.fc12(x_new12)),0.2)
        xb = self.dropout2(xb)
        xb = F.leaky_relu(self.bnl22(self.fc22(xb)),0.2)
        xb = self.dropout2(xb)
        xb = self.fc32(xb)
        logitsa1 = F.log_softmax(xb,dim=1)
        #phase3
        
        l1_xyz,l1_points = self.sa1(points,None)
        l1_max = F.relu(self.pbn(self.pconv(l1_points)))
        l1_max = F.adaptive_max_pool1d(l1_max,1).view(batch_size,-1)
        l1_max = F.relu(self.bnl1p(self.fc1p(l1_max)))
        l1_max = self.dropoutp(l1_max)
        l1_max = F.relu(self.bnl2p(self.fc2p(l1_max)))
        l1_max = self.dropoutp(l1_max)
        l1_max = self.fc3p(l1_max)
        logitsp = F.log_softmax(l1_max,dim=1)
        
        co = F.relu(self.co_bn02_1(self.co_conv02_1(l1_xyz)))
        
        #phase4
        new_points01_t,co = self.gc7(feature=l1_points,k=self.k[0],co=co)
        new_points01_1 = self.gc8(feature=new_points01_t,k=self.k[0],co=co)
        
        new_points_sa1 = torch.cat((new_points01_t,new_points01_1),dim=1)
        x_g1 = self.bnga02_line1(self.wga02_line1(new_points_sa1))
        x_max1 = F.adaptive_max_pool1d(x_g1,1).view(batch_size,-1)
        x_avg1 = F.adaptive_avg_pool1d(x_g1,1).view(batch_size,-1)
        x_new2 = torch.cat((x_max1,x_avg1),dim=1)
        
        x1 = F.leaky_relu(self.bnl11(self.fc11(x_new2)),0.2)
        x1 = self.dropout11(x1)
        x1 = F.leaky_relu(self.bnl21(self.fc21(x1)),0.2)
        x1 = self.dropout11(x1)
        x1 = self.fc31(x1)
        logits1 = F.log_softmax(x1,dim=1)
        #phase4
        
        #score-based graph pool 3
        l1_points0 = self.bnconv_d02a(self.conv_d02a(l1_points))
        new_points01_t = self.bnconv_d02b(self.conv_d02b(new_points01_t))
        new_points01_1 = self.bnconv_d02c(self.conv_d02c(new_points01_1))
        new_points_sa1 = torch.cat((l1_points0,new_points01_t,new_points01_1),dim=1)
        new_points01_pool = F.relu(self.bn_d02(self.weight_d02.matmul(new_points_sa1)))
        #*********************attpool2******************
        t2 = new_points01_pool.contiguous()#[B,C,N]
        score2 = F.softmax(self.bn_att02(self.att_w02(t2)).transpose(2,1),dim=1)#[B,N,1]
        #find idx
        pool_idx = torch.sort(score2,dim=1,descending=True)[1][:,0:self.pool_n[1],:].view(batch_size,self.pool_n[1])
        idx_base = torch.arange(0,batch_size,device=self.device).view(-1,1,1)*512
        pool_idx1 = (pool_idx.view(batch_size,self.pool_n[1],1) + idx_base).view(-1,1)
        #pool
        p3_xyz,new_points01_2 = self.gp02_1(t2,t2,cidx=pool_idx)
        t2 = (new_points01_pool.transpose(2,1)).contiguous()#[B,N,C]
        new_points01_1o = t2.view(batch_size*512,-1)[pool_idx1,:].view(batch_size,self.pool_n[1],-1).transpose(2,1)#[B,C,N']
        new_points01_1 = (t2*score2).view(batch_size*512,-1)[pool_idx1,:].view(batch_size,self.pool_n[1],-1).transpose(2,1)#[B,C,N']
        new_points01_2 = torch.cat((new_points01_1o,new_points01_1,new_points01_2),dim=1)#[B,C,N]
        new_points01_2 = F.relu(self.fatt_bn02_1(self.fatt_conv02_1(new_points01_2)))#[B,C,N]
        #*********************attpool2******************
        
        co = F.relu(self.co_bn02_2(self.co_conv02_2(new_points01_2)))
        
        #phase5
        new_points01,co = self.gc9(feature=new_points01_2,k=self.k[0],co=co)
        new_points01_3 = self.gc10(feature=new_points01,k=self.k[0],co=co)
        
        new_points_sa2 = torch.cat((new_points01_2,new_points01,new_points01_3),dim=1)
        x_g11 = self.bnga02_line2(self.wga02_line2(new_points_sa2))
        x_max2 = F.adaptive_max_pool1d(x_g11,1).view(batch_size,-1)
        x_avg2 = F.adaptive_avg_pool1d(x_g11,1).view(batch_size,-1)
        x_new21 = torch.cat((x_max2,x_avg2),dim=1)
        
        x = F.leaky_relu(self.bnl1(self.fc1(x_new21)),0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.bnl2(self.fc2(x)),0.2)
        x = self.dropout(x)
        x = self.fc3(x)
        logits = F.log_softmax(x,dim=1)
        #phase5
        
        return logitsa,logits0,logitsa1,logitsp,logits1,logits