# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 16:34:43 2019
https://blog.csdn.net/weixin_39373480/article/details/88934146
@author: 81906
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def square_distance(src,dst):
    """
    计算两个矩阵中点与点之间的欧式距离
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2     
	     = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    ^T:转置
    input:
        src:点云数据，[B,N,C]
        dst:点云数据，[B,M,C]
    output:
        dist:src与dst中点与点的欧式距离，[B,N,M]
    """
    B,N,_ = src.shape
    _,M,_ = dst.shape
    dist = -2 * torch.matmul(src,dst.permute(0,2,1))
    dist = dist + torch.sum(src**2,dim=-1).view(B,N,1)
    dist = dist + torch.sum(dst**2,dim=-1).view(B,1,M)
    return dist
    
def farthest_point_sample(xyz,npoints):
    """
    B:一个batch中包含的点云个数
    N：一个点云包含的点的个数（2048）
    C：一个点的坐标维数
    input:
     xyz:点云，[B,N,C]
     npoints:采样的点的个数
    output:
        centroids:各个点云采样得到的点的索引，例：第一个点云的第12,13,14三个点是采样得到的点，
        这里的12,13,14只是点的索引（一个点云用[1,2048,3]的矩阵表示，12,13,14即为第二个维度2048的索引）
        [B,npoints]
    """
    device = xyz.device #选择GPU还是CPU
    B,N,C = xyz.shape
    centroids = torch.zeros(B,npoints,dtype=torch.long).to(device)
    distance = torch.ones(B,N).to(device) * 1e10
    #最远采样点的索引，初始化为随机数，随机数的范围是从0到N，size是[B,1]
    farthest = torch.randint(0,N,(B,),dtype=torch.long).to(device)
    #1维tensor,size是[1,B],batch内点云索引
    batch_indices = torch.arange(B,dtype=torch.long).to(device)
    for i in range(npoints):
        #随机设定一个点是最远点
        centroids[:,i] = farthest
        #取出最远点的坐标
        centroid = xyz[batch_indices,farthest,:].view(B,1,-1)
        #计算点云中所有其他点到该最远的距离的平方和
        dist = torch.sum((xyz-centroid)**2,-1)
        #更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        #从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        farthest = torch.max(distance,-1)[1]
    return centroids

def index_points(points,idx):
    """
    按照索引将点云中的点取出来。
    input:
        points:点云[B,N,C]
        idx:索引[B,S]
    output:
        new_points:根据索引取出的点[B,S,C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1]*(len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B,dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices,idx,:]
    return new_points
    
def query_ball_point(nsample,xyz,new_xyz):
    """
    找到每个球邻域内的点
    input:
        radius:球半径
        nsample:每个球包含的点的个数
        xyz:原始点云，[B,N,C]
        new_xyz:最远采样点（每个球领域的球心），[B,S,C]
    output:
        group_idx:每个球邻域中的点的索引，[B,S,nsample]
    """
    device = xyz.device
    B,N,C = xyz.shape
    _,S,_ = new_xyz.shape
    #用0~N的随机数初始化点的索引
    group_idx = torch.arange(N,dtype=torch.long).to(device).view(1,1,N).repeat([B,S,1])
    #计算球心点与所有点的欧几里得距离sqrdists，[B,S,N]
    sqrdists = square_distance(new_xyz,xyz)
    #找到所有距离大于radius^2的，其group_idx直接置为N；其余的保留原来的值
    
    group_idx = sqrdists.sort(dim=-1)[1][:,:,:nsample]
    #print("t_idx is:",group_idx)
    
    return group_idx

def sample_and_group(npoint,radius,nsample,xyz,points):
    """
    将整个点云划分为nsample个group,每个goup含有npoint个点。
    input:
        npoint:group的个数(最远采样的点数)
        radius:每个group的半径
        nsample:每个group包含的点的个数
        xyz:原始点云,[B,N,C]
        points:点的新的特征（可有可无）,[B,N,D]
    output:
        new_xyz:每个group中心的点（最远采样得到的点），[B,npoint,C]
        new_points:每个group中的点（中心点回到原点），[B,npoint,nsample,C(+D)]
    """
    B,N,C = xyz.shape
    #从原始点云中找出最远采样点
    new_xyz = index_points(xyz,farthest_point_sample(xyz,npoint))
    #每个球形区域中nsample个点的索引idx,[B,npoint,nsample]
    idx = query_ball_point(radius,nsample,xyz,new_xyz)
    #找出每个球形区域中的点grouped_xyz,[B,npoint,nsample,C]
    grouped_xyz = index_points(xyz,idx)
    #减去中心点（让球心坐标变成(0,0,0),相当于将每个球形区域都拉回到坐标原点）
    #grouped_xyz_norm = grouped_xyz - new_xyz.view(B,npoint,1,C)
    #如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
    if points is not None:
        grouped_points = index_points(points, idx)
        #new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        #new_points = grouped_xyz_norm
        new_points = grouped_xyz
    return new_xyz,new_points

def sample_and_group_all(xyz,points):
    """
    将一个点云视作一个group
    input:
        xyz:原始点云，[B,N,C]
        points:点的新的特征（可有可无）,[B,N,D]
    output:
        new_xyz:1个group中心的点（最远采样得到的点），[B,1,C]
        new_points:1个group中的点（中心点回到原点），[B,1,N,C(+D)]
    """
    device = xyz.device
    B,N,C = xyz.shape
    new_xyz = torch.zeros(B,1,C).to(device)
    grouped_xyz_norm = xyz.view(B,1,N,C)
    if points is not None:
        new_points = torch.cat([grouped_xyz_norm, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz,new_points
    
class GraphPool(nn.Module):
    def __init__(self,npoint,nsample,in_channel,b_c1,l_mlp_list,NL=True):
        super(GraphPool,self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.b_c1 = b_c1
        self.NL = NL
        self.l_conv_blocks = nn.ModuleList()
        self.l_bn_blocks = nn.ModuleList()
        self.l_mlp_list = l_mlp_list
        
        self.bconv_0 = nn.Conv2d(3,1,kernel_size=1,bias=False)
        self.bbn_0 = nn.BatchNorm2d(1)
        self.bvconv_0 = nn.Conv2d(3,3,kernel_size=1,bias=False)
        self.bvbn_0 = nn.BatchNorm2d(3)
        self.bww_0 = nn.Conv2d(3+1,1,kernel_size=1,bias=False)
        self.bwbn_0 = nn.BatchNorm2d(1)
        self.wga01_0 = nn.Conv1d(3+2+in_channel-3,in_channel-3,kernel_size=1,bias=False)
        self.bnwga01_0 = nn.BatchNorm1d(in_channel-3)
        
        self.lq_conv = nn.Conv2d(in_channel-3,l_mlp_list[-1],1,bias=False)
        self.lkv_conv = nn.Conv2d(in_channel-3,l_mlp_list[-1]*2,1,bias=False)
        
        self.l1q_conv = nn.Conv2d(in_channel-3,l_mlp_list[-1],1,bias=False)
        self.l1k_conv = nn.Conv2d(in_channel-3,l_mlp_list[-1],1,bias=False)
        self.ch_conv = nn.Conv2d(1,l_mlp_list[-1],1,bias=False)
        
        last_channel = 3
        for out_channel in l_mlp_list:
            self.l_conv_blocks.append(nn.Conv2d(last_channel,out_channel,1))
            self.l_bn_blocks.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.atw = nn.Conv2d(l_mlp_list[-1],l_mlp_list[-1],1)
        self.bnatw = nn.BatchNorm2d(l_mlp_list[-1])
        self.atw1 = nn.Conv2d(l_mlp_list[-1],1,1)
        
        self.pc_conv = nn.Conv2d(l_mlp_list[-1],l_mlp_list[-1],1)
        self.pc_bn = nn.BatchNorm2d(l_mlp_list[-1])
        
        if self.NL is True:
            self.nlq_conv = nn.Conv1d(in_channel,b_c1,1,bias=False)
            self.nlkv_conv = nn.Conv1d(in_channel,b_c1*2,1,bias=False)
            nl_channel = l_mlp_list[-1]
            self.nl_conv = nn.Conv1d(b_c1,nl_channel,1,bias=False)
            self.nl_bn = nn.BatchNorm1d(nl_channel)
        
        self.sk_conv = nn.Conv1d((in_channel)*2,l_mlp_list[-1],1,bias=False)
        self.sk_bn = nn.BatchNorm1d(l_mlp_list[-1])
        
        self.ff_conv = nn.Conv1d(l_mlp_list[-1]*2,l_mlp_list[-1],1,bias=False)
        self.ff_bn = nn.BatchNorm1d(l_mlp_list[-1])
        
        self.conv_sc = nn.Conv1d(l_mlp_list[-1]*2,l_mlp_list[-1],kernel_size=1,bias=False)
        self.bn_sc = nn.BatchNorm1d(l_mlp_list[-1])
        self.conv_sc_h = nn.Conv1d(l_mlp_list[-1],l_mlp_list[-1],kernel_size=1,bias=False)
        self.bn_sc_h = nn.BatchNorm1d(l_mlp_list[-1])
        self.bn = nn.BatchNorm1d(l_mlp_list[-1])
        
    def forward(self,xyz,points,cidx=None,as_neighbor=11):
        """
        input:
            xyz:原始点云数据，[B,C,N]
            points:点云新特征，[B,D,N]
        output:
            new_xyz:每个group的中心点，[B,C,S]
            new_points:每个group的全局特征，[B,D',S]
        """
        xyz = xyz.permute(0,2,1)
        if points is not None:
            points = points.permute(0,2,1)
        #得到各个group，这里不能使用sample_and_group函数，因为这个函数每次调用都将先采样找球心点，但是这里希望球心点固定，只是球的半径改变   
        B,N,C = xyz.shape
        S = self.npoint
        #找到各个group的球心点
        if cidx is None: 
            c_idx = farthest_point_sample(xyz,S)
        else:
            c_idx = cidx
        #print("c_idx is:",c_idx)
        new_xyz = index_points(xyz,c_idx)
        
        #print("new_xyz is:",new_xyz)
        
        K = self.nsample#一个group包含K个点
       
        group_idx = query_ball_point(K,xyz,new_xyz)
        #print("group_idx is:",group_idx)
        grouped_xyz = index_points(xyz,group_idx)
        #print("grouped_xyz is:",grouped_xyz)
        
        if points is not None:
            sf = points
        else:
            sf = xyz
        
        """structure_o"""
        group_idx_s = query_ball_point(as_neighbor,xyz,xyz)
        grouped_xyz_s = index_points(xyz,group_idx_s)[:,:,1:,:]#[B,N,k,C]
        diff = grouped_xyz_s - xyz.view(B,N,1,C)
        #print("diff is:",diff.shape)
        f_abs = torch.abs(diff)
        basevactor = F.leaky_relu(self.bvbn_0(self.bvconv_0(diff.permute(0,3,1,2))),0.2)#[B,C,N,k]
        basevactor = torch.mean(basevactor,dim=3).view(B,-1,N,1)#[B,C,N,1]
        basevactor = basevactor.permute(0,2,3,1)#[B,N,1,C]
        #print("basevactor is:",basevactor.shape)
        distent1 = torch.sum(basevactor ** 2,dim=3) 
        distent2 = torch.sum(diff ** 2,dim=3) 
        inn_product = basevactor.matmul(diff.permute(0,1,3,2)).view(B,N,-1)
        #print("inn_product is:",inn_product.shape)
        cos = (inn_product / torch.sqrt((distent1 * distent2) + 1e-10)).view(B,N,as_neighbor-1,1)
        #print("cos is:",cos.shape)
        diff = F.leaky_relu(self.bbn_0(self.bconv_0(diff.permute(0,3,1,2))),0.2).permute(0,2,3,1)#[B,N,k,1]
        agg_info1 = torch.cat((f_abs,diff),dim=3)#[B,N,k,C]
        t = agg_info1.permute(0,3,1,2)#[B,C,N,k]
        t = F.leaky_relu(self.bwbn_0(self.bww_0(t)),0.2).permute(0,2,3,1)#[B,N,k,1]
        agg_info1 = torch.cat((f_abs,diff,t),dim=3)#[B,N,k,C]
        #print("agg_info1 is:",agg_info1.shape)
        agg_info1 = torch.sum(cos * agg_info1,dim=2)#[B,N,C]
        agg_info1 = torch.cat([sf,agg_info1],dim=-1)#[B,N,C]
        agg_info1 = F.relu(self.bnwga01_0(self.wga01_0(agg_info1.permute(0,2,1))))#[B,C,N]
        #print("agg_info1 is:",agg_info1.shape)
        grouped_feature = index_points(agg_info1.permute(0,2,1),group_idx)#[B,S,K,C]
        """structure_o"""
        
        new_feature = index_points(sf,c_idx)#[B,S,D]
        """Augment sampling"""
        
        agg_info1 = index_points(agg_info1.permute(0,2,1),c_idx)#[B,S,C]
        #print("agg_info1 is:",agg_info1.shape)
        
        new_xyz = new_xyz.permute(0,2,1)#[B,C,S]
        new_feature = new_feature.permute(0,2,1)#[B,D,S]
        
        new_group_feature = torch.cat([grouped_xyz,index_points(points,group_idx)],dim=-1)
        """new"""
        new_c_feature = torch.cat([new_xyz.permute(0,2,1),new_feature.permute(0,2,1)],dim=-1).view(B,S,1,-1)
        
        new_points = torch.cat([new_group_feature - new_c_feature,new_c_feature.repeat(1,1,K,1)],dim=-1)
        
        """skip"""
        skip_spatial = torch.max(new_points,dim=2)[0]
        #print("skip_spatial is:",skip_spatial.shape)
        skip_spatial = F.relu(self.sk_bn(self.sk_conv(skip_spatial.transpose(2,1))))#[B,D',S]
        
        """local"""
        in_gc_t = agg_info1.view(B,S,1,-1)
        #print("new_points is:",new_points.shape)
        Ql = self.lq_conv(in_gc_t.permute(0,3,1,2))#[B,C,S,1]
        #print("Ql is:",Ql.shape)
        KVl = self.lkv_conv(grouped_feature.permute(0,3,1,2))#[B,C,S,K]
        
        Ql1 = self.l1q_conv(in_gc_t.permute(0,3,1,2))#[B,C,S,1]
        #print("Ql is:",Ql.shape)
        Kl1 = self.l1k_conv(grouped_feature.permute(0,3,1,2))#[B,C,S,K]
        QKl1 = torch.matmul(Ql1.permute(0,2,3,1),Kl1.permute(0,2,1,3))#[B,S,1,C][B,S,C,K]=[B,S,1,K]
        QKl1 = self.ch_conv(QKl1.permute(0,2,1,3))#[B,C,S,K]
        
        Kl = KVl[:,:self.l_mlp_list[-1],:,:]#[B,C,S,K]
        #print("Kl is:",Kl.shape)
        Vl = KVl[:,self.l_mlp_list[-1]:,:,:]#[B,C,S,K]
        #print("Vl is:",Vl.shape)
        
        QKl = Ql - Kl + QKl1
        #print("QKl is:",QKl.shape)
        
        position = (grouped_xyz - new_xyz.permute(0,2,1).view(B,S,1,-1)).permute(0,3,1,2)#[B,C,S,K]
        #print("position is:",position.shape)
        for j in range(len(self.l_conv_blocks)):
            l_conv = self.l_conv_blocks[j]
            l_bn = self.l_bn_blocks[j]
            if j == len(self.l_conv_blocks) - 1:
                position = l_conv(position)
            else:
                position =  F.relu(l_bn(l_conv(position)))
            #print("position is:",position.shape)
        
        att = QKl + position
        att = F.relu(self.bnatw(self.atw(att)))
        att = F.softmax(self.atw1(att)/(self.l_mlp_list[-1]**0.5),dim=-1)#[B,1,S,K] 
        new_points = att * (Vl + position)
        new_points = torch.sum(new_points,dim=-1,keepdim=True)#[B,C,S,1]
        new_points = F.relu(self.pc_bn(self.pc_conv(new_points)))#[B,D,S,1]
        new_points = new_points.view(B,-1,S)
        
        """nonlocal"""
        if self.NL is True:
            #print("NL is True")
            nlq = self.nlq_conv(new_feature)#[B,C,S]
            nlkv = self.nlkv_conv(points.permute(0,2,1))#[B,C,N]
        
            nlk = nlkv[:,:self.b_c1,:]
            nlv = nlkv[:,self.b_c1:,:]
        
            attention_map = torch.bmm(nlq.permute(0,2,1),nlk)#[B,S,N]
            attention_map = attention_map/(self.b_c1**0.5)#[B,S,N]
            attention_map = F.softmax(attention_map,dim=-1)
        
            nl_points = torch.bmm(attention_map,nlv.permute(0,2,1))#[B,S,C]
            nl_points = F.relu(self.nl_bn(self.nl_conv(nl_points.permute(0,2,1))))#[B,D',S]
        
            new_points = new_points + nl_points + skip_spatial
        else:
            new_points = torch.cat([new_points , skip_spatial],dim=1)
        
        """Feture Fusion"""
        new_points_h = F.relu(self.ff_bn(self.ff_conv(new_points)))
        
        new_points = self.bn_sc(self.conv_sc(new_points))
        new_points_h = self.bn_sc_h(self.conv_sc_h(new_points_h))
        new_points = F.relu(self.bn(new_points+new_points_h))
        
        if cidx is not None:
            return new_xyz,new_points
        else:
            return new_xyz,new_points,c_idx
