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

class PointNetSetAbstraction(nn.Module):
    def __init__(self,npoint,radius,nsample,group_all):
        """
        input:
            npoint:group的个数
            radius:group的半径
            nsample:每个group包含的点的个数
            in_channel:通道个数
            mlp:一个mlp层的各个维度的列表，例：[64,64,128]
            group_all:bool类型的变量，ture or not
        """
        super(PointNetSetAbstraction,self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
    
    def forward(self,xyz,points):
        """
        input:
            xyz:原始点云，[B,C,N](注意这里的tensor不是[B,N,C])
            points:点的新特征（可有可无），[B,D,N]
        output:
            new_xyz:每个group的中心点，[B,C,S]
            new_points:每个group的点经mlp后得到的group的全局特征，[B,D',S]
        """
        xyz = xyz.permute(0,2,1)
        if points is not None:
            points = points.permute(0,2,1)
        #形成group
        if self.group_all:
            new_xyz,new_points = sample_and_group_all(xyz,points)
        else:
            new_xyz,new_points = sample_and_group(self.npoint,self.radius,self.nsample,xyz,points)
        #new_xyz:每个group的中心点，[B,npoint,C]
        #new_points:每个group中的点（中心点回到原点）,[B,npoint,nsample,C(+D)]
        new_points = new_points.permute(0,3,2,1)#[B,C(+D),nsample,npoint]
        #找到每个group中最大的特征，作为该group的全局特征
        new_points = torch.max(new_points,2)[0]
        new_xyz = new_xyz.permute(0,2,1)
        return new_xyz,new_points
    
class pointASNL(nn.Module):
    def __init__(self,npoint,nsample,in_channel,b_c,b_c1,mlp_list,l_mlp_list,NL=True):
        super(pointASNL,self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.b_c = b_c
        self.b_c1 = b_c1
        self.NL = NL
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.l_conv_blocks = nn.ModuleList()
        self.l_bn_blocks = nn.ModuleList()
        
        self.q_conv = nn.Conv2d(in_channel,b_c,1,bias=False)
        self.kv_conv = nn.Conv2d(in_channel,b_c*2,1,bias=False)
            
        #last_channel = in_channel + 3
        last_channel = b_c
        for out_channel in mlp_list:
            self.conv_blocks.append(nn.Conv2d(last_channel,out_channel,1))
            self.bn_blocks.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        last_channel = in_channel + 3
        for out_channel in l_mlp_list:
            self.l_conv_blocks.append(nn.Conv2d(last_channel,out_channel,1))
            self.l_bn_blocks.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.weight_net = nn.Conv2d(3,32,1)
        self.bn_weight_net = nn.BatchNorm2d(32)
        
        self.pc_conv = nn.Conv2d(32,l_mlp_list[-1],[1,l_mlp_list[-1]])
        self.pc_bn = nn.BatchNorm2d(l_mlp_list[-1])
        
        if self.NL is True:
            self.nlq_conv = nn.Conv1d(in_channel-3,b_c1,1,bias=False)
            self.nlkv_conv = nn.Conv1d(in_channel-3,b_c1*2,1,bias=False)
            nl_channel = l_mlp_list[-1]
            self.nl_conv = nn.Conv1d(b_c1,nl_channel,1,bias=False)
            self.nl_bn = nn.BatchNorm1d(nl_channel)
        
        self.sk_conv = nn.Conv1d(in_channel+3,l_mlp_list[-1],1,bias=False)
        self.sk_bn = nn.BatchNorm1d(l_mlp_list[-1])
        
        self.ff_conv = nn.Conv1d(l_mlp_list[-1],l_mlp_list[-1],1,bias=False)
        self.ff_bn = nn.BatchNorm1d(l_mlp_list[-1])
        
    def forward(self,xyz,points,cidx=None):
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
            grouped_feature = index_points(points,group_idx)
        else:
            grouped_feature = grouped_xyz
        
        #print("grouped_feature is:",grouped_feature)
        shift_group_xyz = grouped_xyz - new_xyz.view(B,S,1,C)
        grouped_points = grouped_feature
        grouped_points = torch.cat([shift_group_xyz,grouped_points],dim=-1)
        #print("grouped_points is:",grouped_points)
        grouped_points = grouped_points.permute(0,3,1,2)#[B,D,S,K]
        #print("grouped_points is:",grouped_points.shape)
        
        q = self.q_conv(grouped_points)
        kv = self.kv_conv(grouped_points)
        
        k = kv[:,:self.b_c,:,:]#[B,D,S,K]
        v = kv[:,self.b_c:,:,:]#[B,D,S,K]
        
        weight = torch.matmul(q.permute(0,2,3,1),k.permute(0,2,1,3))#[B,S,K,K]
        weight = weight/(self.b_c**0.5)
        weight = F.softmax(weight,dim=-1)#[B,S,K,K]
        #print("weight is:",weight.shape)
        
        new_group_points = torch.matmul(weight,v.permute(0,2,3,1))#[B,S,K,D]
        #print("new_group_points is:",new_group_points.shape)
        new_group_points = new_group_points.permute(0,3,1,2)#[B,D,S,K]
        #print("new_group_points is:",new_group_points.shape)
        
        for j in range(len(self.conv_blocks)):
            conv = self.conv_blocks[j]
            bn = self.bn_blocks[j]
            new_group_points =  F.relu(bn(conv(new_group_points)))
            
        #print("new_group_points is:",new_group_points.shape)
        new_group_weight = new_group_points.permute(0,2,3,1)#[B,S,K,D]
        new_group_weight = F.softmax(new_group_weight,dim=2)#[B,S,K,D]
        #print("new_group_weight is:",new_group_weight.shape)
        
        new_weight_xyz = new_group_weight[:,:,:,0].view(B,S,K,1)
        #print("new_weight_xyz is:",new_weight_xyz)
        new_weight_feature = new_group_weight[:,:,:,1:]
        
        new_xyz = torch.sum(new_weight_xyz * grouped_xyz,dim=2)#[B,S,C]
        new_feature = torch.sum(new_weight_feature * grouped_feature,dim=2)#[B,S,D]
        
        new_xyz = new_xyz.permute(0,2,1)#[B,C,S]
        new_feature = new_feature.permute(0,2,1)#[B,C,S]
        
        new_group_feature = torch.cat([grouped_xyz,grouped_feature],dim=-1)
        
        new_points = torch.cat([grouped_xyz - new_xyz.permute(0,2,1).view(B,S,1,C),new_group_feature],dim=-1)
        
        """skip"""
        skip_spatial = torch.max(new_points,dim=2)[0]
        #print("skip_spatial is:",skip_spatial.shape)
        skip_spatial = F.relu(self.sk_bn(self.sk_conv(skip_spatial.transpose(2,1))))#[B,D',S]
        
        """local"""
        #print("new_points is:",new_points.shape)
        new_points = new_points.permute(0,3,1,2)#[B,C,S,K]
        for j in range(len(self.l_conv_blocks)):
            l_conv = self.l_conv_blocks[j]
            l_bn = self.l_bn_blocks[j]
            new_points =  F.relu(l_bn(l_conv(new_points)))
        
        new_grouped_xyz = grouped_xyz - new_xyz.permute(0,2,1).view(B,S,1,C)
        weight = F.relu(self.bn_weight_net(self.weight_net(new_grouped_xyz.permute(0,3,1,2))))#[B,C,S,K]
        new_points = new_points.permute(0,2,1,3)#[B,S,D,K]
        new_points = torch.matmul(new_points,weight.permute(0,2,3,1))#[B,S,D,C]
        
        new_points = F.relu(self.pc_bn(self.pc_conv(new_points.permute(0,3,1,2))))#[B,D,S,1]
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
            new_points = new_points + skip_spatial
        
        """Feture Fusion"""
        new_points = F.relu(self.ff_bn(self.ff_conv(new_points)))
        
        return new_xyz,new_points

class pointASNLDecoding(nn.Module):
    def __init__(self,nsample,in_channel1,in_channel2,b_c1,mlp_list,l_mlp_list,NL=True):
        super(pointASNLDecoding,self).__init__()
        self.nsample = nsample
        self.b_c1 = b_c1
        self.NL = NL
        if self.NL is True:
            self.nlq_conv = nn.Conv1d(in_channel2,b_c1,1,bias=False)
            self.nlkv_conv = nn.Conv1d(in_channel1,b_c1*2,1,bias=False)
            nl_channel = in_channel2
            self.nl_conv = nn.Conv1d(b_c1,nl_channel,1,bias=False)
            self.nl_bn = nn.BatchNorm1d(nl_channel)
        
        self.weight_net = nn.Conv2d(3,32,1)
        self.bn_weight_net = nn.BatchNorm2d(32)
        
        self.l_conv_blocks = nn.ModuleList()
        self.l_bn_blocks = nn.ModuleList()
        
        last_channel = in_channel2
        for out_channel in l_mlp_list:
            self.l_conv_blocks.append(nn.Conv2d(last_channel,out_channel,1))
            self.l_bn_blocks.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.pc_conv = nn.Conv2d(32,l_mlp_list[-1],[1,l_mlp_list[-1]])
        self.pc_bn = nn.BatchNorm2d(l_mlp_list[-1])
        
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        
        last_channel = l_mlp_list[-1] + in_channel1
        for out_channel in mlp_list:
            self.conv_blocks.append(nn.Conv1d(last_channel,out_channel,1))
            self.bn_blocks.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self,xyz1,xyz2,points1,points2):
        xyz1 = xyz1.permute(0, 2, 1)#[B,N,C]
        B, N, C = xyz1.shape
        xyz2 = xyz2.permute(0, 2, 1)#[B,S,C]
        
        points2 = points2.permute(0, 2, 1)#[B,S,D]
        
        dists = square_distance(xyz1, xyz2)#[B,N,M]
        dists, idx = dists.sort(dim=-1)#从小到大
        dists, idx = dists[:, :, :self.nsample], idx[:, :, :self.nsample]  # [B, N, K]
        
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        
        """NonLocal"""
        if self.NL:
            nlq = self.nlq_conv(points2.permute(0,2,1))#[B,D,S]
            nlkv = self.nlkv_conv(points1)#[B,C,N]
        
            nlk = nlkv[:,:self.b_c1,:]
            nlv = nlkv[:,self.b_c1:,:]
        
            attention_map = torch.bmm(nlq.permute(0,2,1),nlk)#[B,S,N]
            attention_map = attention_map/(self.b_c1**0.5)#[B,S,N]
            attention_map = F.softmax(attention_map,dim=-1)
        
            nl_points = torch.bmm(attention_map,nlv.permute(0,2,1))#[B,S,C]
            nl_points = F.relu(self.nl_bn(self.nl_conv(nl_points.permute(0,2,1))))#[B,D,S]
            points2 = points2 + nl_points.permute(0,2,1)#[B,S,D]
            #print("points2 is:",points2.shape)
        
        interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, self.nsample, 1), dim=2)
        
        """Local"""
        grouped_idx = query_ball_point(self.nsample,xyz1,xyz1)
        grouped_xyz = index_points(xyz1,grouped_idx)#[B,N,K,C]
        grouped_feature = index_points(interpolated_points,grouped_idx)#[B,N,K,C]
        
        new_points = grouped_feature.permute(0,3,1,2)#[B,C,N,K]
        for j in range(len(self.l_conv_blocks)):
            l_conv = self.l_conv_blocks[j]
            l_bn = self.l_bn_blocks[j]
            new_points =  F.relu(l_bn(l_conv(new_points)))
        
        grouped_xyz -= xyz1.view(B,N,1,-1)
        weight = F.relu(self.bn_weight_net(self.weight_net(grouped_xyz.permute(0,3,1,2))))#[B,C,N,K]
        
        new_points = torch.matmul(new_points.permute(0,2,1,3),weight.permute(0,2,3,1))#[B,N,C,C]
        new_points = F.relu(self.pc_bn(self.pc_conv(new_points.permute(0,3,1,2))))#[B,C,N,1]
        new_points = new_points.view(B,-1,N)
        
        new_points1 = torch.cat([new_points,points1],dim=1)
        #print("new_points1 is:",new_points1.shape)
        for j in range(len(self.conv_blocks)):
            conv = self.conv_blocks[j]
            bn = self.bn_blocks[j]
            new_points1 =  F.relu(bn(conv(new_points1)))
        
        return new_points1
        