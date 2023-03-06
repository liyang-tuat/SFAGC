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
    
def query_ball_point(radius,nsample,xyz,new_xyz):
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
    group_idx[sqrdists > radius**2] = N
    #做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点
    group_idx = group_idx.sort(dim=-1)[0][:,:,:nsample]
    #考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
    #group_first: [B, S, k]， 实际就是把group_idx中的第一个点的值复制为了[B, S, K]的维度，便利于后面的替换
    group_first = group_idx[:,:,0].view(B,S,1).repeat([1,1,nsample])
    # 找到group_idx中值等于N的点
    mask = group_idx == N
    #将这些点的值替换为第一个点的值
    group_idx[mask] = group_first[mask]
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
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B,npoint,1,C)
    #如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
        
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
    def __init__(self,npoint,radius,nsample,in_channel,mlp,group_all):
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
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
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
        new_points = new_points.permute(0,3,1,2)#[B,C(+D),npoint,nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        #找到每个group中最大的特征，作为该group的全局特征
        new_points = torch.max(new_points,3)[0]
        new_xyz = new_xyz.permute(0,2,1)
        return new_xyz,new_points
    
class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self,npoint,radius_list,nsample_list,in_channel,mlp_list):
        super(PointNetSetAbstractionMsg,self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            
            #last_channel = in_channel + 3
            last_channel = in_channel
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel,out_channel,1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self,xyz,points):
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
        new_xyz = index_points(xyz,farthest_point_sample(xyz,S))
        #保存同一个球心不同半径下的各个全局特征
        new_points_list = []
        for i,radius in enumerate(self.radius_list):
            K = self.nsample_list[i]#一个group包含K个点
            group_idx = query_ball_point(radius,K,xyz,new_xyz)
            grouped_xyz = index_points(xyz,group_idx)
            grouped_xyz -= new_xyz.view(B,S,1,C)
            if points is not None:
                grouped_points = index_points(points,group_idx)
                grouped_points = torch.cat([grouped_xyz,grouped_points],dim=-1)
            else:
                grouped_points = grouped_xyz
            
            grouped_points = grouped_points.permute(0,3,1,2)#[B,D,S,K]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points,3)[0]#在K（K个点）这个维度上找最大，[B,D',S]
            new_points_list.append(new_points)
        
        new_xyz = new_xyz.permute(0,2,1)#[B,C,S]
        #不同半径下的特征拼接成一个特征
        new_points_cat = torch.cat(new_points_list,dim=1)
        return new_xyz,new_points_cat

class PointNetFeaturePropagation(nn.Module):
    def __init__(self,in_channel,mlp):
        super(PointNetFeaturePropagation,self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
            
    def forward(self,xyz1,xyz2,points1,points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)#[B,N,C]
        xyz2 = xyz2.permute(0, 2, 1)#[B,S,C]

        points2 = points2.permute(0, 2, 1)#[B,S,D]
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)#[B,N,M]
            dists, idx = dists.sort(dim=-1)#从小到大
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points