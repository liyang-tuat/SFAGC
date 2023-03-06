# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:34:40 2019

@author: 81906
"""
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from time import time
import numpy as np
from torch.autograd import Variable

def knn1(x,k):
    """
    input:
        x:一个batch,里面有多个3D模型点云，[B,C,N]
        k:近旁点个数
    output:
        idx:近旁点的索引
    """
    
    inner = -2*torch.matmul(x.transpose(2,1),x)#[B,N,N]
    xx = torch.sum(x**2,dim=1,keepdim=True)#[B,1,N]
    #计算点与点之间的距离，并且将自己与自己的距离置0，与其他点的距离全部变成负数（便于找近旁点）
    pairwise_distance = -xx - inner - xx.transpose(2,1) 
    #找近旁点索引，最大的k个置
    idx = pairwise_distance.topk(k=k,dim=-1)[1]#[B,N,k]
    #print("idx is:",idx)
    
    return idx

def get_graph_feature(x,k=20,idx=None):
    """
    计算特征，论文Dynamic Graph CNN for Learning on Point Clouds中式（7）
    input:
        x:一个batch,里面有多个3D模型点云，[B,C,N]
        k:近旁点个数
        idx:点的索引
    output:
        feature:特征
    """
    device = torch.device("cuda")
    #device = torch.device("cpu")
    
    batch_size = x.shape[0]
    num_dims = x.shape[1]
    num_points = x.shape[2]
    if idx is None:
        idx = knn1(x,k)#[B,N,k]
    
    #用于在各个模型合并成一个的矩阵中找点
    idx_base = torch.arange(0,batch_size,device=device).view(-1,1,1)*num_points
    #idx_base = torch.arange(0,batch_size).view(-1,1,1)*num_points
    idx = idx + idx_base
    idx = idx[:,:,1:k].contiguous()
    idx = idx.view(-1)
    #print("idx is:",idx)
    
    #如果我们在 transpose、permute 操作后执行 view会报错，需要加上.contiguous()
    x = x.transpose(2,1).contiguous()#[B,N,C]
    
    #找出每个点的k个最近邻点
    feature = x.view(batch_size*num_points,-1)[idx,:]
    feature = feature.view(batch_size,num_points,k-1,num_dims)
    #在第三个维度上复制k次
    x = x.view(batch_size,num_points,1,num_dims)
    feature = feature-x
    #print("diff feature is:",feature)
    diff = feature
    f_abs = torch.abs(feature)
    #print("distent is:",f_abs)
    #feature = f_abs

    feature = f_abs
            
    return feature,diff

def get_graph_feature_A(x,xyz,k=20,idx=None):
    """
    计算特征，论文Dynamic Graph CNN for Learning on Point Clouds中式（7）
    input:
        x:一个batch,里面有多个3D模型点云，[B,C,N]
        k:近旁点个数
        idx:点的索引
    output:
        feature:特征
    """
    device = torch.device("cuda")
    #device = torch.device("cpu")
    
    batch_size = x.shape[0]
    num_dims = x.shape[1]
    num_points = x.shape[2]
    if idx is None:
        idx = knn1(xyz,k)#[B,N,k]
    
    #用于在各个模型合并成一个的矩阵中找点
    idx_base = torch.arange(0,batch_size,device=device).view(-1,1,1)*num_points
    #idx_base = torch.arange(0,batch_size).view(-1,1,1)*num_points
    idx = idx + idx_base
    idx = idx[:,:,:k].contiguous()
    idx = idx.view(-1)
    #print("idx is:",idx)
    
    #如果我们在 transpose、permute 操作后执行 view会报错，需要加上.contiguous()
    x = x.transpose(2,1).contiguous()#[B,N,C]
    xyz = xyz.transpose(2,1).contiguous()#[B,N,C]
    
    #找出每个点的k个最近邻点
    feature = x.view(batch_size*num_points,-1)[idx,:]
    feature = feature.view(batch_size,num_points,k,num_dims)
    n_xyz = xyz.view(batch_size*num_points,-1)[idx,:]
    n_xyz = n_xyz.view(batch_size,num_points,k,-1)
    
    xyz = xyz.view(batch_size,num_points,1,-1)
    p = n_xyz - xyz
    
    diff = p
    return feature,p,diff
