# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:59:43 2019

@author: 81906
"""

#import multiprocessing as mp
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from time import time
import sklearn.metrics as metrics
from tqdm import tqdm
import numpy as np
from ModelNetLoader import ModelNet40, load_data
from model import Network
#from tensorboardX import SummaryWriter
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#torch.backends.cudnn.enabled = False
#mp.set_start_method('spawn')
datapath = './data/ModelNet/'

parse = argparse.ArgumentParser()

parse.add_argument('--batch_size', type=int,default=16,help='batch size')
parse.add_argument('--num_class', type=int,default=40,help='num class')
parse.add_argument('--num_points', type=int,default=1024,help='num points')
parse.add_argument('--lr',type=float,default=0.001,help='learning rate')
parse.add_argument('--k', type=int,default=[20,32,64],help='batch size')
parse.add_argument('--pool_n', type=int,default=[512,128],help='points number of sub-set')

args = parse.parse_args()

#Data Loader
train_data, train_label, test_data, test_label = load_data(datapath, classification=True)
print("The number of training data is: ",train_data.shape[0])
print("The number of test data is: ", test_data.shape[0])
trainDataset = ModelNet40(1024,train_data, train_label)
testDataset = ModelNet40(1024,test_data, test_label)
trainDataLoader = torch.utils.data.DataLoader(trainDataset,batch_size=args.batch_size,shuffle=True,num_workers=8,pin_memory=True)
testDataLoader = torch.utils.data.DataLoader(testDataset,batch_size=args.batch_size,shuffle=False,num_workers=8,pin_memory=True)

num_class = args.num_class
num_points = args.num_points

device=torch.device("cuda")
model = Network(num_points,args.k,args.pool_n,num_class,cuda=True).to(device)
#model = nn.DataParallel(model)
#opt = optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=1e-4)
opt = optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.999),eps=1e-08,weight_decay=1e-4)

#writer = SummaryWriter('log')
def adjust_learning_rate(opt, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in opt.param_groups:
        param_group['lr'] = lr

best_test_acc = 0.0
best_epoch = 0
for epoch in range(200):
    print('************epoch is %d **************' % (epoch))
    adjust_learning_rate(opt,epoch) 
    for param_group in opt.param_groups:
        print("learning rate is:",param_group['lr'])
    train_loss = 0.0
    count = 0.0
    model.train()
    train_pred = []
    train_true = []
    for batch_id,data in tqdm(enumerate(trainDataLoader,0),total=len(trainDataLoader),smoothing=0.9,ascii=True):
        points,target = data
        target = target[:,0].to(device)
        batch_size = points.shape[0]
        target = target.long()
        points = points.transpose(2,1).to(device)#[B,C,N]
        opt.zero_grad()
        logits1,logits2,logits3,logits4,logits5,logits6 = model(points)
        target = target.cuda()      
        loss1 = F.nll_loss(logits1,target)
        loss2 = F.nll_loss(logits2,target)
        loss3 = F.nll_loss(logits3,target)
        loss4 = F.nll_loss(logits4,target)
        loss5 = F.nll_loss(logits5,target)
        loss6 = F.nll_loss(logits6,target)
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        loss.backward()
        opt.step()
        logits = logits1 + logits2 + logits3 + logits4 + logits5 + logits6
        preds = logits.max(dim=1)[1]
        count += batch_size
        train_loss += loss.item() * batch_size
        train_true.append(target.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
    #拼接数组
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    print('loss:%.6f'%(train_loss*1.0/count))
    print('train acc:%.6f'%(metrics.accuracy_score(train_true,train_pred)))
    print('train avg acc:%.6f'%(metrics.balanced_accuracy_score(train_true,train_pred)))
    print('************epoch %d is END**************' % (epoch))
    #writer.add_scalar('Train/Accu',metrics.accuracy_score(train_true,train_pred),epoch)
    
    if epoch % 1 == 0:
        print("************test start************")
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        tq = tqdm(enumerate(testDataLoader,0),total=len(testDataLoader),smoothing=0.9,ascii=True)
        for batch_id,data in tq:
            points,target = data
            target = target[:,0].to(device)
            target = target.long()
            points = points.transpose(2,1).to(device)#[B,C,N]
            batch_size = points.shape[0]
            logits1,logits2,logits3,logits4,logits5,logits6 = model(points)
            target = target.cuda()
            loss1 = F.nll_loss(logits1,target)
            loss2 = F.nll_loss(logits2,target)
            loss3 = F.nll_loss(logits3,target)
            loss4 = F.nll_loss(logits4,target)
            loss5 = F.nll_loss(logits5,target)
            loss6 = F.nll_loss(logits6,target)
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            logits = logits1 + logits2 + logits3 + logits4 + logits5 + logits6
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(target.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true,test_pred)
        print('loss:%.6f'%(test_loss*1.0/count))
        print('test acc:%.6f'%(test_acc))
        print('test avg acc:%.6f'%(metrics.balanced_accuracy_score(test_true,test_pred)))
       # writer.add_scalar('Test/Accu',test_acc,epoch)
        
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_loss = test_loss*1.0/count
            best_test_avg_acc = metrics.balanced_accuracy_score(test_true,test_pred)
        print('best loss:%.6f'%(best_loss))
        print('best test acc:%.6f'%(best_test_acc)) 
        print('best test avg acc:%.6f'%(best_test_avg_acc))
        print('best epoch:%d'%(best_epoch))
        print("************test end************")
#writer.close()
print('BEST_TEST_ACC is: %.6f'%(best_test_acc))    
