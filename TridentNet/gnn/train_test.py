import time
import torch
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import torch.distributed as dist
from config import args
import math
import pandas as pd
import copy
import torch_geometric
import os
# constants
c = 0.2998
n_water = 1.35  # for pure water
c_n = c / n_water
costh = 1 / n_water
tanth = math.sqrt(1 - costh*costh) / costh
sinth = costh * tanth

def getMLogl(preds, hits):

    pred_r0 = preds[:,3:]
    pred_n = preds[:,:3]
    pred_n = pred_n[:,:3] / pred_n[:,:3].norm(dim=1).view(-1,1)

    hitt = hits[:,:1]
    hitr = hits[:,1:]

    delta_r = hitr - pred_r0
    l = (pred_n * delta_r).sum(dim=1).view(-1,1)
    d = torch.sqrt((delta_r**2).sum(dim=1).view(-1,1) - l**2)
    t = (l - d / tanth)/c + (d / sinth)/c_n
    resT = hitt - t

    # resT = resT / 20
    # d = (d - 20) / 50
    # mlogl = 0.5*(torch.sqrt(resT**2 / 2 + 1) - 1) + 0.5*(torch.sqrt(d**2 / 2 + 1) - 1) 

    distance = d
    time = resT
    mean = 0.0199 * 5*torch.sqrt(distance) - 0.114
    tau = 0.0669 * 5*torch.sqrt(distance) + 0.328
    alpha = 0.0127 * 5*torch.sqrt(distance) + 0.538
    time_ = (time - mean) / tau
    time_ = torch.clip(time_, -20, 40)
    t_sk = alpha * time_
    rst = 2 / (torch.exp(-t_sk) + 1) / (torch.pi*tau*(time_*time_+1))
    mlogl = rst
    print(rst, flush=True)
    return mlogl

def leastsq3DLineFitAccWithT(output, labels, extra):
    '''
    reference: https://blog.csdn.net/qwertyu_1234567/article/details/117918602
    '''
    extra = copy.deepcopy(extra.detach()).cpu()
    output = output.detach().cpu()
    labels = labels.detach().cpu()

    dis = np.sqrt((output.numpy()**2).sum(axis=1)).reshape(-1)
    output = output + extra.pos
    t = extra.x[:,1] / c - dis / c_n

    points = pd.DataFrame({'x':output[:,0], 'y':output[:,1], 'z':output[:,2], 't':t,
                           'id':extra.batch}).set_index('id', drop=True)
                           


    # weight for each dom
    # # best:
    weight = torch.clamp(extra.nhits, min=0, max=30)
    # weight = labels[:,7]
    # weight = torch.ones(len(labels))

    points['weight'] = weight
    points['tt'] = points['t'] * points['t'] * points['weight']
    points['xt'] = points['x'] * points['t'] * points['weight']
    points['yt'] = points['y'] * points['t'] * points['weight']
    points['zt'] = points['z'] * points['t'] * points['weight']
    points['t'] = points['t'] * points['weight']
    points['x'] = points['x'] * points['weight']
    points['y'] = points['y'] * points['weight']
    points['z'] = points['z'] * points['weight']
    # points['n'] = 1

    # grouped = points.groupby('id').mean()
    grouped = points.groupby('id').sum()
    grouped = grouped.div(grouped.weight, axis=0)

    # Line fit k1, b2, k2, b2
    den = grouped.tt - grouped.t * grouped.t + 1e-12 # 公式分母. 防止其为0
    n1 = (grouped.xt - grouped.x * grouped.t) / den
    n2 = (grouped.yt - grouped.y * grouped.t) / den
    n3 = (grouped.zt - grouped.z * grouped.t) / den

    # Get nx,ny,nz
    grouped['nx'] = n1
    grouped['ny'] = n2
    grouped['nz'] = n3
    preds = torch.from_numpy(grouped[['nx','ny','nz']].to_numpy()).type(torch.float)
    preds = preds.to(device)

    return preds

mean = [0.002680, 0.000874, -0.046995, 0.427360, -0.354793, 20.003420]
std = [0.517624, 0.518305, 0.679124, 927.159267, 930.445741, 295.157116]
def getLossAcc(criterion, output, labels, extra=None):
    if torch.isnan(output).sum():
        print("Error: there is nan in output")
    # normalize labels, un-normalize output
    to_pred = labels.clone()
    pred_output = output.clone()
    # for i in range(len(mean)):
    #     to_pred[:,i] = (to_pred[:,i] - mean[i]) / std[i] 
    #     output[:,i] = output[:,i] * std[i] + mean[i]

    # # Get nDoms for each element
    # _, nDoms = torch.unique(inputs[0][:,-1], return_counts=True)
    # preds = torch.repeat_interleave(output, nDoms, dim=0)
    # r0 = torch.repeat_interleave(labels[:,3:], nDoms, dim=0)
    # preds = torch.concat([preds, r0], axis=1)
    # mlogl = getMLogl(preds, extra)
    # loss = mlogl.sum()

    # Loss
    # mse_nonReduce = criterion(pred_output / pred_output.norm(dim=1).view(-1,1), to_pred  / to_pred.norm(dim=1).view(-1,1))
    # # mse_nonReduce = criterion(output, labels)
    # loss = mse_nonReduce.sum() + (0.01 * (pred_output.norm(dim=1).view(-1,1) - 1)**2).sum()
    loss = ((criterion(pred_output, to_pred))).sum()
    # loss = criterion(pred_output, to_pred).sum()

    # Acc
    # output = leastsq3DLineFitAccWithT(output, labels, extra)

    truth_normed = labels[:,:3] / labels[:,:3].norm(dim=1).view(-1,1)
    output_normed = output[:,:3] / output[:,:3].norm(dim=1).view(-1,1)
    diff = ((output_normed - truth_normed) ** 2).sum(dim=1).clamp(0,4)
    acc = torch.arccos( 1 - 1./2 * (diff) ) / np.pi * 180
    # print(acc, flush=True)
    # print(extra.x, flush=True)
    # print(pred_output[:5], flush=True)
    # print(to_pred[:5], flush=True)
    # print(loss/len(to_pred), flush=True)
    # print(acc.mean(), flush=True)
    # print(f"{truth_normed[0:5]},{output_normed[0:5]}", flush=True)

    return loss, acc, output_normed


device = args.device
def train_one_epoch(model, trainloader, criterion, optimizer, res):
    model.train()
    
    timeBegin = time.time()
    total_n_data, total_correct, total_loss = torch.tensor(0.).to(device),\
                        torch.tensor(0.).to(device), torch.tensor(0.).to(device)
    total_n_loss = 0
    for i, data in enumerate(trainloader, 0):
        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # Model output
        data = data.to(device)
        target = data.y
        output = model(data)
        
        # print(output,flush=True)
        loss, acc, _ = getLossAcc(criterion, output, target, extra=data)    
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        optimizer.step()

        total_n_loss += len(output)
        total_n_data += len(acc)
        total_correct += acc.sum().item()
        total_loss += loss.item()

    # record result
    if args.distributed:
        dist.barrier()
        for ary in [total_n_data, total_correct, total_loss]:   # collect data from all gpus
            dist.all_reduce(ary, op=dist.ReduceOp.SUM)      
    res['train_time'].append(time.time()-timeBegin)
    res['train_loss'].append(float(total_loss/total_n_loss))
    res['train_acc'].append(float(total_correct/total_n_data))
    

def test_one_epoch(model, testloader, criterion, res, check_output = False, save_new_graph_loc=None, save_new_graph_id=None):
    model.eval()
    timeBegin = time.time()
    # Record scores
    predLog = []
    outputLog = []

    total_n_data, total_correct, total_loss = 0, 0, 0
    total_n_loss = 0
    for i, data in enumerate(testloader, 0):
        data = data.to(device)
        target = data.y

        with torch.no_grad():
            output = model(data)        
            
            loss, acc, output_normed = getLossAcc(criterion, output, target, extra=data)     

            total_n_loss += len(output)
            total_n_data += len(acc)
            total_correct += acc.sum().item()
            total_loss += loss.item()
            predLog.append(torch.cat((target, output_normed), 1).detach().cpu().numpy() )
            outputLog.append(torch.cat((target, output_normed), 1).detach().cpu().numpy() )
            # predLog.append(torch.cat((output, inputs[0][:,-1:]), 1).detach().cpu().numpy() )

    if check_output:
        # print(" labels: ", labels[0:5,:3]-labels[0:5,3:])
        print(" labels: ", data.inject[0:5])
        print(" output: ", output[0:5, :])
        print()

    # record result
    prediction = np.zeros((0,predLog[0].shape[1]))
    for s in predLog:
        prediction = np.concatenate( (prediction, s), 0)
    
    outputs = np.concatenate(outputLog, axis=0)

    del predLog, outputLog

    res['test_time'].append(time.time()-timeBegin)
    res['test_loss'].append(float(total_loss/total_n_loss))
    res['test_acc'].append(float(total_correct/total_n_data))

    return prediction, outputs
