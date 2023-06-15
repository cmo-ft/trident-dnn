import fileinput
import gc
import os
import pathlib
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch
import random
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import copy
import time

import config
from config import args
import myDataset
from utils.confusionMatrix import get_confusion_matrix
import utils.draw as draw

if __name__ == '__main__':
    # Initialization is completed in config.py
    timeProgramStart = time.time()
    if args.distributed:
        import train_test
    else:
        import train_test

    #Find device
    device = args.device
    
    # Initialize network, result log and optimizer
    net = copy.deepcopy(args.net)
    res = copy.deepcopy(args.reslog)
    args.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer = args.optimizer
    args.reduce_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(args.optimizer, mode='min', factor=0.5, patience=15,
                 verbose=False, threshold=0.1, threshold_mode='rel',
                 cooldown=0, min_lr=1e-6, eps=1e-8)

    epoch_start = 1

    if args.pre_train != 0 and (not args.apply_only):
        net.load_state_dict(torch.load(args.pre_net, map_location=device))
        
        try:
            with open(args.pre_log,'r') as f:
                res = json.load(f)
            epoch_start = res['epochs'][-1]+1
        except:
            pass
            
    
    net.to(device)
    if args.distributed:
        net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Loss function 
    criterion = args.criterion

    # Record the smallest loss epoch
    bestLoss = 1e10
    bestEpoch = 0

    if args.apply_only == 0:
        # Loop epoch 
        for epoch in range(epoch_start, epoch_start + args.num_epochs):
            # Optimizer

            # Record test pred
            testpred = np.zeros(0)
            testOut = np.zeros(0)

            # Loop slices of data
            meanTrainLoss, meanTrainAcc = 0, 0
            meanLoss, meanAcc = 0, 0
            for data_slice_id in range(args.num_slices_train):
                # continue
                trainloader = myDataset.get_dataloader('train',data_slice_id, args.num_slices_train, args.data_size, args.batch_size)                
                if len(trainloader)==0:
                    continue
                # Log epoch and lr
                res['epochs'].append(epoch)
                res['lr'].append(optimizer.param_groups[0]['lr'])

                # Train
                res['train_slice'].append(data_slice_id)
                train_test.train_one_epoch(net, trainloader, criterion, optimizer, res)

                meanTrainLoss = meanTrainLoss + res['train_loss'][-1]
                meanTrainAcc = meanTrainAcc + res['train_acc'][-1]

                # Time usage
                if args.rank==0:
                    if (data_slice_id+1) in [int(i/4*args.num_slices_train) for i in range(1,5)]:
                        print("Epoch %d/%d with lr %f, data_slice %d/%d training finished in %.2f min. Total time used: %.2f min." \
                                % (epoch, epoch_start+args.num_epochs-1, optimizer.param_groups[0]['lr'], data_slice_id+1, args.num_slices_train,\
                                    (res['train_time'][-1])/60, (time.time() - timeProgramStart)/60), flush=True)
                del trainloader
                torch.cuda.empty_cache()
                
            # Test
            for data_slice_id in range(args.num_slices_test):
                res['test_slice'].append(data_slice_id)

                testloader = myDataset.get_dataloader('test',data_slice_id, args.num_slices_test, args.data_size, args.batch_size)
                ifcheckOutput = ( args.rank==0 and data_slice_id==args.num_slices_test-1 )
                pred_tmp, out_tmp = train_test.test_one_epoch(net, testloader, criterion, res, check_output = ifcheckOutput, save_new_graph_loc="./out/test/", save_new_graph_id=data_slice_id)

                if len(testpred) == 0:
                    testpred = pred_tmp
                    testOut = out_tmp
                else:
                    testpred = np.concatenate((testpred, pred_tmp))
                    testOut = np.concatenate((testOut, out_tmp))

                meanLoss = meanLoss + res['test_loss'][-1]
                meanAcc = meanAcc + res['test_acc'][-1]
                # Save result
                if args.rank==0:
                    json_object = json.dumps(res, indent=4)
                    with open(f"{args.logDir}/train-result.json", "w") as outfile:
                        outfile.write(json_object)
                del testloader
                torch.cuda.empty_cache()

            meanTrainLoss = meanTrainLoss / args.num_slices_train
            meanTrainAcc = meanTrainAcc / args.num_slices_train
            meanLoss = meanLoss / args.num_slices_test
            meanAcc = meanAcc / args.num_slices_test
            args.reduce_schedule.step(meanLoss)
            if args.rank==0:
                print(f"Test: epoch: {epoch}/{epoch_start+args.num_epochs-1}.")
                print(f"mean train loss: {meanTrainLoss}, mean train acc: {meanTrainAcc}")
                print(f"mean test loss: {meanLoss}, mean test acc: {meanAcc}")
                print("Total time used: %.2f min.\n"%((time.time() - timeProgramStart)/60))
                
                # draw.draw_loss_acc(res, args.logDir)

                # Record the best epoch
                if meanLoss < bestLoss:
                    bestLoss = meanLoss
                    bestEpoch = epoch
                    torch.save(net.state_dict(), f'{args.logDir}/net.pt')

                if True:
                    # save pred given by network during test
                    np.save(args.logDir+f'/predTest_GPU{args.rank}.npy', arr=testpred)
                    np.save(args.logDir+f'/outTest_GPU{args.rank}.npy', arr=testOut)
                    if args.distributed:
                        torch.save(net.module.state_dict(), f'{args.logDir}/net.pt')
                    else:
                        torch.save(net.state_dict(), f'{args.logDir}/net{epoch}.pt')

                print(f"Best epoch: {bestEpoch} with loss {bestLoss}")
                print("\n\n", flush=True)
            if args.distributed:
                dist.barrier()

    # Apply
    net = copy.deepcopy(args.net)
    if args.distributed:
    # Use a barrier() to make sure that process 1 loads the model after process 0 saves it.
        dist.barrier()
        net.load_state_dict(torch.load(f'{args.logDir}/net.pt', map_location=device))
        net.to(device)
        net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        net.load_state_dict(torch.load(f'{args.logDir}/net.pt', map_location=device))
        net.to(device)
        
    applyRes = args.reslog.copy()
    pred = np.zeros(0)
    out = np.zeros(0)
    for data_slice_id in range(args.num_slices_apply):
        applyloader = myDataset.get_dataloader('apply',data_slice_id, args.num_slices_test, args.data_size, args.batch_size)
             
        pred_tmp, out_tmp = train_test.test_one_epoch(net, applyloader, criterion, applyRes, save_new_graph_loc="./out/", save_new_graph_id=data_slice_id)
        if len(pred) == 0:
            pred = pred_tmp
            out = out_tmp
        else:
            pred = np.concatenate((pred, pred_tmp))
            out = np.concatenate((out, out_tmp))

    # save pred given by network during applying
    np.save(args.logDir+f'/predApply_GPU{args.rank}.npy', arr=pred)
    np.save(args.logDir+f'/outApply_GPU{args.rank}.npy', arr=out)
    if args.rank==0:
        print("\n\n")
        print("Apply finished.")
        print("Apply time %.2f min" % (sum(applyRes['test_time'])/60.))
        print("Apply loss: %.4f \t Apply acc: %.4f" % (sum(applyRes['test_loss'])/len(applyRes['test_loss']),
                                                        sum(applyRes['test_acc'])/len(applyRes['test_acc'])))        
        print("\nTotal time used: %.2f min.\n"%((time.time() - timeProgramStart)/60))
        

