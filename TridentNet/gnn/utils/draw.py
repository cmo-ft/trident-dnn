import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import json

def unsorted_segment_mean(data, segment_ids, num_segments):
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    Adapted from https://github.com/vgsatorras/egnn.
    '''
    data = torch.tensor(data).reshape(-1,1)
    segment_ids = torch.tensor(segment_ids)
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def simpleConvolve1d(signal:np.array, kernel:np.array, stride=1,method='mean')->np.array:
    # method: mean, sum
    # signal:[N], kernel:[M]. output: [(N-M+1)/stride]
    N, M = len(signal), len(kernel)
    # output = np.zeros( int((N-M+1)/stride) )
    if method=='mean':
        denom = M
    elif method=='sum':
        denom = 1
    else:
        print('error: method should be "mean" or "sum" ')
        exit()

    output = [(signal[M-1::-1] * kernel).sum()/denom]
    for isignal in range(stride,N,stride):
        sigSlice = signal[isignal+M-1:isignal-1:-1]
        if len(sigSlice)<len(kernel):
            continue
        output.append((sigSlice*kernel).sum()/denom)

    return np.array(output)

def draw_loss_acc(res, logDir):
    trainSliceId = np.unique(res['train_slice'])
    testSliceId = np.unique(res['test_slice'])
    # Draw loss with data set
    trainloss = res['train_loss']
    testloss = res['test_loss']
    # loss limit
    # ymax = max(np.max(trainloss), np.max(testloss))
    ymax = 500
    plt.figure(dpi=500)
    plt.plot(range(1,len(trainloss)+1), trainloss, label="train loss")
    plt.xlabel('set')
    plt.ylim(0, ymax)
    plt.legend()
    plt.savefig(logDir+"/loss_to_sets_train.png")
    
    trainloss = res['train_loss']
    testloss = res['test_loss']
    plt.figure(dpi=500)
    plt.plot(range(1,len(testloss)+1), testloss, label="test loss")
    plt.xlabel('set')
    plt.ylim(0, ymax)
    plt.legend()
    plt.savefig(logDir+"/loss_to_sets_test.png")

    # Draw loss with epoch
    trainloss = simpleConvolve1d(trainloss, np.ones(len(trainSliceId)), stride=len(trainSliceId), method='mean')
    testloss = simpleConvolve1d(testloss, np.ones(len(testSliceId)), stride=len(testSliceId), method='mean')
    plt.figure(dpi=500)
    plt.plot(range(1,len(trainloss)+1), trainloss, label="train loss")
    plt.plot(range(1,len(trainloss)+1), testloss, label="test loss")
    plt.xlabel('epoch')
    plt.ylim(0, ymax)
    plt.legend()
    plt.savefig(logDir+"/loss_to_epoch.png")

    # Draw acc with data set
    trainacc = res['train_acc']
    testacc = res['test_acc']
    # Acc limit
    ymax = max(np.max(trainacc), np.max(testacc))
    # ymax = 30
    plt.figure(dpi=500)
    plt.plot(range(1,len(trainacc)+1), trainacc, label="train acc")
    plt.xlabel('set')
    plt.ylim(0, ymax)
    plt.legend()
    plt.savefig(logDir+"/acc_to_sets_train.png")
    
    trainacc = res['train_acc']
    testacc = res['test_acc']
    plt.figure(dpi=500)
    plt.plot(range(1,len(testacc)+1), testacc, label="test acc")
    plt.xlabel('set')
    plt.ylim(0, ymax)
    plt.legend()
    plt.savefig(logDir+"/acc_to_sets_test.png")

    # Draw acc with epoch
    trainacc = simpleConvolve1d(trainacc, np.ones(len(trainSliceId)), stride=len(trainSliceId), method='mean')
    testacc = simpleConvolve1d(testacc, np.ones(len(testSliceId)), stride=len(testSliceId), method='mean')
    plt.figure(dpi=500)
    plt.plot(range(1,len(trainacc)+1), trainacc, label="train acc")
    plt.plot(range(1,len(trainacc)+1), testacc, label="test acc")
    plt.xlabel('epoch')
    plt.ylim(0, ymax)
    plt.legend()
    plt.savefig(logDir+"/acc_to_epoch.png")
    plt.cla()
    plt.close('all')




if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser(description="Draw loss and acc by Cen Mo")
    parser.add_argument('--in_file', type=str, default="./train-result.json", help='Trainning result: train-result.json')
    parser.add_argument('--out_dir', type=str, default="./", help='Output directory')
    args = parser.parse_args()

    # Get labels and predicts
    
    with open(args.in_file,'r') as f:
        res = json.load(f)
    
    draw_loss_acc(res, args.out_dir)



