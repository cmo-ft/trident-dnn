from sklearn.metrics import confusion_matrix
import numpy as np
import argparse

# get confusion matrix from scores with shape [ndata, nclass+1]
def get_confusion_matrix(scores):
    labels = scores[:,0]
    scores = scores[:,1:]
    predict = np.argmax(scores, 1)
    # Get confusion matrix
    return confusion_matrix(labels, predict)


def custom_formatter(x):
    return f'{x:.2f}'


if __name__ == '__main__':
    # Set the formatter to the custom formatter function
    # np.set_printoptions(precision=3) #设置小数位置为3位
    np.set_printoptions(formatter={'float_kind': custom_formatter})
    
    # Input arguments
    parser = argparse.ArgumentParser(description="Fetch confusion matrix from apply result by Cen Mo")
    parser.add_argument('--in_file', type=str, default="./scoreApply.npy", help='Apply result: scoresApply.npy')
    parser.add_argument('--out_dir', type=str, default="./", help='Output directory')
    args = parser.parse_args()

    # Get labels and predicts
    scores = np.load(args.in_file)

    cm = get_confusion_matrix(scores)
    print("Confusion matrix:")
    print(cm)
    print("\n\n")

    print("ratio: ")
    for truth in cm:
        print(truth/truth.sum())
    
    # Save confusion matrix
    np.save(args.out_dir+'/confusion_matrix.npy', arr=cm)



