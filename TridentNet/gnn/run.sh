#! /bin/bash

# Configure input data 
#filePrefix="../data_process/data/"
filePrefix="/lustre/collider/mocen/project/hailing/machineLearning/data/shower/100tev/"

num_epochs=100
num_slices_train=1398
num_slices_test=50
num_slices_apply=50
lr=0.01
batch_size=400

# Apply only
apply_only=0

# Pre-train
pre_train=0
pre_net="./net.pt"
pre_log="./train-result.json"


python gnn/main.py --fileList "${filePrefix}" \
                --data_size $data_size --num_slices_train $num_slices_train --num_slices_test $num_slices_test \
                --num_slices_apply $num_slices_apply --num_classes $nClasses --num_channels $nChannels \
                --apply_only $apply_only  --pre_train $pre_train  --pre_net $pre_net  --pre_log $pre_log --num_epochs=$num_epochs \
                --lr $lr --batch_size $batch_size
