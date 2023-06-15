# Train and Apply TridentNet
To train and apply your own TridentNet, you should modify the file ```gnn/run.sh```. The variables in the script is understand as follows:

| variable | meaning|
|---|---|
| filePrefix | data file for training & applying |
| num_epochs | how many times the network can see the training data |
| num_slices_ | numbers of train, test and apply samples |
| lr | starting learning rate |
| batch_size | number of events in a single batch. too small makes performance worse, too large may exceeds storage capacity |
| apply_only | if you only want to apply the model, make this variable to be 1|
| pre_train | if you want to train a pre-trained model, specify pre_net, pre_log and make pre_train to be 1|
| pre_net | the pre-trained model|
| pre_log | logs for pre-training |

If you want to furture modify the scripts to complete your own task, you can:

* modify ```gnn/conig``` to change the architecture of TridentNet, including the input format, output format, layers of ParticleDynamicEdgeConv block and so on. 
* modify ```gnn/myDataset.py``` to change the data that can be seen by TridentNet.
* modify ```gnn/train_test.py``` to change the details of training.