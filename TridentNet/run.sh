#!/bin/bash

source ~/hailing.env
./gnn/run.sh | tee -a log_train
