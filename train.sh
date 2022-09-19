#!/bin/sh

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python kalicalib/train.py --config_file configs/train_sviewds.yml
