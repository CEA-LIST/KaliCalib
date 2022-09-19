#!/bin/sh

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python src/train.py --config_file configs/train_sviewds_full_dataset.yml
