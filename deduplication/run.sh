#!/bin/bash

GPUS="0"
export CUDA_VISIBLE_DEVICES=$GPUS

python dedup_preprocess_data.py

python dedup_exact.py

python dedup_fuzzy.py