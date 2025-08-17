#!/bin/bash

GPUS="0,1"
export CUDA_VISIBLE_DEVICES=$GPUS

echo "*** CARREGANDO OS DATASETS ***"
python dedup_preprocess_data.py

echo "*** DEDUPLICAÇÃO EXATA ***"
python dedup_exact.py

echo "*** DEDUPLICAÇÃO FUZZY ***"
python dedup_fuzzy.py

echo "*** LIMPANDO COM BASE EM HEURÍSTICAS ***"
python clean_heuristics.py
