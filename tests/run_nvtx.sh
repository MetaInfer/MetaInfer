#!/bin/bash
eval "$(/home/honglin/miniconda3/bin/conda shell.bash hook)"
conda activate meta
exec python /home/honglin/meta-infer/tests/profile_vllm_nvtx.py
