#!/bin/bash
#export MODEL="RARe-E5-Mistral-7B-Instruct-Ret-hf"
#export MODEL="RARe-Llama-3.1-8B-Instruct-LLM-hf"
#export MODEL="RARe-Llama-3-LLM-hf"
export MODEL="RARe-LLM2Vec-Llama-3-8B-Instruct-Ret-hf"
CUDA_VISIBLE_DEVICES=0 python3 run_eval.py --model_name_or_path $MODEL --task "ArguAna" --n_ic_examples 5 --batch_size 64
