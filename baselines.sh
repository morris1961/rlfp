#!/bin/bash
# check if dir log exists
if [ ! -d "logs" ]; then
  mkdir logs
fi
# This script is used to run the baseline model
python -u -W ignore baselines.py --model gpt-3.5 2>&1 | tee -i logs/gpt-3.5.log
python -u -W ignore baselines.py --model bard 2>&1 | tee -i logs/bard.log
python -u -W ignore baselines.py --model bard2 2>&1 | tee -i logs/bard2.log
python -u -W ignore baselines.py --model llama2 2>&1 | tee -i logs/llama2.log