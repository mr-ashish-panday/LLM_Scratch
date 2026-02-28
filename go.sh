#!/bin/bash
pip install transformers datasets bitsandbytes
python -u run_ablation.py --mode width_only --max_steps 3000 --batch_size 128 --grad_accum 1 --lambda-cost 0.005 --penalty-warmup 1000 2>&1 | tee width_only_train.log
