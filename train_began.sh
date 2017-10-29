#!/bin/bash
python main.py \
--dataset=world-cities \
--data_dir=/home/data/ \
--log_dir=/home/workspace/citygan/ \
--num_gpu=4 \
--comment="learn BEGAN model of urban form" \
--use_tensorboard=True