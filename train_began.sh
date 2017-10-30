#!/bin/bash
python main.py \
--dataset=spatial-maps \
--data_dir=/home/data/world-cities/ \
--log_dir=/home/workspace/citygan/ \
--num_gpu=4 \
--comment="learn BEGAN model of urban form" \
--use_tensorboard=True \
--load_attributes="region,profiles" \
--src_names="bldg,pop,lum,water,bounds" \
--rotate_angle=10 \