#!/bin/bash
python main.py \
--dataset=CelebA \
--data_dir=/home/data/ \
--log_dir=/home/workspace/citygan/ \
--num_gpu=4 \
--comment="testing work environment" \
--use_tensorboard=True \
--save_image_channels=False