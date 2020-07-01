#!/bin/bash

python ../training/prepare_spots_ivana.py \
    --path /work2/scratch/guacamole_data/Ivana/ \
    --basename ivana_spot \
    --test_split 0.1 \
    --valid_split 0.2 \
    --image_format tif \
    --label_format csv \
    --cell_size 4 \
