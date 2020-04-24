#!/bin/bash

python ../training/prepare_spots_pia.py \
    --path ../raw_data/pia_spots/ \
    --basename pia_spots \
    --conversion 6.25 \
    --test_split 0.1 \
    --valid_split 0.2 \
    --cell_size 4

