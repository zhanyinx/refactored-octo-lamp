#!/bin/bash

python ../training/baseline_spot_pia_trackmate.py \
	--dataset ../data/pia_spots_2486151f.npz \
	--size 512 \
	--cell_size 4
