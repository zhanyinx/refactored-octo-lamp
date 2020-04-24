#!/bin/bash

python ../training/baseline_spot_pia_trackmate.py \
	--dataset ../data/pia_spots_9ee3da55.npz \
	--size 512 \
	--cell_size 4
