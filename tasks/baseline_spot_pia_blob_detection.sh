#!/bin/bash

python ../training/baseline_spot_pia_trackmate.py \
	--dataset ../data/pia_spots_caefebf6.npz \
	--size 512 \
	--cell_size 4
