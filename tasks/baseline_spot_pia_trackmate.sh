#!/bin/bash

python ../training/baseline_spot_pia_trackmate.py \
	--trackmate ../raw_data/pia_spots/ \
	--conversion 6.25 \
	--size 512 \
	--cell_size 4
