#!/bin/bash
python ../training/prepare_spots_synt.py \
    --path '../../nogithub_folders/syntetic_images_harder/'\
    --basename 'spots_synt_harder'\
    --image_format tif \
    --label_format csv \
    --cell_size 4\
