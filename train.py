#!/usr/bin/env python3

import os

from data_loader.birds_loader import BirdsDataLoader

data_root = '/Users/rhonin/Desktop/Github/Learning_Deep_Representations/cvpr2016_cub'
image_dir = os.path.join(data_root, 'images')
text_dir = os.path.join(data_root, 'text_c10')
manifest_file = os.path.join(data_root, 'manifest.txt')
data_loader = BirdsDataLoader(image_dir, text_dir, manifest_file)
iterations = 10

ite = 0
while ite < iterations:
    data = next(iter(data_loader))
    print(data[2][2].shape)
    ite += 1
"""
for item in data_loader:
    print(item)
    break
"""
