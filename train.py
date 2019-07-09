#!/usr/bin/env python3

import os

from data_loader.birds_loader import BirdsDataLoader

data_root = '/Users/luwin/Mycode/Text-to-Image-Synthesis/data/bird/'
image_dir = os.path.join(data_root, 'images')
text_dir = os.path.join(data_root, 'text_c10')
manifest_file = os.path.join(data_root, 'manifest.txt')
data_loader = BirdsDataLoader(image_dir, text_dir, manifest_file)

for item in data_loader:
    print(item)
    break
