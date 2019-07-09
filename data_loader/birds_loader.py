#!/usr/bin/env python3

import os
import random
import torchfile
from torch.utils.data import Dataset, DataLoader, Sampler

from utils.util import onehot_encoder

class BirdsDataset(Dataset):
    """Birds dataset."""

    def __init__(self, image_dir, text_dir, manifest_file):
        """
        Args:
            image_dir (string): Path to the image files.
            text_dir (string): Path to the text files.
            manifest_file (string): Path to the manifest file.
        """
        self.image_dir = image_dir
        self.text_dir = text_dir

        def get_class_list(m_file):
            """Get all the class name from manifest file"""
            f = open(m_file, 'r')
            content = f.read()
            f.close()
            return content.strip().split('\n')

        self.class_list = get_class_list(manifest_file)

    def __len__(self):
        return len(self.class_list)

    def __getitem__(self, idx):
        class_name = self.class_list[idx]
        image_path = os.path.join(self.image_dir, class_name)
        image = torchfile.load(image_path)
        text_path = os.path.join(self.text_dir, class_name)
        text = torchfile.load(text_path)

        return {'image': image, 'text': text}

class BirdsBatchSampler(Sampler):
    """
    Args:
        data_source (Dataset): dataset to sample from.
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            class_idx_list = list(range(0, len(self.data_source)))
            random.shuffle(class_idx_list)
            yield class_idx_list[:self.batch_size]

    def __len__(self):
        if len(self.data_source) < self.batch_size:
            return len(self.data_source)
        else:
            return self.batch_size

def collate_fn(in_batch):
    out_batch = []
    for idx, class_data in enumerate(in_batch):
        label = idx

        image_idx = random.randint(0, class_data['image'].shape[0]-1)
        image_pos_idx = random.randint(0, class_data['image'].shape[2]-1)
        image = class_data['image'][image_idx,:,image_pos_idx]

        text_idx = random.randint(0, class_data['text'].shape[2]-1)
        raw_text = class_data['text'][image_idx,:,text_idx]
        text = onehot_encoder(raw_text)
        out_batch.append([idx, image, text])

    return out_batch


class BirdsDataLoader(DataLoader):
    """docstring for BirdsData."""
    def __init__(self, image_dir, text_dir, manifest_file):
        dataset = BirdsDataset(image_dir, text_dir, manifest_file)
        batch_sampler = BirdsBatchSampler(dataset, 40)
        super().__init__(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
