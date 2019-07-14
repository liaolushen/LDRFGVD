#!/usr/bin/env python3
import numpy as np

def onehot_encoder(input, onehot_len=71):
    output = np.eye(onehot_len)[[int(i) for i in input]]
    return output
