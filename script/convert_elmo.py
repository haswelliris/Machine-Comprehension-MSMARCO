import cntk as C
from cntk.layers import *
import numpy as np
import h5py

class Elmo(object):
    def __init__(self):
        self.weight_file='elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

    def _load_weight(self):
        with h5py.File(self.weight_file, 'r') as f:
            pass
