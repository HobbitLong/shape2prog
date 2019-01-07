from __future__ import print_function

import os
import h5py
import numpy as np
from programs.sample_blocks import sample_batch


def synthesize_data():
    """
    synthesize the (block, program) pairs
    :return: train_shape, train_prog, val_shape, val_prog
    """

    # == training data ==
    data = []
    label = []

    n_samples = [5000,
                 30000, 5000, 5000, 5000, 10000,
                 5000, 5000, 5000, 5000, 30000,
                 15000, 30000, 8000, 5000, 30000,
                 15000, 6000, 6000, 6000, 10000,
                 30000, 10000, 6000, 6000, 6000,
                 30000, 40000, 30000, 10000]

    for i in range(len(n_samples)):
        d, s = sample_batch(num=n_samples[i], primitive_type=i)
        data.append(d)
        label.append(s)

    n_samples = [30000, 30000, 40000, 30000, 30000,
                 30000, 20000, 10000, 40000, 35000,
                 40000, 35000, 35000, 35000, 50000]

    for i in range(len(n_samples)):
        d, s = sample_batch(num=n_samples[i], primitive_type=100+i+1)
        data.append(d)
        label.append(s)

    train_data = np.vstack(data)
    train_label = np.vstack(label)

    # == validation data ==
    data = []
    label = []

    for i in range(30):
        d, s = sample_batch(num=640, primitive_type=i)
        data.append(d)
        label.append(s)

    for i in range(15):
        d, s = sample_batch(num=640, primitive_type=100+i+1)
        data.append(d)
        label.append(s)

    val_data = np.vstack(data)
    val_label = np.vstack(label)

    return train_data, train_label, val_data, val_label


if __name__ == '__main__':

    print('==> synthesizing (part, block_program) pairs')
    train_x, train_y, val_x, val_y = synthesize_data()
    print('Done')

    if not os.path.isdir('./data'):
        os.makedirs('./data')
    train_file = './data/train_blocks.h5'
    val_file = './data/val_blocks.h5'

    print('==> saving data')

    f_train = h5py.File(train_file, 'w')
    f_train['data'] = train_x
    f_train['label'] = train_y
    f_train.close()

    f_val = h5py.File(val_file, 'w')
    f_val['data'] = val_x
    f_val['label'] = val_y
    f_val.close()

    print('Done')
