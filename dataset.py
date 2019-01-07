from __future__ import print_function

from torch.utils.data import Dataset
import numpy as np
import h5py
from programs.label_config import num_params, max_param
from programs.loop_gen import translate, rotate, end


class PartPrimitive(Dataset):
    """
    dataset for (part, block program) pairs
    """
    def __init__(self, file_path):
        f = h5py.File(file_path, 'r')
        self.data = np.array(f['data'])
        self.labels = np.array(f['label'])

        assert self.data.shape[0] == self.labels.shape[0]

        self.num = self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index, :, 0]
        param = self.labels[index, :, 1:]

        data = data.astype(np.int64)
        label = label.astype(np.int64)
        param = param.astype(np.float32)

        return data, label, param

    def __len__(self):
        return self.num


class Synthesis3D(Dataset):
    """
    dataset for (shape, program) pairs
    """
    def __init__(self, file_path, n_block=6, n_step=3, w1=1, w2=1):
        f = h5py.File(file_path, 'r')
        self.data = np.array(f['data'])
        self.labels = np.array(f['label'])
        self.n_block = n_block
        self.n_step = n_step
        self.max_block = 0
        self.pgm_weight = w1
        self.param_weight = w2

        assert self.data.shape[0] == self.labels.shape[0]

        self.num = self.data.shape[0]

        self.pgms = np.zeros((self.num, self.n_block, self.n_step), dtype=np.int32)
        self.pgm_masks = np.zeros((self.num, self.n_block, self.n_step), dtype=np.float32)
        self.params = np.zeros((self.num, self.n_block, self.n_step, max_param-1), dtype=np.float32)
        self.param_masks = np.zeros((self.num, self.n_block, self.n_step, max_param-1), dtype=np.float32)

        for i in range(self.num):
            pgm, param, pgm_mask, param_mask = self.process_label(self.labels[i])
            self.pgms[i] = pgm
            self.pgm_masks[i] = pgm_mask
            self.params[i] = param
            self.param_masks[i] = param_mask

    def __getitem__(self, index):
        data = np.copy(self.data[index])

        pgm = self.pgms[index]
        pgm_mask = self.pgm_masks[index]
        param = self.params[index]
        param_mask = self.param_masks[index]

        data = data.astype(np.float32)
        pgm = pgm.astype(np.int64)

        return data, pgm, pgm_mask, param, param_mask

    def __len__(self):
        return self.num

    def process_label(self, label):
        pgm = np.zeros((self.n_block, self.n_step), dtype=np.int32)
        param = np.zeros((self.n_block, self.n_step, max_param - 1), dtype=np.float32)
        pgm_mask = 0.1 * np.ones((self.n_block, self.n_step), dtype=np.float32)
        param_mask = 0.1 * np.ones((self.n_block, self.n_step, max_param - 1), dtype=np.float32)

        max_step = label.shape[0]

        pgm_weight = self.pgm_weight
        param_weight = self.param_weight

        i = 0
        j = 0
        while j < max_step:
            if label[j, 0] == translate:
                if label[j+1, 0] == translate:
                    # pgm
                    pgm[i, 0] = translate
                    pgm[i, 1] = translate
                    pgm[i, 2] = label[j+2, 0]
                    pgm_mask[i, :3] = pgm_weight
                    # param
                    param[i, :3] = label[j:j+3, 1:]
                    param_mask[i, :2, :num_params[translate]] = param_weight
                    param_mask[i, 2, :num_params[pgm[i, 2]]] = param_weight
                    j = j + 5
                    i = i + 1
                else:
                    # pgm
                    pgm[i, 0] = translate
                    pgm[i, 1] = label[j+1, 0]
                    pgm_mask[i, :2] = pgm_weight
                    # param
                    param[i, :2] = label[j:j+2, 1:]
                    param_mask[i, 0, :num_params[translate]] = param_weight
                    param_mask[i, 1, :num_params[pgm[i, 1]]] = param_weight
                    j = j + 3
                    i = i + 1
            elif label[j, 0] == rotate:
                # pgm
                pgm[i, 0] = rotate
                pgm[i, 1] = label[j+1, 0]
                pgm_mask[i, :2] = pgm_weight
                # param
                param[i, :2] = label[j:j+2, 1:]
                param_mask[i, 0, :num_params[rotate]] = param_weight
                param_mask[i, 1, :num_params[pgm[i, 1]]] = param_weight
                j = j + 3
                i = i + 1
            elif label[j, 0] == end:
                j = j + 1
            elif label[j, 0] > 0:
                # pgm
                pgm[i, 0] = label[j, 0]
                pgm_mask[i, 0] = pgm_weight
                # param
                param[i, 0] = label[j, 1:]
                param_mask[i, 0, :num_params[pgm[i, 0]]] = param_weight
                j = j + 1
                i = i + 1
            else:
                break

            if i == self.n_block:
                print(label)

        if i > self.max_block:
            self.max_block = i

        return pgm, param, pgm_mask, param_mask


class ShapeNet3D(Dataset):
    """
    dataset for ShapeNet
    """
    def __init__(self, file_path):
        super(ShapeNet3D, self).__init__()

        f = h5py.File(file_path, "r")
        self.data = np.array(f['data'])
        self.num = self.data.shape[0]

    def __getitem__(self, index):
        data = np.copy(self.data[index, ...])
        data = data.astype(np.float32)

        return data

    def __len__(self):
        return self.num
