'''For viewing testing data and corresponding generate voxels'''

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py
import argparse
from visualization.util_vtk import visualization
from dataset import ShapeNet3D

def parse_argument():

    parser = argparse.ArgumentParser(description="testing the program generator")

    parser.add_argument('--target', type=str, default='./data/bed_testing.h5',
                        help='path to the testing data')
    parser.add_argument('--output', type=str, default='./output/bed/',
                        help='path to save the output results')
    parser.add_argument('-s', '--start', type=int, default=0,
            help='start samples to be rendered')
    parser.add_argument('-e', '--end', type=int, default=10,
            help='end samples to be rendered')
    parser.add_argument('-o', action="store_true", help="Check the target dataset or not")

    opt = parser.parse_args()


    opt.imgs_save_path = os.path.join(opt.output, "images")
    opt.output = os.path.join(opt.output, "shapes.h5")
    opt.is_cuda = torch.cuda.is_available()

    # Whether to view target/original voxel or generated voxel
    if opt.o:
        opt.path = opt.target
    else:
        opt.path = opt.output

    return opt

def voxel_from_data(data_loader):
    '''Convert from original format to voxels'''
    for i, data in enumerate(data_loader):
        print("test data",data.shape)
    
def run():

    opt = parse_argument()

    print('========= arguments =========')
    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))
    print('========= arguments =========')

    if opt.o:
        # Load target data
        f = h5py.File(opt.path, "r")
        shapes = np.array(f['data'])
    else: 
        # Load output/rendered data
        f = h5py.File(opt.path, "r")
        shapes = np.array(f['data'])
        pgms = np.array(f['pgms'])
        params = np.array(f['params'])

    # Visualization
    data = shapes.transpose((0, 3, 2, 1))
    data = np.flip(data, axis=2)
    num_shapes = data.shape[0]

    num_gen = abs(opt.end - opt.start)
    if num_shapes < num_gen:
        opt.start = 0
        opt.end = num_shapes

    for i in range(opt.start, opt.end):
        voxels = data[i]

        # Display generated
        prefix = "original" if opt.o else "generated"
        save_name = os.path.join(opt.imgs_save_path, '{}_{}.png'.format(prefix, i))
        visualization(voxels,
                      threshold=0.1,
                      save_name=save_name,
                      uniform_size=0.9)


if __name__ == "__main__":
    run()

