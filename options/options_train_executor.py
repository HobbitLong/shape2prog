from __future__ import print_function

import os
import argparse
import socket
import torch

from programs.label_config import max_param, stop_id


def get_parser():
    """
    a parser for training the program executor
    """
    parser = argparse.ArgumentParser(description="arguments for training program executor")

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='20,25', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='threshold for gradient clipping')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')

    # print and save
    parser.add_argument('--info_interval', type=int, default=10, help='freq for printing info')
    parser.add_argument('--save_interval', type=int, default=1, help='freq for saving model')

    # model parameters
    parser.add_argument('--program_size', type=int, default=stop_id-1, help='number of programs')
    parser.add_argument('--input_encoding_size', type=int, default=128, help='dim of input encoding')
    parser.add_argument('--program_vector_size', type=int, default=128, help='dim of program encoding')
    parser.add_argument('--nc', type=int, default=2, help='number of output channels')
    parser.add_argument('--rnn_size', type=int, default=128, help='core dim of aggregation LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='number of LSTM layers')
    parser.add_argument('--drop_prob_lm', type=float, default=0, help='dropout prob of LSTM')
    parser.add_argument('--seq_length', type=int, default=3, help='sequence length')
    parser.add_argument('--max_param', type=int, default=max_param-1, help='maximum number of parameters')

    # data parameter
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training and validating')
    parser.add_argument('--num_workers', type=int, default=8, help='num of threads for data loader')
    parser.add_argument('--train_file', type=str, default='./data/train_blocks.h5', help='path to training file')
    parser.add_argument('--val_file', type=str, default='./data/val_blocks.h5', help='path to val file')
    parser.add_argument('--model_name', type=str, default='program_executor', help='folder name to save model')

    # weighted loss
    parser.add_argument('--n_weight', type=int, default=1, help='weight for negative voxels')
    parser.add_argument('--p_weight', type=int, default=5, help='weight for positive voxels')

    # randomization file for validation
    parser.add_argument('--rand1', type=str, default='./data/rand1.npy', help='directory to rand file 1')
    parser.add_argument('--rand2', type=str, default='./data/rand2.npy', help='directory to rand file 2')
    parser.add_argument('--rand3', type=str, default='./data/rand3.npy', help='directory to rand file 3')

    return parser


def parse():

    parser = get_parser()
    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.save_folder = os.path.join('./model', 'ckpts_{}'.format(opt.model_name))

    opt.is_cuda = torch.cuda.is_available()
    opt.num_gpu = torch.cuda.device_count()

    return opt


if __name__ == '__main__':

    opt = parse()

    print('===== arguments: training program executor =====')

    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))
