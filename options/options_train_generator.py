from __future__ import print_function

import os
import argparse
import socket
import torch

from programs.label_config import max_param, stop_id


def get_parser():
    """
    :return: return a parser which stores the arguments for training the program generator
    """
    parser = argparse.ArgumentParser(description="arguments for training program generator")

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='threshold for gradient clipping')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')

    # print and save
    parser.add_argument('--info_interval', type=int, default=10, help='freq for printing info')
    parser.add_argument('--save_interval', type=int, default=1, help='freq for saving model')

    # model parameters
    parser.add_argument('--program_size', type=int, default=stop_id-1, help='number of programs')
    parser.add_argument('--max_param', type=int, default=max_param-1, help='maximum number of parameters')
    parser.add_argument('--shape_feat_size', type=int, default=64, help='dimension of CNN shape vector')

    parser.add_argument('--outer_input_size', type=int, default=64, help='input dim of block LSTM')
    parser.add_argument('--outer_rnn_size', type=int, default=64, help='core dim of block LSTM')
    parser.add_argument('--outer_num_layers', type=int, default=1, help='number of layers of block LSTM')
    parser.add_argument('--outer_drop_prob', type=float, default=0, help='dropout prob of block LSTM')
    parser.add_argument('--outer_seq_length', type=int, default=10, help='length of block LSTM')

    parser.add_argument('--inner_input_size', type=int, default=64, help='input dim of step LSTM')
    parser.add_argument('--inner_rnn_size', type=int, default=64, help='core dim of step LSTM')
    parser.add_argument('--inner_num_layers', type=int, default=1, help='number of layers of step LSTM')
    parser.add_argument('--inner_drop_prob', type=float, default=0, help='dropout prob of step LSTM')
    parser.add_argument('--inner_seq_length', type=int, default=3, help='length of step LSTM')
    parser.add_argument('--inner_cls_feat_size', type=int, default=64, help='dim of feat for program id cls')
    parser.add_argument('--inner_reg_feat_size', type=int, default=64, help='dim of feat for program parameter reg')
    parser.add_argument('--inner_sample_prob', type=float, default=1.1, help='prob to sample rather gt (1.1->no gt)')

    # weight balance
    parser.add_argument('--cls_weight', type=int, default=1, help='program classification weight')
    parser.add_argument('--reg_weight', type=int, default=3, help='parameter regression weight')

    # data parameter
    parser.add_argument('--train_file', type=str, default='./data/train_shapes.h5', help='path to training file')
    parser.add_argument('--val_file', type=str, default='./data/val_shapes.h5', help='path to val file')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of training and validating')
    parser.add_argument('--num_workers', type=int, default=4, help='num of threads for data loader')
    parser.add_argument('--model_name', type=str, default='program_generator', help='folder name to save model')

    return parser


def parse():

    parser = get_parser()
    opt = parser.parse_args()

    opt.save_folder = os.path.join('./model', 'ckpts_{}'.format(opt.model_name))

    opt.is_cuda = torch.cuda.is_available()

    return opt


if __name__ == '__main__':

    opt = parse()

    print('===== arguments: training program generator =====')

    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))
