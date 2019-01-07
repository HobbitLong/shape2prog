from __future__ import print_function

import os
import argparse
import socket
import torch


def get_parser():
    """
    a parser for Guided Adaptation
    """
    parser = argparse.ArgumentParser(description="arguments for Guided Adaptation")

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.00002, help='learning rate for GA')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--epochs', type=int, default=20, help='epochs for GA')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='gradient clip threshold')

    # print freq
    parser.add_argument('--info_interval', type=int, default=10, help='freq for printing info')
    parser.add_argument('--save_interval', type=int, default=5, help='freq for saving model')

    # data info
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of GA')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--data_folder', type=str, default='./data/', help='directory to data')
    parser.add_argument('--cls', type=str, default='chair',
                        help='furniture classes: chair, table, bed, sofa, cabinet, bench')

    # model info
    parser.add_argument('--p_gen_path', type=str, default='./model/ckpts_program_generator/program_generator.t7',
                        help='path to the program generator')
    parser.add_argument('--p_exe_path', type=str, default='./model/ckpts_program_executor/program_executor.t7',
                        help='path to the program executor')
    parser.add_argument('--model_name', type=str, default='GA', help='folder name to save model')

    return parser


def parse():
    """
    parse and modify the options accordingly
    """
    parser = get_parser()
    opt = parser.parse_args()

    opt.save_folder = os.path.join('./model', 'ckpts_{}_{}'.format(opt.model_name, opt.cls))

    opt.train_file = os.path.join(opt.data_folder, '{}_training.h5'.format(opt.cls))
    opt.val_file = os.path.join(opt.data_folder, '{}_testing.h5'.format(opt.cls))

    if opt.cls in ['chair', 'table']:
        pass
    elif opt.cls in ['sofa', 'cabinet', 'bench']:
        opt.epochs = max(60, opt.epochs)
    elif opt.cls in ['bed']:
        opt.learning_rate = 0.0001
        opt.epochs = max(150, opt.epochs)
    else:
        raise NotImplementedError('{} is invalid class'.format(opt.cls))

    opt.is_cuda = torch.cuda.is_available()

    return opt


if __name__ == '__main__':

    opt = parse()

    print('===== arguments: guided adaptation =====')

    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))


