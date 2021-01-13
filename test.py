from __future__ import print_function

import os
import argparse
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import h5py
from torch.utils.data import DataLoader
from torch.autograd import Variable

from visualization.util_vtk import visualization
from dataset import ShapeNet3D
from model import BlockOuterNet
from criterion import BatchIoU
from misc import decode_multiple_block, execute_shape_program
from interpreter import Interpreter
from programs.loop_gen import translate, rotate, end

import socket


def parse_argument():

    parser = argparse.ArgumentParser(description="testing the program generator")

    parser.add_argument('--model', type=str, default='./model/program_generator_GA_bed.t7',
                        help='path to the testing model')
    parser.add_argument('--data', type=str, default='./data/bed_testing.h5',
                        help='path to the testing data')
    parser.add_argument('--save_path', type=str, default='./output/bed/',
                        help='path to save the output results')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--info_interval', type=int, default=10, help='freq for printing info')

    parser.add_argument('--save_prog', action='store_true', help='save programs to text file')
    parser.add_argument('--save_img', action='store_true', help='render reconstructed shapes to images')
    parser.add_argument('--num_render', type=int, default=10, help='how many samples to be rendered')

    opt = parser.parse_args()

    opt.prog_save_path = os.path.join(opt.save_path, 'programs')
    opt.imgs_save_path = os.path.join(opt.save_path, 'images')

    opt.is_cuda = torch.cuda.is_available()

    return opt


def test_on_shapenet_data(epoch, test_loader, model, opt, gen_shape=False):

    model.eval()
    generated_shapes = []
    original_shapes = []
    gen_pgms = []
    gen_params = []

    for idx, data in enumerate(test_loader):
        start = time.time()

        shapes = data
        shapes = Variable(torch.unsqueeze(shapes, 1), requires_grad=False).cuda()

        out = model.decode(shapes)

        if opt.is_cuda:
            torch.cuda.synchronize()
        end = time.time()

        if gen_shape:
            generated_shapes.append(decode_multiple_block(out[0], out[1]))
            original_shapes.append(data.clone().numpy())
            _, save_pgms = torch.max(out[0].data, dim=3)
            save_pgms = save_pgms.cpu().numpy()
            save_params = out[1].data.cpu().numpy()
            gen_pgms.append(save_pgms)
            gen_params.append(save_params)

        if idx % opt.info_interval == 0:
            print("Test: epoch {} batch {}/{}, time={:.3f}".format(epoch, idx, len(test_loader), end - start))

    if gen_shape:
        generated_shapes = np.concatenate(generated_shapes, axis=0)
        original_shapes = np.concatenate(original_shapes, axis=0)
        gen_pgms = np.concatenate(gen_pgms, axis=0)
        gen_params = np.concatenate(gen_params, axis=0)

    return original_shapes, generated_shapes, gen_pgms, gen_params


def run():
    opt = parse_argument()

    if not os.path.isdir(opt.prog_save_path):
        os.makedirs(opt.prog_save_path)
    if not os.path.isdir(opt.imgs_save_path):
        os.makedirs(opt.imgs_save_path)

    print('========= arguments =========')
    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))
    print('========= arguments =========')

    # data loader
    test_set = ShapeNet3D(opt.data)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )

    # model
    ckpt = torch.load(opt.model)
    model = BlockOuterNet(ckpt['opt'])
    model.load_state_dict(ckpt['model'])
    if opt.is_cuda:
        model = model.cuda()
        cudnn.benchmark = True

    # test the model and evaluate the IoU
    ori_shapes, gen_shapes, pgms, params = test_on_shapenet_data(epoch=0,
                                                                 test_loader=test_loader,
                                                                 model=model,
                                                                 opt=opt,
                                                                 gen_shape=True)
    IoU = BatchIoU(ori_shapes, gen_shapes)
    print("Mean IoU: {:.3f}".format(IoU.mean()))

    # execute the generated program to generate the reconstructed shapes
    # for double-check purpose, can be disabled
    num_shapes = gen_shapes.shape[0]
    res = []
    for i in range(num_shapes):
        data = execute_shape_program(pgms[i], params[i])
        res.append(data.reshape((1, 32, 32, 32)))
    res = np.concatenate(res, axis=0)
    IoU_2 = BatchIoU(ori_shapes, res)

    assert abs(IoU.mean() - IoU_2.mean()) < 0.1, 'IoUs are not matched'

    # save results
    save_file = os.path.join(opt.save_path, 'shapes.h5')
    f = h5py.File(save_file, 'w')
    f['data'] = gen_shapes
    f['pgms'] = pgms
    f['params'] = params
    f.close()

    # Interpreting programs to understandable program strings
    if opt.save_prog:
        interpreter = Interpreter(translate, rotate, end)
        num_programs = gen_shapes.shape[0]
        for i in range(min(num_programs, opt.num_render)):
            # pgms[0]: 10 x 3, params[0]: 10 x 3 x 7
            print(pgms[0], params[0])
            program = interpreter.interpret(pgms[i], params[i])
            save_file = os.path.join(opt.prog_save_path, '{}.txt'.format(i))
            with open(save_file, 'w') as out:
                out.write(program)

    # Visualization
    if opt.save_img:
        data = ori_shapes.transpose((0, 3, 2, 1))
        data = np.flip(data, axis=2)
        num_shapes = data.shape[0]
        for i in range(min(num_shapes, opt.num_render)):
            voxels = data[i]
            save_name = os.path.join(opt.imgs_save_path, '{}.png'.format(i))
            visualization(voxels,
                          threshold=0.1,
                          save_name=save_name,
                          uniform_size=0.9)


if __name__ == '__main__':
    run()
