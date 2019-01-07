from __future__ import print_function

import sys
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import Synthesis3D
from model import BlockOuterNet
from criterion import LSTMClassCriterion, LSTMRegressCriterion
from misc import clip_gradient, decode_to_shape_new
from options import options_train_generator


def train(epoch, train_loader, model, crit_cls, crit_reg, optimizer, opt):
    """
    One epoch training
    """
    model.train()
    crit_cls.train()
    crit_reg.train()

    cls_w = opt.cls_weight
    reg_w = opt.reg_weight

    # the prob: > 1
    # the input of step t is always sampled from the output of step t-1
    sample_prob = opt.inner_sample_prob

    for idx, data in enumerate(train_loader):
        start = time.time()

        shapes, labels, masks, params, param_masks = data[0], data[1], data[2], data[3], data[4]
        shapes = torch.unsqueeze(shapes, 1)

        if opt.is_cuda:
            shapes = shapes.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
            params = params.cuda()
            param_masks = param_masks.cuda()

        optimizer.zero_grad()

        out = model(shapes, labels, sample_prob)

        # reshape
        bsz, n_block, n_step = labels.size()
        labels = labels.contiguous().view(bsz, n_block * n_step)
        masks = masks.contiguous().view(bsz, n_block * n_step)
        out_pgm = out[0].view(bsz, n_block * n_step, opt.program_size + 1)

        bsz, n_block, n_step, n_param = params.size()
        params = params.contiguous().view(bsz, n_block * n_step, n_param)
        param_masks = param_masks.contiguous().view(bsz, n_block * n_step, n_param)
        out_param = out[1].view(bsz, n_block * n_step, n_param)

        loss_cls, acc = crit_cls(out_pgm, labels, masks)
        loss_reg = crit_reg(out_param, params, param_masks)
        loss = cls_w * loss_cls + reg_w * loss_reg
        loss.backward()

        clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()

        if opt.is_cuda:
            torch.cuda.synchronize()

        end = time.time()

        if idx % opt.info_interval == 0:
            print("Train: epoch {} batch {}/{}, loss_cls = {:.3f}, loss_reg = {:.3f}, acc = {:.3f}, time = {:.3f}"
                  .format(epoch, idx, len(train_loader), loss_cls.data[0], loss_reg.data[0], acc.data[0], end - start))
            sys.stdout.flush()


def validate(epoch, val_loader, model, crit_cls, crit_reg, opt, gen_shape=False):
    """
    One validation
    """
    model.eval()
    crit_cls.eval()
    crit_reg.eval()

    generated_shapes = []

    for idx, data in enumerate(val_loader):
        start = time.time()

        shapes, labels, masks, params, param_masks = data[0], data[1], data[2], data[3], data[4]
        shapes = torch.unsqueeze(shapes, 1)

        if opt.is_cuda:
            shapes = shapes.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
            params = params.cuda()
            param_masks = param_masks.cuda()

        out = model.decode(shapes)

        # reshape
        bsz, n_block, n_step = labels.size()
        labels = labels.contiguous().view(bsz, n_block * n_step)
        masks = masks.contiguous().view(bsz, n_block * n_step)
        out_pgm = out[0].view(bsz, n_block * n_step, opt.program_size + 1)

        bsz, n_block, n_step, n_param = params.size()
        params = params.contiguous().view(bsz, n_block * n_step, n_param)
        param_masks = param_masks.contiguous().view(bsz, n_block * n_step, n_param)
        out_param = out[1].view(bsz, n_block * n_step, n_param)

        loss_cls, acc = crit_cls(out_pgm, labels, masks)
        loss_reg = crit_reg(out_param, params, param_masks)

        if opt.is_cuda:
            torch.cuda.synchronize()

        end = time.time()

        if gen_shape:
            generated_shapes.append(decode_to_shape_new(out[0], out[1]))

        if idx % opt.info_interval == 0:
            print("Test: epoch {} batch {}/{}, loss_cls = {:.3f}, loss_reg = {:.3f}, acc = {:.3f}, time = {:.3f}"
                  .format(epoch, idx, len(val_loader), loss_cls.data[0], loss_reg.data[0], acc.data[0], end - start))
            sys.stdout.flush()

    if gen_shape:
        generated_shapes = np.concatenate(generated_shapes, axis=0)

    return generated_shapes


def run():

    opt = options_train_generator.parse()

    print('===== arguments: program generator =====')
    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))
    print('===== arguments: program generator =====')

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # build dataloader
    train_set = Synthesis3D(opt.train_file, n_block=opt.outer_seq_length)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    val_set = Synthesis3D(opt.val_file, n_block=opt.outer_seq_length)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )

    # build model
    model = BlockOuterNet(opt)
    crit_cls = LSTMClassCriterion()
    crit_reg = LSTMRegressCriterion()
    if opt.is_cuda:
        model = model.cuda()
        crit_cls = crit_cls.cuda()
        crit_reg = crit_reg.cuda()
        cudnn.benchmark = True

    optimizer = optim.Adam(model.parameters(),
                           lr=opt.learning_rate,
                           betas=(opt.beta1, opt.beta2),
                           weight_decay=opt.weight_decay)

    for epoch in range(1, opt.epochs+1):
        print("###################")
        print("training")
        train(epoch, train_loader, model, crit_cls, crit_reg, optimizer, opt)
        print("###################")
        print("testing")
        validate(epoch, val_loader, model, crit_cls, crit_reg, opt)

        if epoch % opt.save_interval == 0:
            print('Saving...')
            state = {
                'opt': opt,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.t7'.format(epoch=epoch))
            torch.save(state, save_file)

    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': opt.epochs
    }
    save_file = os.path.join(opt.save_folder, 'program_generator.t7')
    torch.save(state, save_file)


if __name__ == '__main__':
    run()
