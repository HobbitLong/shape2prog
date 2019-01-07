from __future__ import print_function

import sys
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import PartPrimitive
from model import RenderNet
from criterion import BatchIoU
from misc import clip_gradient
from programs.label_config import max_param, stop_id
from options import options_train_executor


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch >= np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def train(epoch, train_loader, model, logsoft, soft, criterion, optimizer, opt):
    """
    one epoch training for program executor
    """
    model.train()
    criterion.train()

    for idx, data in enumerate(train_loader):
        start_t = time.time()

        optimizer.zero_grad()

        shape, label, param = data[0], data[1], data[2]

        bsz = shape.size(0)
        n_step = label.size(1)

        index = np.array(list(map(lambda x: n_step, label)))
        index = index - 1

        # add noise during training, making the executor accept
        # continuous output from program generator

        label = label.view(-1, 1)
        pgm_vector = 0.1 * torch.rand(bsz * n_step, stop_id)
        pgm_noise = 0.1 * torch.rand(bsz * n_step, 1)
        pgm_value = torch.ones(bsz * n_step, 1) - pgm_noise
        pgm_vector.scatter_(1, label, pgm_value)
        pgm_vector = pgm_vector.view(bsz, n_step, stop_id)

        param_noise = torch.rand(param.size())
        param_vector = param + 0.6 * (param_noise - 0.5)

        gt = shape
        index = torch.from_numpy(index).long()
        pgm_vector = pgm_vector.float()
        param_vector = param_vector.float()

        if opt.is_cuda:
            gt = gt.cuda()
            index = index.cuda()
            pgm_vector = pgm_vector.cuda()
            param_vector = param_vector.cuda()

        pred = model(pgm_vector, param_vector, index)
        scores = logsoft(pred)
        loss = criterion(scores, gt)

        loss.backward()
        clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        loss = loss.data[0]

        pred = soft(pred)
        pred = pred[:, 1, :, :, :]
        s1 = gt.view(-1, 32, 32, 32).data.cpu().numpy()
        s2 = pred.squeeze().data.cpu().numpy()
        s2 = (s2 > 0.5)

        batch_iou = BatchIoU(s1, s2)
        iou = batch_iou.sum() / s1.shape[0]

        end_t = time.time()

        if idx % (opt.info_interval * 10) == 0:
            print("Train: epoch {} batch {}/{}, loss13 = {:.3f}, iou = {:.3f}, time = {:.3f}"
                  .format(epoch, idx, len(train_loader), loss, iou, end_t - start_t))
            sys.stdout.flush()


def validate(epoch, val_loader, model, logsoft, soft, criterion, opt, gen_shape=False):

    # load pre-fixed randomization
    try:
        rand1 = np.load(opt.rand1)
        rand2 = np.load(opt.rand2)
        rand3 = np.load(opt.rand3)
    except:
        rand1 = np.random.rand(opt.batch_size * opt.seq_length, stop_id).astype(np.float32)
        rand2 = np.random.rand(opt.batch_size * opt.seq_length, 1).astype(np.float32)
        rand3 = np.random.rand(opt.batch_size, opt.seq_length, max_param - 1).astype(np.float32)
        np.save(opt.rand1, rand1)
        np.save(opt.rand2, rand2)
        np.save(opt.rand3, rand3)

    model.eval()
    criterion.eval()

    generated_shapes = []
    original_shapes = []

    for idx, data in enumerate(val_loader):
        start_t = time.time()

        shape, label, param = data[0], data[1], data[2]

        bsz = shape.size(0)
        n_step = label.size(1)

        index = np.array(list(map(lambda x: n_step, label)))
        index = index - 1

        label = label.view(-1, 1)
        pgm_vector = 0.1 * torch.from_numpy(rand1)
        pgm_noise = 0.1 * torch.from_numpy(rand2)
        pgm_value = torch.ones(bsz * n_step, 1) - pgm_noise
        pgm_vector.scatter_(1, label, pgm_value)
        pgm_vector = pgm_vector.view(bsz, n_step, stop_id)

        param_noise = torch.from_numpy(rand3)
        param_vector = param + 0.6 * (param_noise - 0.5)

        gt = shape
        index = torch.from_numpy(index).long()
        pgm_vector = pgm_vector.float()
        param_vector = param_vector.float()

        if opt.is_cuda:
            gt = gt.cuda()
            index = index.cuda()
            pgm_vector = pgm_vector.cuda()
            param_vector = param_vector.cuda()

        pred = model(pgm_vector, param_vector, index)
        scores = logsoft(pred)
        loss = criterion(scores, gt)

        loss = loss.data[0]

        pred = soft(pred)
        pred = pred[:, 1, :, :, :]
        s1 = gt.view(-1, 32, 32, 32).data.cpu().numpy()
        s2 = pred.squeeze().data.cpu().numpy()
        s2 = (s2 > 0.5)

        batch_iou = BatchIoU(s1, s2)
        iou = batch_iou.sum() / s1.shape[0]

        original_shapes.append(s1)
        generated_shapes.append(s2)

        end_t = time.time()

        if (idx + 1) % opt.info_interval == 0:
            print("Test: epoch {} batch {}/{}, loss13 = {:.3f}, iou = {:.3f}, time = {:.3f}"
                  .format(epoch, idx + 1, len(val_loader), loss, iou, end_t - start_t))
            sys.stdout.flush()

    if gen_shape:
        generated_shapes = np.asarray(generated_shapes)
        original_shapes = np.asarray(original_shapes)

    return generated_shapes, original_shapes


def run():

    opt = options_train_executor.parse()

    print('===== arguments: program executor =====')
    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))
    print('===== arguments: program executor =====')

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # build dataloader
    train_set = PartPrimitive(opt.train_file)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    val_set = PartPrimitive(opt.val_file)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )

    # build the model
    model = RenderNet(opt)
    logsoft = nn.LogSoftmax(dim=1)
    soft = nn.Softmax(dim=1)
    criterion = nn.NLLLoss(weight=torch.Tensor([opt.n_weight, opt.p_weight]))

    if opt.is_cuda:
        if opt.num_gpu > 1:
            gpu_ids = [i for i in range(opt.num_gpu)]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        model = model.cuda()
        logsoft = logsoft.cuda()
        soft = soft.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    optimizer = optim.Adam(model.parameters(),
                           lr=opt.learning_rate,
                           betas=(opt.beta1, opt.beta2),
                           weight_decay=opt.weight_decay)

    for epoch in range(1, opt.epochs+1):
        adjust_learning_rate(epoch, opt, optimizer)

        print("###################")
        print("training")
        train(epoch, train_loader, model, logsoft, soft, criterion, optimizer, opt)

        print("###################")
        print("testing")
        gen_shapes, ori_shapes = validate(epoch, val_loader, model,
                                          logsoft, soft, criterion, opt, gen_shape=True)
        gen_shapes = (gen_shapes > 0.5)
        gen_shapes = gen_shapes.astype(np.float32)
        iou = BatchIoU(ori_shapes, gen_shapes)
        print("Mean IoU: {:.3f}".format(iou.mean()))

        if epoch % opt.save_interval == 0:
            print('Saving...')
            state = {
                'opt': opt,
                'model': model.module.state_dict() if opt.num_gpu > 1 else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.t7'.format(epoch=epoch))
            torch.save(state, save_file)

    state = {
        'opt': opt,
        'model': model.module.state_dict() if opt.num_gpu > 1 else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': opt.epochs,
    }
    save_file = os.path.join(opt.save_folder, 'program_executor.t7')
    torch.save(state, save_file)


if __name__ == '__main__':
    run()
