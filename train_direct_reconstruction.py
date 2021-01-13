import os
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import point_mesh_face_distance,chamfer_distance
from pytorch3d.structures import Pointclouds
import sys
import numpy as np
from tqdm import tqdm

from remove_loop import expend_pgm_params,clean_batch_program
from distance_field import batch_sample_points_from_primitives, standardize_primitives, weighted_loss
from misc import clip_gradient,decode_multiple_block
from options import options_direct_adaptation
from dataset import vox_mesh_from_shapeNet, ShapeNetVM
from model import BlockOuterNet
from criterion import BatchIoU

# define training function
def train(epoch, train_loader, model, criterion, optimizer, opt):
    '''train one epoch'''
    model.train()
    # current not implemented as nn.module
    
    cov_w = 1
    con_w = 1
    
    for idx, data in enumerate(train_loader):
        start = time.time()
        
        # prepare voxels, load meshes
        voxes, meshes_path = data[0], data[1]
        voxes = torch.unsqueeze(voxes, 1)
        voxes = voxes.cuda()
        meshes = load_objs_as_meshes(meshes_path, load_textures=False).cuda()
#         meshes.scale_verts_(40)

        optimizer.zero_grad()
        
        with torch.autograd.set_detect_anomaly(True):
            # decode, reorganize output
            pgms, params = model.decode(voxes)
            batch_prim_idx, batch_prim_param = expend_pgm_params(pgms, params,
                                                                sample=True)
            batch_clean_idx, batch_clean_param = clean_batch_program(batch_prim_idx,
                                                                     batch_prim_param)

#             for i,pg in enumerate(batch_clean_idx):print("cleaned pg:",i,pg)
            shape_primitives,cyl_cub_idx = standardize_primitives(batch_clean_idx,
                                              batch_clean_param)
            primis_samples,confs = batch_sample_points_from_primitives(\
                                            shape_primitives=shape_primitives,
                                            cyl_cub_idx_bth=cyl_cub_idx,
                                            num_samples=5000)
            primis_samples = primis_samples[:,:,[2,0,1]]/40
            meshes_samples = sample_points_from_meshes(meshes)

            loss = chamfer_distance(primis_samples, meshes_samples)[0]

            loss.backward()

            # update
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()

            torch.cuda.synchronize()
            
        # print out
        end = time.time()
        if idx % opt.info_interval == 0:
            print("Train: epoch {} batch {}/{}, chamfer_distance = {:.3f}, time ={:.3f}"\
                  .format(epoch, idx, len(train_loader), 
                          loss.data.item(),
                          end - start))
        #print("Train: epoch {} batch {}/{}, loss_coverage = {:.3f}, loss_consistent = {:.3f}, time = {:.3f}".format(
        #                epoch, idx, len(train_loader), 
        #                loss_cov.data.item(),
        #                loss_con.data.item(),
        #                end - start))
            sys.stdout.flush()
        
# define training function
def validate(epoch, val_loader, generator, opt, gen_shape=False):
    '''train one epoch'''
    generator.eval()
#     crit_cov.eval()
#     crit_con.eval()
    
    generated_shapes = []
    original_shapes = []
    
    for idx, data in enumerate(val_loader):
        start = time.time()
        
        voxes, meshes_path = data[0], data[1]
        voxes = torch.unsqueeze(voxes, 1)
        voxes = voxes.cuda()
        meshes = load_objs_as_meshes(meshes_path, load_textures=False).cuda()
#         meshes.scale_verts_(40)
        
        with torch.no_grad():
            pgms, params = generator.decode(voxes)
            batch_prim_idx, batch_prim_param = expend_pgm_params(pgms, params,
                                                                sample=False)
            batch_clean_idx, batch_clean_param = clean_batch_program(batch_prim_idx,
                                                                     batch_prim_param)

            shape_primitives,cyl_cub_idx = standardize_primitives(batch_clean_idx,
                                              batch_clean_param)
            primis_samples,confs = batch_sample_points_from_primitives(\
                                            shape_primitives=shape_primitives,
                                            cyl_cub_idx_bth=cyl_cub_idx,
                                            num_samples=5000)
            primis_samples = primis_samples[:,:,[2,0,1]]/40
            meshes_samples = sample_points_from_meshes(meshes)
            
        torch.cuda.synchronize()
        
        end = time.time()

        if idx % opt.info_interval == 0:

            print("Test: epoch {} batch {}/{}, time={:.3f}"
                  .format(epoch, idx, len(val_loader), end-start))
            
#         if gen_shape:
#             generated_shapes.append(decode_multiple_block(pgms, params))
#             original_shapes.append(data[0].clone().numpy())
        
#     if gen_shape:
#         generated_shapes = np.concatenate(generated_shapes, axis=0)
#         original_shapes = np.concatenate(original_shapes, axis=0)
        if gen_shape:
            generated_shapes.append(primis_samples)
            original_shapes.append(meshes_samples)
        
    if gen_shape:
        generated_shapes = torch.cat(generated_shapes)
        original_shapes = torch.cat(original_shapes)
        
    return generated_shapes, original_shapes

def run():
    # get options
    os.environ["CUDA_VISIBLE_DEVICE"]="3"
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    
    opt = options_direct_adaptation.parse()
    print('===== arguments: direct adaptation =====')
    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))
    print('===== arguments: direct adaptation =====')

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # build loaders    
    trn_vox, val_vox, trn_mesh_p, val_mesh_p = vox_mesh_from_shapeNet(opt.synset_p)
    train_set = ShapeNetVM(trn_vox, trn_mesh_p)
    train_loader = DataLoader(
                dataset=train_set,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.num_workers,
                )
    val_set = ShapeNetVM(val_vox, val_mesh_p)
    val_loader = DataLoader(
                dataset=val_set,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
                )
 
    # load program generator
    ckpt_p_gen = torch.load(opt.p_gen_path)
    generator = BlockOuterNet(ckpt_p_gen['opt'])
    generator.load_state_dict(ckpt_p_gen['model'])

    # build loss function
    criterion = point_mesh_face_distance

    if opt.is_cuda:
        generator = generator.cuda()
        cudnn.benchmark = True

    optimizer = optim.Adam(generator.parameters(),
                           lr=opt.learning_rate,
                           betas=(opt.beta1, opt.beta2),
                           weight_decay=opt.weight_decay)

    print("#######################")
    print("testing")
    gen_shapes, ori_shapes = validate(0, val_loader, generator, opt, gen_shape=True)
#     IoU = BatchIoU(ori_shapes, gen_shapes)
    CD = chamfer_distance(gen_shapes, ori_shapes)[0].data.item()
    print("Chamfer Distance: ", CD)
    
    best_cd = 0

    for epoch in range(1, opt.epochs+1):
        print("####################")
        print("adaptation")
        train(epoch, train_loader, generator, criterion, optimizer, opt)
        
        print("####################")
        print("testing")
        gen_shapes, ori_shapes = validate(epoch, val_loader, generator, opt, gen_shape=True)
#         IoU = BatchIoU(ori_shapes, gen_shapes)
        CD = chamfer_distance(gen_shapes, ori_shapes)[0].data.item()

#         print("iou: ", IoU.mean())
        print("Chamfer Distance: ", CD)

        if epoch % opt.save_interval == 0:
            print('Saving...')
            state = {
                    'opt': ckpt_p_gen['opt'],
                    'model': generator.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    }
            save_file = os.path.join(opt.save_folder,
                    'ckpt_epoch_{epoch}.t7'.format(epoch=epoch))
            torch.save(state, save_file)

        if CD <= best_cd:
            print('Saving best model')
            state = {
                     'opt': ckpt_p_gen['opt'],
                     'model': generator.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch,
                     }
            save_file = os.path.join(opt.save_folder, 
                    'program_generator_GA_{}.t7'.format(opt.cls))
            torch.save(state, save_file)
            best_cd = CD

if __name__ == '__main__':
    run()
