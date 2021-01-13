from __future__ import print_function

from torch.utils.data import Dataset
import numpy as np
import h5py
from programs.label_config import num_params, max_param
from programs.loop_gen import translate, rotate, end
import torch

# ned added
import os
import subprocess
import trimesh
import glob
from pytorch3d.io import load_objs_as_meshes, load_obj
from tqdm import tqdm

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

# new ones
class ShapeNetVM(Dataset):
    '''dataset include both voxels and meshes'''
    def __init__(self, voxels, mesh_path):
        super(ShapeNetVM, self).__init__()
        self.vox = voxels 
        self.mesh_paths = mesh_path
        self.num = self.vox.shape[0]
            
    def __getitem__(self, index):
        voxel = torch.clone(self.vox[index])
        voxel = voxel.type(torch.float)
        mesh_path = self.mesh_paths[index]
        
        return voxel, mesh_path
    
    def __len__(self):
        return self.num
    
def vox_mesh_from_shapeNet(synset_path):
    '''Store all the voxel data of the synset under the path
    in a .h5 file.
    :args:
    synset_path (str): path to an unziped synset directory
    save_file (str): path to save .h5 file
    '''
    # find all files under synset
    shape_path_list = glob.glob(os.path.join(synset_path, "*"))
    shapes = []
    mesh_paths = []
    dirname = os.path.dirname(synset_path)
    basename = os.path.basename(synset_path)
    trn_cache_name = os.path.join(dirname,basename+"_trn_voxels.h5")
    val_cache_name = os.path.join(dirname,basename+"_val_voxels.h5")
    vox_cache_exist = os.path.exists(trn_cache_name)

    if not vox_cache_exist:
        print("==> Loading voxels [no cache]...")
        for shape_path in tqdm(shape_path_list):
            # mesh path
            mesh_path = os.path.join(shape_path, 
                                    "models/model_normalized.obj")  
            mesh_paths.append(mesh_path)
            
            # get voxal path, convert to numpy arrays
            vox_path = os.path.join(shape_path, 
                                    "models/model_normalized.binvox")        
            
            if not os.path.exists(vox_path):
                # generate 32*32*32 centered voxels
                cmd = 'binvox -d 32 -e -cb ' + mesh_path
                subprocess.run(cmd.split())
                
            vox = trimesh.load(vox_path)
            vox_array = vox.matrix
            shapes.append(vox_array)           
        print("==> Done!")  
        
        print("==> Caching")
        
        num_shapes = len(shapes)
        num_val = min(1000, int(num_shapes*0.2))
        trn_shapes = shapes[:num_shapes-num_val]
        val_shapes = shapes[num_shapes-num_val:]
        
        f_vox_trn = h5py.File(trn_cache_name, 'w')
        f_vox_trn['data'] = trn_shapes
        f_vox_trn.close()
        
        f_vox_val = h5py.File(val_cache_name, 'w')
        f_vox_val['data'] = val_shapes
        f_vox_val.close()
        
        print("==> Done!")  
    else: 
        for shape_path in shape_path_list:
            # mesh path
            mesh_path = os.path.join(shape_path, 
                                    "models/model_normalized.obj")  
            mesh_paths.append(mesh_path)
        
        print("==> Loading voxels [has cache]...")
        f_vox_trn = h5py.File(trn_cache_name, 'r')
        trn_shapes = f_vox_trn['data']
        f_vox_val = h5py.File(val_cache_name, 'r')
        val_shapes = f_vox_val['data']
        
        print("==> Done!")  

    trn_shapes = torch.tensor(trn_shapes).transpose(1,2).transpose(2,3)
    val_shapes = torch.tensor(val_shapes).transpose(1,2).transpose(2,3)
#     trn_shapes = np.array(trn_shapes)
#     val_shapes = np.array(val_shapes)
    
    trn_mesh_p = mesh_paths[:len(trn_shapes)]
    val_mesh_p = mesh_paths[len(trn_shapes):]
    
    return trn_shapes, val_shapes, trn_mesh_p, val_mesh_p