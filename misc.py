from __future__ import print_function

import torch
import numpy as np

from programs.utils import draw_vertical_leg as draw_vertical_leg_new
from programs.utils import draw_rectangle_top as draw_rectangle_top_new
from programs.utils import draw_square_top as draw_square_top_new
from programs.utils import draw_circle_top as draw_circle_top_new
from programs.utils import draw_middle_rect_layer as draw_middle_rect_layer_new
from programs.utils import draw_circle_support as draw_circle_support_new
from programs.utils import draw_square_support as draw_square_support_new
from programs.utils import draw_circle_base as draw_circle_base_new
from programs.utils import draw_square_base as draw_square_base_new
from programs.utils import draw_cross_base as draw_cross_base_new
from programs.utils import draw_sideboard as draw_sideboard_new
from programs.utils import draw_horizontal_bar as draw_horizontal_bar_new
from programs.utils import draw_vertboard as draw_vertboard_new
from programs.utils import draw_locker as draw_locker_new
from programs.utils import draw_tilt_back as draw_tilt_back_new
from programs.utils import draw_chair_beam as draw_chair_beam_new
from programs.utils import draw_line as draw_line_new
from programs.utils import draw_back_support as draw_back_support_new

from programs.loop_gen import decode_loop, translate, rotate, end

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from pytorch3d.ops import sample_points_from_meshes

def plot_pointcloud(points, title=""):
    # Sample points uniformly from the surface of the mesh.
    x, y, z = points.clone().detach().cpu().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    #ax.scatter3D(x, y, z)
    ax.scatter3D(z, y, x)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
#     mid_x = (x.max()+x.min()) * 0.5
#     mid_y = (y.max()+y.min()) * 0.5
#     mid_z = (z.max()+z.min()) * 0.5
#     ax.set_xlim(mid_x - max_range, mid_x + max_range)
#     ax.set_ylim(mid_y - max_range, mid_y + max_range)
#     ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(20,)
#     plt.axis('off')
    plt.show()    
    plt.savefig("save_img/" + "render" + title)
    
def plot_mesh(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 1000).squeeze_()
#     points=points[:,[1,2,0]]
    x, y, z = points.clone().detach().cpu().unbind(1)
    print("z: min: {}, max: {}".format(torch.min(z),torch.max(z)))
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
#     ax.view_init(190, 30)
    plt.show()
    
def plot_voxels(voxel,title=""):
    # plot one voxel data
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
#     ax.view_init(190, 30)
    plt.axis('off')

    plt.show()
    plt.savefig("save_img/" + "voxel" + title)

def get_distance_to_center():
    '''Distance of every grid point to the center'''
    x = np.arange(32)
    y = np.arange(32)
    xx, yy = np.meshgrid(x, y)
    xx = xx + 0.5
    yy = yy + 0.5
    d = np.sqrt(np.square(xx - int(32 / 2)) + np.square(yy - int(32 / 2)))
    return d


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for i,param in enumerate(group['params']):
            try:
                param.grad.data.clamp_(-grad_clip, grad_clip)
            except:
                pass

def get_class(pgm):
    if pgm.dim() == 3:
        _, idx = torch.max(pgm, dim=2)
    elif pgm.dim() == 2:
        idx = pgm
    else:
        raise IndexError("dimension of pgm is wrong")
    return idx


def get_last_block(pgm):
    bsz = pgm.size(0)
    n_block = pgm.size(1)
    n_step = pgm.size(2)

    if torch.is_tensor(pgm):
        pgm = pgm.clone()
    else:
        pgm = pgm.data.clone()

    if pgm.dim() == 4:
        _, idx = torch.max(pgm, dim=3)
        idx = idx.cpu()
    elif pgm.dim() == 3:
        idx = pgm.cpu()
    else:
        raise ValueError("pgm.dim() != 2 or 3")

    max_inds = []
    for i in range(bsz):
        j = n_block - 1
        while j >= 0:
            if idx[i, j, 0] == 0:
                break
            j = j - 1

        if j == -1:
            max_inds.append(0)
        else:
            max_inds.append(j)

    return np.asarray(max_inds)


def sample_block(max_inds, include_tail=False):
    sample_inds = []
    for ind in max_inds:
        if include_tail:
            sample_inds.append(np.random.randint(0, ind + 1))
        else:
            sample_inds.append(np.random.randint(0, ind))
    return np.asarray(sample_inds)


def get_max_step_pgm(pgm):
    batch_size = pgm.size(0)

    if torch.is_tensor(pgm):
        pgm = pgm.clone()
    else:
        pgm = pgm.data.clone()

    if pgm.dim() == 3:
        pgm = pgm[:, 1:, :]
        idx = get_class(pgm).cpu()
    elif pgm.dim() == 2:
        idx = pgm[:, 1:].cpu()
    else:
        raise ValueError("pgm.dim() != 2 or 3")

    max_inds = []

    for i in range(batch_size):
        j = 0
        while j < idx.shape[1]:
            if idx[i, j] == 0:
                break
            j = j + 1
        if j == 0:
            raise ValueError("no programs for such sample")
        max_inds.append(j)

    return np.asarray(max_inds)


def get_vacancy(pgm):
    batch_size = pgm.size(0)

    if torch.is_tensor(pgm):
        pgm = pgm.clone()
    else:
        pgm = pgm.data.clone()

    if pgm.dim() == 3:
        pgm = pgm[:, 1:, :]
        idx = get_class(pgm).cpu()
    elif pgm.dim() == 2:
        idx = pgm[:, 1:].cpu()
    else:
        raise ValueError("pgm.dim() != 2 or 3")

    vac_inds = []

    for i in range(batch_size):
        j = 0
        while j < idx.shape[1]:
            if idx[i, j] == 0:
                break
            j = j + 1
        if j == idx.shape[1]:
            j = j - 1
        vac_inds.append(j)

    return np.asarray(vac_inds)


def sample_ind(max_inds, include_start=False):
    sample_inds = []
    for ind in max_inds:
        if include_start:
            sample_inds.append(np.random.randint(0, ind + 1))
        else:
            sample_inds.append(np.random.randint(0, ind))
    return np.asarray(sample_inds)


def sample_last_ind(max_inds, include_start=False):
    sample_inds = []
    for ind in max_inds:
        if include_start:
            sample_inds.append(ind)
        else:
            sample_inds.append(ind - 1)
    return np.array(sample_inds)


def decode_to_shape_new(pred_pgm, pred_param):
    batch_size = pred_pgm.size(0)

    idx = get_class(pred_pgm)

    pgm = idx.data.cpu().numpy()
    params = pred_param.data.cpu().numpy()
    params = np.round(params).astype(np.int32)

    data = np.zeros((batch_size, 32, 32, 32), dtype=np.uint8)
    for i in range(batch_size):
        for j in range(1, pgm.shape[1]):
            if pgm[i, j] == 0:
                continue
            data[i] = render_one_step_new(data[i], pgm[i, j], params[i, j])

    return data


def decode_pgm(pgm, param, loop_free=True):
    """
    decode and check one single block
    remove occasionally-happened illegal programs
    """
    flag = 1
    data_loop = []
    if pgm[0] == translate:
        if pgm[1] == translate:
            if 1 <= pgm[2] < translate:
                data_loop.append(np.hstack((pgm[0], param[0])))
                data_loop.append(np.hstack((pgm[1], param[1])))
                data_loop.append(np.hstack((pgm[2], param[2])))
                data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
                data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
            else:
                flag = 0
        elif 1 <= pgm[1] < translate:
            data_loop.append(np.hstack((pgm[0], param[0])))
            data_loop.append(np.hstack((pgm[1], param[1])))
            data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
        else:
            flag = 0
    elif pgm[0] == rotate:
        if pgm[1] == 10:
            data_loop.append(np.hstack((pgm[0], param[0])))
            data_loop.append(np.hstack((pgm[1], param[1])))
            data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
        if pgm[1] == 17:
            data_loop.append(np.hstack((pgm[0], param[0])))
            data_loop.append(np.hstack((pgm[1], param[1])))
            data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
        else:
            flag = 0
    elif 1 <= pgm[0] < translate:
        data_loop.append(np.hstack((pgm[0], param[0])))
        data_loop.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))
    else:
        flag = 0

    if flag == 0:
        data_loop.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))
        data_loop.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))
    
    data_loop = [x.tolist() for x in data_loop]
    data_loop_free = decode_loop(data_loop)
    data_loop_free = np.asarray(data_loop_free)

    if len(data_loop_free) == 0:
        data_loop_free = np.zeros((2, 8), dtype=np.int32)

    if loop_free:
        return data_loop_free
    else:
        return np.asarray(data_loop)


def decode_all(pgm, param, loop_free=False):
    """
    decode program to loop-free (or include loop)
    """
    n_block = pgm.shape[0]
    param = np.round(param).astype(np.int32)

    result = []
    for i in range(n_block):
        res = decode_pgm(pgm[i], param[i], loop_free=loop_free)
        result.append(res)
    result = np.concatenate(result, axis=0)
    return result


def execute_shape_program(pgm, param):
    """
    execute a single shape program
    """
    trace_sets = decode_all(pgm, param, loop_free=True)
    data = np.zeros((32, 32, 32), dtype=np.uint8)

    for trace in trace_sets:
        cur_pgm = trace[0]
        cur_param = trace[1:]
        data = render_one_step_new(data, cur_pgm, cur_param)

    return data


def decode_multiple_block(pgm, param):
    """
    decode and execute multiple blocks
    can run with batch style
    """
    # pgm: bsz x n_block x n_step x n_class
    # param: bsz x n_block x n_step x n_class
    bsz = pgm.size(0)
    n_block = pgm.size(1)
    data = np.zeros((bsz, 32, 32, 32), dtype=np.uint8)
    for i in range(n_block):
        if pgm.dim() == 4:
            prob_pre = torch.exp(pgm[:, i, :, :].data)
            _, it1 = torch.max(prob_pre, dim=2)
        elif pgm.dim() == 3:
            it1 = pgm[:, i, :]
        else:
            raise NotImplementedError('pgm has incorrect dimension')
        it2 = param[:, i, :, :].data.clone()
        it1 = it1.cpu().numpy()
        it2 = it2.cpu().numpy()
        data = render_block(data, it1, it2)

    return data


def count_blocks(pgm):
    """
    count the number of effective blocks
    """
    # pgm: bsz x n_block x n_step x n_class
    pgm = pgm.data.clone().cpu()
    bsz = pgm.size(0)
    n_blocks = []
    n_for = []
    for i in range(bsz):
        prob = torch.exp(pgm[i, :, :, :])
        _, it = torch.max(prob, dim=2)
        v = it[:, 0].numpy()
        n_blocks.append((v > 0).sum())
        n_for.append((v == translate).sum() + (v == rotate).sum())

    return np.asarray(n_blocks), np.asarray(n_for)


def render_new(data, pgms, params):
    """
    render one step for a batch
    """
    batch_size = data.shape[0]
    params = np.round(params).astype(np.int32)

    for i in range(batch_size):
        data[i] = render_one_step_new(data[i], pgms[i], params[i])

    return data


def render_block(data, pgm, param):
    """
    render one single block
    """
    param = np.round(param).astype(np.int32)
    bsz = data.shape[0]
    for i in range(bsz):
        loop_free = decode_pgm(pgm[i], param[i])
        cur_pgm = loop_free[:, 0]
        #if i==3:print(cur_pgm)
        cur_param = loop_free[:, 1:]
        #if i==3:print(cur_param)
        for j in range(len(cur_pgm)):
            data[i] = render_one_step_new(data[i], cur_pgm[j], cur_param[j])

    return data


def render_one_step_new(data, pgm, param):
    """
    render one step
    """
    if pgm == 0:
        pass
    elif pgm == 1:
        data = draw_vertical_leg_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 2:
        data = draw_rectangle_top_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 3:
        data = draw_square_top_new(data, param[0], param[1], param[2], param[3], param[4])[0]
    elif pgm == 4:
        data = draw_circle_top_new(data, param[0], param[1], param[2], param[3], param[4])[0]
    elif pgm == 5:
        data = draw_middle_rect_layer_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 6:
        data = draw_circle_support_new(data, param[0], param[1], param[2], param[3], param[4])[0]
    elif pgm == 7:
        data = draw_square_support_new(data, param[0], param[1], param[2], param[3], param[4])[0]
    elif pgm == 8:
        data = draw_circle_base_new(data, param[0], param[1], param[2], param[3], param[4])[0]
    elif pgm == 9:
        data = draw_square_base_new(data, param[0], param[1], param[2], param[3], param[4])[0]
    elif pgm == 10:
        data = draw_cross_base_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 11:
        data = draw_sideboard_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 12:
        data = draw_horizontal_bar_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 13:
        data = draw_vertboard_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 14:
        data = draw_locker_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 15:
        data = draw_tilt_back_new(data, param[0], param[1], param[2], param[3], param[4], param[5], param[6])[0]
    elif pgm == 16:
        data = draw_chair_beam_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 17:
        data = draw_line_new(data, param[0], param[1], param[2], param[3], param[4], param[5], param[6])[0]
    elif pgm == 18:
        data = draw_back_support_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    else:
        raise RuntimeError("program id is out of range, pgm={}".format(pgm))

    return data


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import neuralnet_pytorch as nnt
import torch as T
from torch_scatter import scatter_add


def pointcloud2voxel_fast(pc: T.Tensor, voxel_size: int, grid_size=1., filter_outlier=True):
    b, n, _ = pc.shape
    half_size = grid_size / 2.
    breakpoint()
    valid = (pc >= -half_size) & (pc <= half_size)
    valid = T.all(valid, 2)
    pc_grid = (pc + half_size) * (voxel_size - 1.)
    indices_floor = T.floor(pc_grid)
    indices = indices_floor.long()
    batch_indices = T.arange(b).to(pc.device)
    batch_indices = nnt.utils.shape_padright(batch_indices)
    batch_indices = nnt.utils.tile(batch_indices, (1, n))
    batch_indices = nnt.utils.shape_padright(batch_indices)
    indices = T.cat((batch_indices, indices), 2)
    indices = T.reshape(indices, (-1, 4))

    r = pc_grid - indices_floor
    rr = (1. - r, r)
    if filter_outlier:
        valid = valid.flatten()
        indices = indices[valid]

    def interpolate_scatter3d(pos):
        updates_raw = rr[pos[0]][..., 0] * rr[pos[1]][..., 1] * rr[pos[2]][..., 2]
        updates = updates_raw.flatten()

        if filter_outlier:
            updates = updates[valid]

        indices_shift = T.tensor([[0] + pos]).to(pc.device)
        indices_loc = indices + indices_shift
        out_shape = (b,) + (voxel_size,) * 3
        out = T.zeros(*out_shape).to(pc.device).flatten()
        voxels = scatter_add(updates, nnt.utils.ravel_index(indices_loc.t(), out_shape), out=out).view(*out_shape)
        return voxels

    voxels = [interpolate_scatter3d([k, j, i]) for k in range(2) for j in range(2) for i in range(2)]
    voxels = sum(voxels)
    voxels = T.clamp(voxels, 0., 1.)
    return voxels


def voxelize(pc, vox_size=32):
    vox = pointcloud2voxel_fast(pc, vox_size)
    vox = T.clamp(vox, 0., 1.)
    vox = T.squeeze(vox)
    return vox


def iou(pred, gt, th=.5):
    pred = pred > th
    gt = gt > th
    intersect = T.sum(pred & gt).float()
    union = T.sum(pred | gt).float()
    iou_score = intersect / (union + 1e-8)
    return iou_score


def batch_iou(bpc1, bpc2, voxsize=32, thres=.4):
    def _iou(pc1, pc2):
        pc1 = pc1 - T.mean(pc1, -2, keepdim=True)
        pc1 = voxelize(pc1[None], voxsize)

        pc2 = pc2 - T.mean(pc2, -2, keepdim=True)
        pc2 = voxelize(pc2[None], voxsize)
        return iou(pc1, pc2, thres)

    total = map(_iou, bpc1, bpc2)
    return sum(total) / len(bpc1)