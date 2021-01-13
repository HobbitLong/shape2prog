import torch
import numpy as np
import torch.nn.functional as F
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.loss import point_mesh_distance
from pytorch3d.structures import Pointclouds
import kornia
import math

def partition_primitives_np(pgm_i):
    '''partition programs with different geometry parameters
        # partition based on different parameter length, structure
        # 2 shape param: cuboid:[3,7,9]; cylinder:[4,6,8]
        # 3 shape param: cuboid:[1,2,5,11,12,13,14,16,18],[15]
        # line (two position, 1 radius): 17
        # cross_base (cb) id: 10
    '''
    # - three param, [5:7] half width/depth: three00_mask
    # 2: rectangle_top, 5: mid_rect_layer

    # - three param, [5:7] whole width/depth: three11_mask
    # 1: vertical_leg, 12 horizontal_bar, 13 vertboard, 14 locker, 16 chair beam
    # 18: back_support

    # - three param, [5] half, [6] whole: three01_mask
    # 11: sideboard
    p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18 = pgm_i==1,\
         pgm_i==2, pgm_i==3, pgm_i==4, pgm_i==5, pgm_i==6, pgm_i==7, pgm_i==8,\
         pgm_i==9,pgm_i==10, pgm_i==11, pgm_i==12, pgm_i==13, pgm_i==14,\
         pgm_i==15, pgm_i==16, pgm_i==17, pgm_i==18
    
    two_mask = p3|p4|p6|p7|p8|p9
    line_mask = p17
    cb_mask = p10
    tiltback_mask = p15
    three00_mask = p2|p5
    three11_mask = p1|p12|p13|p14|p16|p18
    three01_mask = p11
    cyl_mask = p4|p6|p8
    not_cyl_mask = ~(cyl_mask)
    
    two_idx = torch.nonzero(two_mask).squeeze(1).cuda()
    line_idx = torch.nonzero(line_mask).squeeze(1).cuda()
    cb_idx = torch.nonzero(cb_mask).squeeze(1).cuda()
    tb_idx = torch.nonzero(tiltback_mask).squeeze(1).cuda()
    three00_idx = torch.nonzero(three00_mask).squeeze(1).cuda()
    three11_idx = torch.nonzero(three11_mask).squeeze(1).cuda()
    three01_idx = torch.nonzero(three01_mask).squeeze(1).cuda()
    cyl_idx = torch.nonzero(cyl_mask).squeeze(1).cuda()
    cub_idx = torch.nonzero(not_cyl_mask).squeeze(1).cuda()
    
    return (two_idx,line_idx,cb_idx,tb_idx,three00_idx,three11_idx,three01_idx,cyl_idx,cub_idx)
    
def standardize_primitives(pgms, params):
    '''standardize the shapes to transition (x,y,z), geometry (h,w,d), 
    rotation matrix
    :return:
    tuple of (trans, geometry, rotation)
    '''
    num_bth = len(pgms)
    shapes = []
    cyl_idx_bth = []
    cub_idx_bth = []

    for i in range(num_bth):        
        # - partition based on different parameter length (used)
        # 2 shape param: cuboid:[3,7,9]; cylinder:[4,6,8]
        # 3 shape param: cuboid:[1,2,5,11,12,13,14,16,18],[15]
        # lines: 17
        # cross_base: 10
#         breakpoint()
        pgm_i = torch.tensor(pgms[i])
        num_par = params[i].shape[0]
        two_pid,line_pid,cb_pid,tb_pid,three00_pid,three11_pid,\
        three01_pid,cyl_pid,not_cyl_pid = partition_primitives_np(pgm_i)
        
        # new params to store 3 axis rotations, intotal: 1 conf, 3 pos of the center 
        # point, 3 geo all 1/2 of the side length, 3 rotation axis-angle parameter.
        new_params = torch.cat((params[i].clone(), torch.zeros(num_par,2).cuda()), dim=1)
                               
        # 1. 17 draw_line
        num_line = line_pid.shape[0]
        if num_line > 0 :
            line_param = params[i][line_pid] # npar * 8            
            ln_vec = line_param[:,1:4]-line_param[:,4:7]            
            ln_length = torch.sqrt(torch.sum(ln_vec**2,dim=1))

            # translation
            ln_t = (line_param[:,1:4]+line_param[:,4:7])/2
            
            # geometry; length /2 because we need half the height
            radius = line_param[:,7:8]
            ln_z = torch.cat((ln_length.unsqueeze(1)/2, radius, radius), dim=1)
            
            # rotation
            x = torch.FloatTensor([1,0,0]).repeat(num_line,1).cuda()
            theta_x = torch.acos(torch.sum(x*ln_vec,dim=1)/ln_length).unsqueeze(1)
            axis = torch.cross(x, ln_vec, dim=1)
            ln_q = axis / torch.norm(axis, p=2, dim=1, keepdim=True) 
            new_params[line_pid,1:] = torch.cat((ln_t, ln_z, ln_q*theta_x), 
                                                        dim=1)#.clone()
          
        # 2. two shape parameter case:
        #    draw_square_top3, circle_top4, circle_support6, square_support7,
        #    circle_base8, square_base9
        if two_pid.shape[0] > 0:
            
            # change two parameters shape to the standard shape format (as for lines)
            new_params[two_pid,1] = new_params[two_pid,1] + 1/2 * params[i][two_pid,4]#.clone()
            new_params[two_pid,4] = new_params[two_pid,4] * 1/2
            
            new_params[two_pid,6] = new_params[two_pid,5]#.clone() # set depth = width
            new_params[two_pid,7:10] = 10e-20 # zero out rotations 
        
        # 3. 10 cross_base case
        num_cb = cb_pid.shape[0]
        if num_cb > 0:
            new_params[cb_pid,1] = new_params[cb_pid,1] + 1/2 * params[i][cb_pid,4]#.clone()
            # y are one ,z are params[i][cb_pid,5:6]
            new_params[cb_pid,5:7] = torch.cat((torch.ones(num_cb,1).cuda(), 
                                               params[i][cb_pid,5:6]), dim=1)#.clone()
            # rotation w.r.t. x-axis
            new_params[cb_pid,7:8] = math.pi/2 * params[i][cb_pid,6:7].remainder(4)#.clone().remainder(4)
            new_params[cb_pid,4] = new_params[cb_pid,4] * 1/2
            
        # 4. 15 tilt_back case
        if tb_pid.shape[0] > 0:
            # divide by two to fit the rendered size
            new_params[tb_pid,1:4] = new_params[tb_pid,1:4] + 1/2 * params[i][tb_pid,4:7]#.clone()
            
            # rotation axis-angle w.r.t. z-axis
            new_params[tb_pid,9:10] = kornia.deg2rad(params[i][tb_pid,7:8]*5)#.clone()*5)
            new_params[tb_pid,7:9] = 10e-20
            new_params[tb_pid,4:7] = new_params[tb_pid,4:7] * 1/2
        
        # 5. three param, [5:7] half width/depth
        # 2: rectangle_top, 5: mid_rect_layer
        if three00_pid.shape[0] > 0:
            new_params[three00_pid,7:10] = 10e-20
            new_params[three00_pid,1] = new_params[three00_pid,1] + \
                                           1/2 * params[i][three00_pid,4]#.clone()
            new_params[three00_pid,4] = new_params[three00_pid,4] * 1/2

        # 6. three param, [5:7] whole width/depth
        # 1: vertical_leg, 12 horizontal_bar, 13 vertboard, 14 locker, 16 chair beam
        # 18: back_support
        if three11_pid.shape[0] > 0:
            new_params[three11_pid,7:10] = 10e-20
            new_params[three11_pid,1:4] = new_params[three11_pid,1:4] +\
                                        1/2 * params[i][three11_pid,4:7]#.clone()
            new_params[three11_pid,4:7] = new_params[three11_pid,4:7] * 1/2
        
        # 7. three param, [5] half, [6] whole
        # 11: sideboard
        if three01_pid.shape[0] > 0:
            new_params[three01_pid,7:10] = 10e-20
            new_params[three01_pid,1] = new_params[three01_pid,1] + \
                                             1/2 * params[i][three01_pid,4]#.clone()
            new_params[three01_pid,3] = new_params[three01_pid,3] + \
                                             1/2 * params[i][three01_pid,6]#.clone()
            new_params[three01_pid,4] = new_params[three01_pid,4] * 1/2
            new_params[three01_pid,6] = new_params[three01_pid,6] * 1/2
            
        # extract the parameters from the standardized parameter tensor
        # make a mask of shapes with param great than 0.5
        min_param, _ = new_params[:,4:7].detach().min(dim=1)
        keep_mask = torch.gt(min_param,0.1)
        conf = new_params[:,0]
        trans = new_params[:,1:4] # npar * 3 (y,z,-x)
        shape = new_params[:,4:7]
        rotat = new_params[:,7:10] # npar * 3
        # npar*3*3
        rotatm = kornia.angle_axis_to_rotation_matrix(rotat.clone()).clone() 
        shapes.append((conf, trans, shape, rotatm)) # for later sample from 
        
        cyl_id = list(cyl_pid.detach().cpu().numpy())
        cub_id = list(not_cyl_pid.detach().cpu().numpy())
        for i,mask in enumerate(keep_mask):
            if not mask:
                try:cyl_id.remove(i)
                except: cub_id.remove(i)
        
        cyl_idx_bth.append(torch.LongTensor(cyl_id).cuda())
        cub_idx_bth.append(torch.LongTensor(cub_id).cuda())
    return shapes,(cyl_idx_bth,cub_idx_bth)




def batch_sample_points_from_primitives(shape_primitives, cyl_cub_idx_bth, num_samples=1000):
    num_bth = len(shape_primitives)
    samples = []
    conf_bth = []
    
    cyl_idx_bth, cub_idx_bth = cyl_cub_idx_bth
    
    for i in range(num_bth):
        
            
        samps,confs = sample_points_from_primitives(num_samples,
                                                     cyl_idx_bth[i],
                                                     cub_idx_bth[i],
                                                   shape_primitives[i])
        samples.append(samps)
        conf_bth.append(confs)
        
    batched_samples = torch.stack(samples)
    conf_bth = torch.stack(conf_bth)
    return batched_samples, conf_bth

def init_sample(num_per_pri,shape,cyl_idx,shape_primitives):
    '''sample on the surface of primitives'''
    def shift(rand):
        return (2 * rand - 1)
        
    all_sam=[]

    for i,num_sam in enumerate(num_per_pri):

        if i in cyl_idx:
            sfd,sfs = (shape[i,1]**2 * np.pi * 2,
                       shape[i,0] * shape[i,1] * 4 * np.pi)
            sf_areas = torch.abs(torch.stack((sfd,sfs)))
            sf_ratio = sf_areas/torch.sum(sf_areas)
            num_per_sf = torch.ceil(num_sam * sf_ratio)
            d,s = num_per_sf.to(torch.int).cpu()

            # sample disc points
            radius_sam = torch.rand(d,1)
            angle_sam = (2* np.pi * torch.rand(d,1) - np.pi)
            
            yz_sam = torch.cat((torch.sin(angle_sam) * radius_sam,
                                torch.cos(angle_sam) * radius_sam,
                               ),dim=1)

            disc_sam = torch.cat((torch.ones(d,1),yz_sam),dim=1).cuda()
            disc_sam[:d//2,0] = disc_sam[:d//2,0] * -1
            new_disc_sam = disc_sam * shape[i]

            # sample side points
            height_sam = torch.rand(s,1) * 2 - 1
            ang_sam = torch.rand(s,1) * np.pi * 2 - np.pi

            side_sam = torch.cat((height_sam,
                                  torch.cos(ang_sam),
                                  torch.sin(ang_sam),
                                 ),dim=1).cuda()
            new_side_sam = side_sam * shape[i]
            
            # concat together
            cyl_sam = torch.cat((new_disc_sam, new_side_sam),dim=0)
            all_sam.append(cyl_sam[:num_sam])

        else:
            sfa,sfb,sfc = (shape[i,1]*shape[i,2],
                           shape[i,2]*shape[i,0],
                           shape[i,0]*shape[i,1],)
            sf_areas = torch.abs(torch.stack((sfa,sfb,sfc)))
            sf_ratio = sf_areas/torch.sum(sf_areas)
            num_per_sf = torch.ceil(num_sam * sf_ratio)

            # num samples per side
            x,y,z = num_per_sf.to(torch.int)
            x_sam = torch.cat((torch.ones((x,1)),shift(torch.rand(x,2))),
                                                          dim=1).cuda()
            x_sam[:x//2,0] = x_sam[:x//2,0] * -1
            y_sam = torch.cat((shift(torch.rand(y,1)),torch.ones(y,1),
                               shift(torch.rand(y,1))),dim=1).cuda()
            y_sam[:y//2,1] = y_sam[:y//2,1] * -1
            z_sam = torch.cat((shift(torch.rand(z,2)),torch.ones(z,1)),dim=1).cuda()
            z_sam[:z//2,2] = z_sam[:z//2,2] * -1
            cub_sam = torch.cat((x_sam,y_sam,z_sam),dim=0) * shape[i]
            all_sam.append(cub_sam[:num_sam])
                             
    return torch.cat(all_sam,dim=0)
                                 
def sample_points_from_primitives(num_samples, cyl_idx, cub_idx, shape_primitives):
    '''Given a list of pgms and params, sample `num_samples` number
    of points on primitives' surface
    '''
    
    conf, trans, shape, rotat = shape_primitives
    
#     volumes = torch.prod(shape, dim=1)
    # approximating surface areas
#     print("shape dim:",shape.shape)
#     sf_area = shape[:,0]*shape[:,1]+shape[:,1]*shape[:,2]+shape[:,0]*shape[:,2]
#     sample_ratio = torch.abs(volumes/torch.sum(volumes))
#     num_per_pri = torch.ceil(num_samples * sample_ratio).type(torch.int64)
    
    shape = shape.abs()
    sf_area = torch.zeros(shape.shape[0]).cuda()
    sf_area[cyl_idx] = (shape[cyl_idx,0] * shape[cyl_idx,1] * 4 * np.pi +
                    shape[cyl_idx,1]**2 * np.pi * 2)
    sf_area[cub_idx] = 8*(shape[cub_idx,0] * shape[cub_idx,1] + 
                    shape[cub_idx,1] * shape[cub_idx,2] +
                    shape[cub_idx,2] * shape[cub_idx,0] )
    sf_area = torch.abs(sf_area)
    pri_ratio = sf_area/sf_area.sum()
    num_per_pri = torch.ceil(num_samples * pri_ratio).abs().type(torch.int64)
    sum_num = num_per_pri.sum()
    
    # conf
    try:
        confs = torch.repeat_interleave(conf, num_per_pri, dim=0)
        
        # scaled samples
        sampled_points = init_sample(num_per_pri,shape,cyl_idx,
                                                             shape_primitives)

        # rotate
        rotations = torch.eye(4).unsqueeze(0).repeat(sum_num,1,1).cuda()
        rotations[:, :3, :3] = torch.repeat_interleave(rotat, num_per_pri, dim=0)

        # translate
        sampled_points = kornia.geometry.linalg.transform_points(rotations,\
                                        sampled_points.unsqueeze(1)).squeeze()
        sampled_points = sampled_points + torch.repeat_interleave(trans, num_per_pri,
                                                                  dim=0)
    except:
        sampled_points = torch.rand(1,num_samples,3).cuda() * float('inf')
        confs = torch.rand(num_samples).cuda()
    
    return sampled_points.squeeze()[:num_samples],confs[:num_samples]

def weighted_loss(meshes, primis_samples, confs):
    pcls = Pointclouds(primis_samples)
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_mesh_distance.point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    
    # point to face in batch
    p_to_f = torch.stack(point_to_face.chunk(len(meshes)))
    p_to_f_all = p_to_f.mean(dim=1)
    
    # face to point
    f_to_p_all = []
    face_to_point = point_mesh_distance.face_point_distance(
            points, points_first_idx, tris, tris_first_idx, max_tris).mean()
    for n in range(len(meshes)):

        pcls = Pointclouds(primis_samples[n:n+1])

        points = pcls.points_packed()  # (P, 3)
        points_first_idx = pcls.cloud_to_packed_first_idx()

        # packed representation for faces
        verts_packed = meshes[n].verts_packed()
        faces_packed = meshes[n].faces_packed()
        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = meshes[n].mesh_to_faces_packed_first_idx()
        max_tris = meshes[n].num_faces_per_mesh().max().item()

        face_to_point = point_mesh_distance.face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris)
        
        f_to_p_all.append(face_to_point.mean())
        
    f_to_p_all = torch.stack(f_to_p_all)
    
    return p_to_f_all, f_to_p_all

# def consistency_loss_bth(meshes, primis_samples):
#     '''Batch consistency loss'''
#     num_bth = len(meshes)
#     loss = torch.zeros(num_bth)
# #     primis_samples = primis_samples[:,:,[2,0,1]] # change axes
#     primis_samples = primis_samples / 40 # the scaling factor
    
#     for i in range(num_bth):
#         loss[i] = point_mesh_face_distance(meshes[i],
#                             Pointclouds(primis_samples[i].unsqueeze(0)))
    
#     return loss
# # define the two loss function and compute them
# def dsfd_cub(sample_point, pgm, params):
#     '''Calculate the distance field between a point and a cuboid
#     specified by params.
#     1 "Leg", "Cuboid"
#     :param (h, s1, s2): position
#     :param (t, r1, r2): shape
#     2 "Top", "Rectangle"
#     :param (h, c1, c2): position
#     :param (t, r1, r2): shape
#     3 "Top", "Square"
#     :param (h, c1, c2): position
#     :param (r, t): shape
#     5 "Layer", "Rectangle"
#     :param (h, c1, c2): position
#     :param (t, r1, r2): shape
#     7 "Support", "Cuboid"
#     :param (h, c1, c2): position
#     :param (r, t): shape
#     9 "Base", "Square"
#     :param (h, c1, c2): position
#     :param (r, t): shape
#     11 "Sideboard", "Cuboid"
#     :param (h, s1, s2): position
#     :param (t, r1, r2): shape
#     12 "Horizontal_Bar", "Cuboid"
#     :param (h, s1, s2): position
#     :param (t, r1, r2): shape
#     13 "Vertical_board", "Cuboid"
#     :param (h, s1, s2): position
#     :param (t, r1, r2): shape
#     14 "Locker", "Cuboid"
#     :param (h, s1, s2): position
#     :param (t, r1, r2): shape
#     15 "Back", "Cuboid"
#     :param (h, s1, s2): position
#     :param (t, r1, r2, tilt_fact): shape
#     16 "Chair_Beam", "Cuboid"
#     :param (h, s1, s2): position
#     :param (t, r1, r2): shape
#     17 "Line", "line"
#     draw a line from (x1, y1, z1) to (x2, y2, z2) with a radius of r; the sampling version
#     18 "Back_support", "Cuboid"
#     :param (h, s1, s2): position
#     :param (t, r1, r2): shape
#     '''
#     # calculate the corresponding: 
#     # translation t (x,y,z), shape z (t,d,w), rotation q, 
#     if pgm in [1,2,5,11,12,13,14,16,18]:
#         # normal case with 3 shape param, 0 rotation
#         conf,t,z,q = params[0],params[1:4],params[4:7],torch.tensor(0).cuda()
#     elif pgm in [15]:
#         # 3 shape param, 1 rotation
#         conf,t,z,q = params[0],params[1:4],params[4:7],params[7]
#     elif pgm in [3,7,9]:
#         # 2 shape param, 0 rotation
#         r = params[5]
#         conf,t,z,q = params[0],params[1:4],torch.cat( (params[4:6],r) ),torch.tensor(0).cuda()
#     elif pgm == 17:
#         conf,t,z,q = (params[0],(params[1:4]-params[4:7])/2,
#                      torch.tensor((torch.dist(params[1:4],params[4:7]),1,1)),
#                      torch.tensor(0).cuda())
#     else:
#         raise Exception("not handled program")
        
#     # calculate the distant field loss between sample point and 
#     # transformed cuboid, which is equivalent to the loss between
#     # the transformed sample point and canonical cuboid.
# #     q = np.zeros(3,3) # convert to a rotation matrix
#     tran_p = transform_sample(sample_point, q, t).cuda()
#     loss = torch.abs(tran_p) - z
#     loss_sq = F.relu(loss).pow(2).sum()
    
#     return loss_sq

# def dsfd_cyl(sample_point, pgm, params):
#     '''Calculate the distance field between a point and a cylinder
#     specified by params.
#     - 4 "Top", "Circle"
#     :param (h, c1, c2): position
#     :param (t,r): shape
#     - 6 "Support", "Cylindar"
#     :param (h, c1, c2): position
#     :param (t, r): shape
#     - 8 "Base", "Circle"
#     :param (h, c1, c2): position
#     :param (t, r): shape
#     '''
#     conf,t,z,q = params[0], params[1:4], params[4:6], torch.tensor(0).cuda()
#     p = transform_sample(sample_point, q, t)
    
#     c = torch.zeros(3).cuda() # center
#     a, b = c+z[0], c-z[0] # center of top, bottom disc
#     x = torch.dot((c-p.squeeze()),a) # projection of (p-c) vector to central axis
#     if torch.abs(x) < z[0]/2:
#         n_sq = torch.sum(torch.pow(c-p,2)) # vecotr from c to p
#         y_sq = n_sq - torch.pow(x,2) # distance between p and central axis
#         # point is inside the cylinder  
#         if y_sq < torch.pow(z[1],1): loss = 0
#         # shortest distance is to the side
#         else: loss = torch.pow(torch.pow(y_sq, -2) - z[1], 2)
#     else:
#         n_sq = torch.sum(torch.pow(c-p,2)) # vecotr from c to p
#         y_sq = n_sq - torch.pow(x,2) # distance between p and central axis
#         # p projects onto the disc (top/bottom) of cylinder
#         if y_sq.item() < z[1].item(): loss = torch.pow(torch.abs(x)-z[0]-2, 2)
#         # p projects onto a circle (the rim/edge) of cylinder
#         else: loss = (torch.pow(torch.pow(y_sq, -2) - z[1], 2) +
#                       torch.pow(torch.abs(x) - z[0]/2, 2))
    
#     return loss

# def dsfd_line(sample_point, params):
#     '''Calculate the distance field between a point and a line
#     specified by params.
#     10 "Base", "Cross" ("line")
#     :param (h, c1, c2): position, angle: angle position
#     :param (r, t): shape
#     :param (angle): number of lines
#     '''
#     pass

# def df_cuboid(sample_points, shapes):
#     '''find the lost as if all points are cuboids
#     '''
#     loss = torch.abs(sample_points) - shapes # npar * nsam * 3 - npar * 1 * 3 = npar * nsam * 3
#     loss_sq = F.relu(loss).pow(2).sum(dim=2) # npar * nsam
#     return loss_sq


# def df_cylind(sample_points, shapes, num_sam):
#     '''find the loss to cuboids
#     '''
#     c = torch.zeros(num_par, enum_sam, 3)
#     a, b = c+torch.tensor(z[0],0,0), c-torch.tensor(z[0],0,0)
    
# def coverage_loss_bth(sample_points, pgms, params):
#     '''coverage loss in batch mode
#     '''
#     num_bth = sample_points.shape[0]
#     loss_bth = torch.zeros(num_bth)
#     for i in range(num_bth):
#         loss_bth[i] = coverage_loss(sample_points[i],
#                                     pgms[i],
#                                     params[i])
        
#     return loss_bth

# def coverage_loss(sample_point, pgms, params):
#     '''Calculate the coverage loss between sampled points on the 
#     target shape object and the shape primitves'''
#     loss_sum = 0
#     num_ps = sample_point.shape[0]
#     loss_prv = torch.tensor(np.inf) # init to infinity; used in comparison
#     for point in sample_point:
#         for pri_id, param in zip(pgms,params):
#             if pri_id in [1,2,3,5,7,9,11,12,13,14,15,16,17,18]: # e.g. cuboid
#                 loss_tmp = dsfd_cub(point, pri_id, param)
#             elif pri_id in [4,6,8]:
#                 loss_tmp = dsfd_cyl(point, pri_id, param)
#             elif pri_id in [10]:
#                 raise Exception("cross base shouldn't exist")
#             else:
#                 raise Exception("primitive not implemented")

#             if loss_tmp.item() < loss_prv.item():
                
#                 loss_prv = loss_tmp

#         loss_sum = loss_prv + loss_sum
#     loss_exp = loss_sum/num_ps
    
#     return loss_exp
# def coverage_loss_bth_fast(sample_points, pgms, params):
#     '''coverage loss in parallel
#     '''
#     sample_points = sample_points * 37.5 # the scaling factor
#     num_bth,num_sam,_ = sample_points.shape
#     loss = torch.zeros(num_bth,num_sam)
#     shapes = []

#     for i in range(num_bth):
#         # - partition based on different parameter length (used)
#         # 2 shape param: cuboid:[3,7,9]; cylinder:[4,6,8]
#         # 3 shape param: cuboid:[1,2,5,11,12,13,14,16,18],[15]
#         # lines: 17
#         # cross_base: 10
#         pgm_i = torch.tensor(pgms[i])
#         num_par = params[i].shape[0]
#         two_pid, line_pid, three_pid,_ = partition_primitives_np(pgm_i)
        
#         # rearange line parameters to standard shape format
#         # namely 1 conf, 3 translation, 3 geometry, 1 rotation.
#         num_line = line_pid.shape[0]
#         if num_line > 0 :
#             line_param = params[i][line_pid] # npar * 8
#             ln_t = (line_param[:,1:4]-line_param[:,4:7])/2
#             dist = torch.sqrt(torch.sum((line_param[:,1:4]-\
#                                          line_param[:,4:7])**2,dim=1))
#             ln_z = torch.cat((dist.unsqueeze(1), torch.ones(num_line, 2).cuda()),\
#                                                                             dim=1)
#             ln_q = line_param[:,7:9]
#             params[i][line_pid,1:] = torch.cat((ln_t, ln_z, ln_q),dim=1)
        
#         # change two parameters shape to the standard shape format (as for lines)
#         if two_pid.shape[0] > 0:
#             params[i][two_pid, 6] = params[i][two_pid, 5]
        
#         # extract the parameters from the standardized parameter tensor
#         conf = params[i][:,0]
#         trans = params[i][:,1:4] # npar * 3 (y,z,-x)
# #         trans = trans[:, torch.tensor([2,0,1])] # npar * 3 (-x,y,z)
# #         trans[:,0] = -1*trans[:,0] # npar * (x,y,z)
#         trans = trans.unsqueeze(1) # npar * 1 * 3 (x,y,z)
#         shape = params[i][:,4:7]
#         shape = shape.unsqueeze(1) # npar * 1 * 3
#         rotat = kornia.deg2rad(params[i][:,7]) # npar * 1 * 1
#         rotat_3d = torch.zeros((num_par, 3)).cuda() # npar * 3
#         rotat_3d[:,0] = rotat # npar * 3
#         rotat = kornia.angle_axis_to_rotation_matrix(rotat_3d) # npar * 4 * 4 (quaternions)
#         shapes.append((trans, shape, rotat)) # for later sample from primitives use.

#         # transform sample points
#         sample_points_expen = sample_points[i].repeat(num_par,1,1) # npar * nsam * 3
#         sample_points_trans = sample_points_expen - trans # npar * nsam * 3
#         sample_points_rotat = kornia.transform_points(rotat, sample_points_trans) # same
        
#         # select the closest part-primitive
#         df_loss = df_cuboid(sample_points_rotat, shape) # npar * nsam
#         df_loss,_ = df_loss.min(dim=0)
#         loss[i] = df_loss
    
#     return loss, shapes
