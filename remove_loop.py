# replace the loops with primitives and clean up the blank draw statements
from programs.loop_gen import decode_loop, translate, rotate, end
import torch.nn as nn
import torch
import numpy as np

import pdb

def expend_loop_blk(blk_pg_ids, blk_cnf_params):
    '''
    Given ONE block of unexpended program ids (including for loops) and
    parameters, return an tensor of expended primitive programs (with updated position parameters) 
    and a list of program indices (a program id or 0) without loops (with a presumed max length of 6)
    '''
    max_blk_prims = 10
    blk_pg_ids = blk_pg_ids.detach().cpu().numpy()
    
    if blk_pg_ids[0] == translate:
    # with translate
        if blk_pg_ids[1] == translate:
            if blk_pg_ids[2] <= 0:
                # no program
                batch_prim_idx = [0] * max_blk_prims
                batch_prim_param = blk_cnf_params[0].repeat(max_blk_prims,1)
                
            else:
                # two tran loops + draw; and discretize
                num_draw = blk_cnf_params[0, 1].round() * blk_cnf_params[1, 1].round()
                num_draw = int(num_draw.item())
                assert num_draw <= max_blk_prims, "more than {} draws! {}".format(max_blk_prims,num_draw)

#                 print(num_draw)
                if num_draw <= 0:
                    # no program
                    batch_prim_idx = [0] * max_blk_prims
                    batch_prim_param = blk_cnf_params[0].repeat(max_blk_prims,1)
                else:
                    # note the number of loops   
                    batch_prim_idx = [blk_pg_ids[2]] * num_draw + [0] * (max_blk_prims - num_draw)
                    batch_prim_param = blk_cnf_params[2].repeat(max_blk_prims,1)

                    # alter the entries with the loop vairable
                    u1 = torch.FloatTensor([blk_cnf_params[0, 2],
                                             blk_cnf_params[0, 3],
                                             blk_cnf_params[0, 4]]).cuda()
                    u2 = torch.FloatTensor([blk_cnf_params[1, 2],
                                             blk_cnf_params[1, 3],
                                             blk_cnf_params[1, 4]]).cuda()
                    ind = 0
                    for i in range(0, int(blk_cnf_params[0, 1].round().item()) ):
                        for j in range(0, int(blk_cnf_params[1, 1].round().item())):
                            batch_prim_param[ind , 1:4] = batch_prim_param[ind,1:4] + (u1*i) + (u2*j)
                            ind += 1
                    
        elif 1 <= blk_pg_ids[1] < translate:
            # one tran + draw; discretize
            num_draw = blk_cnf_params[0, 1]
            num_draw = int(num_draw.round().item())
            assert num_draw <= max_blk_prims, "more than {} draws! {}".format(max_blk_prims,num_draw)
            
            if num_draw <= 0:
                # no program
                batch_prim_idx = [0] * max_blk_prims
                batch_prim_param = blk_cnf_params[0].repeat(max_blk_prims,1)
            
            else:
                # note the number of loops
                batch_prim_idx = ([blk_pg_ids[1]] * num_draw + 
                        [0] * (max_blk_prims - num_draw))
                batch_prim_param = blk_cnf_params[1].repeat(max_blk_prims,1)

                # alter the entries with the loop vairable
                u1 = torch.FloatTensor([blk_cnf_params[0, 2],
                                         blk_cnf_params[0, 3],
                                         blk_cnf_params[0, 4]]).cuda()
            
                for i in range(1,1+num_draw):
                    batch_prim_param[i-1, 1:4] = batch_prim_param[i-1,1:4] + (u1) * (i-1)

                
        elif blk_pg_ids[1] <= 0:
            # no program
            batch_prim_idx = [0] * max_blk_prims
            batch_prim_param = blk_cnf_params[0].repeat(max_blk_prims,1)
            
        else:
            breakpoint()
            raise Exception("unhandled case", blk_pg_ids[1])

    elif blk_pg_ids[0] == rotate:
    # with rotate; discretize
        num_draw = blk_cnf_params[0, 1]
        num_draw = int(num_draw.round().item())
        assert num_draw <= max_blk_prims, "more than {} draws! {}".format(max_blk_prims,num_draw)
        
        if blk_pg_ids[1] == 0 or num_draw <= 0:
                # no program
                batch_prim_idx = [0] * max_blk_prims
                batch_prim_param = blk_cnf_params[0].repeat(max_blk_prims,1)
        else:
            # note the number of loops
            batch_prim_idx = ([blk_pg_ids[1]] * num_draw + 
                    [0] * (max_blk_prims - num_draw))
            batch_prim_param = blk_cnf_params[1].repeat(max_blk_prims,1)

            for i_draw in range(0, num_draw):
                if blk_pg_ids[1] == 10:
                    batch_prim_param[i_draw, 6] = batch_prim_param[i_draw,6] +\
                                                    (blk_cnf_params[0, 2] * (i_draw))

                elif blk_pg_ids[1] == 17:
                    rot_time = min(num_draw, 6)
                    rot_time = max(rot_time, 3)
                    origin_y, origin_z = (blk_cnf_params[1, 2],
                                         blk_cnf_params[1, 3])
                    sin_calc = np.sin(np.deg2rad(360/rot_time*i_draw))
                    cos_calc = np.cos(np.deg2rad(360/rot_time*i_draw))
                    y_init_offset = blk_cnf_params[1, 2] - origin_y
                    z_init_offset = blk_cnf_params[1, 3] - origin_z
                    y_fnal_offset = blk_cnf_params[1, 5] - origin_y
                    z_fnal_offset = blk_cnf_params[1, 6] - origin_z
                    y_init_new = (cos_calc * y_init_offset - sin_calc *\
                                              z_init_offset) + origin_y
                    z_init_new = (sin_calc * y_init_offset + cos_calc *\
                                              z_init_offset) + origin_z
                    y_fnal_new = (cos_calc * y_fnal_offset - sin_calc *\
                                              z_fnal_offset) + origin_y
                    z_fnal_new = (sin_calc * y_fnal_offset + cos_calc *\
                                              z_fnal_offset) + origin_z

                    batch_prim_param[i_draw, 2] = y_init_new
                    batch_prim_param[i_draw, 3] = z_init_new
                    batch_prim_param[i_draw, 5] = y_fnal_new
                    batch_prim_param[i_draw, 6] = z_fnal_new

                else:
                    # no program
                    batch_prim_idx = [0] * max_blk_prims
                    batch_prim_param = blk_cnf_params[0].repeat(max_blk_prims,1)

#                 else:
#                     breakpoint()
#                     raise Exception("unhandled case")
            
    elif 1 <= blk_pg_ids[0] < translate:
    # without loops
        batch_prim_idx = [blk_pg_ids[0]] + [0] * (max_blk_prims-1)
        batch_prim_param = blk_cnf_params[0].repeat(max_blk_prims,1)
    
    elif blk_pg_ids[0] == 0:
    # no program
        batch_prim_idx = [0] * max_blk_prims
        batch_prim_param = blk_cnf_params[0].repeat(max_blk_prims,1)
    
    else:
    # exceptional case
        raise Exception("unhandled case: {}".format(blk_pg_ids))
    
    return batch_prim_idx, batch_prim_param

def expend_loops(pg_ids, pg_cnf_params):
    '''
    Given the program ids, confidence scores, parameters, replace the
    for-loops in each block into a list of primitives.
    '''
    # batch of blocks of programs (each block has 3 lines)
    batch_expend_pgm_idx = []
    batch_expend_pgm_params = []
    bsz, n_blk, _ = pg_ids.shape
    
    for idx,i in enumerate(range(bsz)):
        # blocks of programs
        blks_expend_pgm_idx = []
        blks_expend_params = []
        
        for idj,j in enumerate(range(n_blk)):
            
            # index to the step program ids and parameters
            blk_pg_ids = pg_ids[i, j]
            blk_cnf_params = pg_cnf_params[i, j]
            
            blk_expend_pgm_idx, blk_expend_params = expend_loop_blk(
                                                    blk_pg_ids,
                                                    blk_cnf_params)
            
            # store
            blks_expend_pgm_idx.append(blk_expend_pgm_idx)
            blks_expend_params.append(blk_expend_params)
    
        batch_expend_pgm_idx.append(blks_expend_pgm_idx)
        batch_expend_pgm_params.append(blks_expend_params)

    return batch_expend_pgm_idx, batch_expend_pgm_params

def expend_pgm_params(pgms, params, sample=False):
    '''
    Takes p_g (#_batches * #_blocks * #_Steps * #_statements), 
    p_m (#_batches * #_blocks * #_steps * #_max_parameters) into:
    an expended tensor of only shape primitives (#_batches *
    #_blocks * #_max_primitives * #_max_parameters).
    '''
    # softmax over possible statements
    soft = nn.Softmax(dim=3)
    pgms = soft(pgms)

    # use sample instead
    if sample:
        prob_dist = torch.distributions.Categorical(pgms)
        pg_ids = prob_dist.sample()
        pg_conf = torch.zeros(pg_ids.shape).cuda()
#         for i,pg in enumerate(pg_ids): print("sampled pg:",i, pg)
    else:
        pg_conf, pg_ids = torch.max(pgms, dim = 3)
        
    # stack pg_conf with params
    # #bsz * #blks * #steps * ([conf_sc] + params)
    pg_cnf_params = torch.cat([pg_conf.unsqueeze(-1), params], dim=3)
    
    # assume each code block, after expended, contains <= 6 draw 
    # statements.
    batch_prim_idx, batch_prim_param = expend_loops(pg_ids, pg_cnf_params)
    
    return batch_prim_idx, batch_prim_param

def clean_batch_program(batch_prim_idx, batch_prim_param):
    '''
    args:
    batch_prim_idx list(list(int)): representing the padding primitive program blocks.
    batch_prim_param list(list(tensor)): representing the confidence value and parameters
        for the corresponding shape primitives.
    returns:
    batch_clean_idx list(list(int)):
    batch_clean_parma list(tensor): tensor.shape = # primitives in the particular shape * # parameters
        the same list with empty block of the program (those with program id 0) removed.
        Two lists should have the same length.
    '''
    batch_clean_idx = []
    batch_clean_param = []
    
    # check the batch size
    assert len(batch_prim_idx) == len(batch_prim_param)
    
    for single_shp_idx, single_shp_param in zip(batch_prim_idx, batch_prim_param):
        # concate the lists for each shape
        single_shp_idx = list(np.array(single_shp_idx).flat)
        single_shp_param = torch.cat(single_shp_param)
        
        # store the indexes of desired parameters in the concatnated param tensor
        keep_ids = [i for i,element in enumerate(single_shp_idx) if element != 0]
        
        shape_clean_idx = [single_shp_idx[keep_id] for keep_id in keep_ids] # store the program indexes
        try:
            shape_clean_param = single_shp_param.index_select(dim=0, index = torch.tensor(keep_ids).cuda())
        except:
            print("keep ids:",keep_ids)
        
        assert len(shape_clean_idx) == shape_clean_param.shape[0]
        batch_clean_idx.append(shape_clean_idx)
        batch_clean_param.append(shape_clean_param)
    
    return batch_clean_idx, batch_clean_param
