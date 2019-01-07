from __future__ import print_function

import os
import h5py
import numpy as np

from programs.program_table_1 import generate_batch as table_gen1
from programs.program_table_2 import generate_batch as table_gen2
from programs.program_table_3 import generate_batch as table_gen3
from programs.program_table_4 import generate_batch as table_gen4
from programs.program_table_5 import generate_batch as table_gen5
from programs.program_table_6 import generate_batch as table_gen6
from programs.program_table_7 import generate_batch as table_gen7
from programs.program_table_8 import generate_batch as table_gen8
from programs.program_table_9 import generate_batch as table_gen9
from programs.program_table_10 import generate_batch as table_gen10

from programs.program_chair_1 import generate_batch as chair_gen1
from programs.program_chair_2 import generate_batch as chair_gen2
from programs.program_chair_3 import generate_batch as chair_gen3
from programs.program_chair_4 import generate_batch as chair_gen4

from programs.loop_gen import gen_loop, decode_loop
from programs.label_config import max_for_step


def gen_loop_batch(batch_data):
    """
    roll trace sets into loops
    """
    d1, d2, d3 = batch_data.shape
    res = np.zeros((d1, max_for_step, d3), dtype=np.int32)
    for i in range(d1):
        data = batch_data[i]
        data_loop = gen_loop(data)
        len1 = len(data_loop)
        if len1 <= max_for_step:
            res[i, :len1, :] = np.asarray(data_loop).astype(np.int32)
        else:
            if np.sum(data_loop[:, 0] != 0) > 18:
                print(data)
                print(data_loop)
            # print(len1)
            # print(data_loop)
            res[i, :, :] = np.asarray(data_loop[:max_for_step]).astype(np.int32)
    return res


def decode_loop_batch(batch_data):
    """
    unroll loop into trace sets
    """
    d1, d2, d3 = batch_data.shape
    res = np.zeros((d1, max_for_step, d3), dtype=np.int32)
    for i in range(d1):
        data = batch_data[i]
        data_free = decode_loop(data)
        len1 = len(data_free)
        if len1 <= max_for_step:
            res[i, :len1, :] = data_free.astype(np.int32)
        else:
            if np.sum(data_free[:, 0] != 0) > 18:
                print(data)
                print(data_free)
            # print(data_free)
            # print(data_free)
            res[i, :, :] = data_free[:max_for_step].astype(np.int32)
    return res


def synthesize_data():
    """
    synthesize the (shape, program) pairs
    :return: train_shape, train_prog, val_shape, val_prog
    """
    # == training data ==

    # table
    data1_t, label1_t = table_gen1(4000)
    data2_t, label2_t = table_gen2(4000)
    data3_t, label3_t = table_gen3(4000)
    data4_t, label4_t = table_gen4(4000)
    data5_t, label5_t = table_gen5(4000)
    data6_t, label6_t = table_gen6(4000)
    data7_t, label7_t = table_gen7(4000)
    data8_t, label8_t = table_gen8(4000)
    data9_t, label9_t = table_gen9(4000)
    data10_t, label10_t = table_gen10(4000)

    label1_t = gen_loop_batch(label1_t)
    label2_t = gen_loop_batch(label2_t)
    label3_t = gen_loop_batch(label3_t)
    label4_t = gen_loop_batch(label4_t)
    label5_t = gen_loop_batch(label5_t)
    label6_t = gen_loop_batch(label6_t)
    label7_t = gen_loop_batch(label7_t)
    label8_t = gen_loop_batch(label8_t)
    label9_t = gen_loop_batch(label9_t)
    label10_t = gen_loop_batch(label10_t)

    # chair
    data1_c, label1_c = chair_gen1(50000)
    data2_c, label2_c = chair_gen2(20000)
    data5_c, label5_c = chair_gen3(5000)
    data6_c, label6_c = chair_gen4(5000)

    label1_c = gen_loop_batch(label1_c)
    label2_c = gen_loop_batch(label2_c)
    label5_c = gen_loop_batch(label5_c)
    label6_c = gen_loop_batch(label6_c)

    train_shape = np.vstack((data1_t, data2_t, data3_t, data4_t, data5_t,
                             data6_t, data7_t, data8_t, data9_t, data10_t,
                             data1_c, data2_c, data5_c, data6_c))
    train_prog = np.vstack((label1_t, label2_t, label3_t, label4_t, label5_t,
                            label6_t, label7_t, label8_t, label9_t, label10_t,
                            label1_c, label2_c, label5_c, label6_c))

    # == validation data ==

    # table
    data1_t, label1_t = table_gen1(500)
    data2_t, label2_t = table_gen2(500)
    data3_t, label3_t = table_gen3(500)
    data4_t, label4_t = table_gen4(500)
    data5_t, label5_t = table_gen5(500)
    data6_t, label6_t = table_gen6(500)
    data7_t, label7_t = table_gen7(500)
    data8_t, label8_t = table_gen8(500)
    data9_t, label9_t = table_gen9(500)
    data10_t, label10_t = table_gen10(500)

    label1_t = gen_loop_batch(label1_t)
    label2_t = gen_loop_batch(label2_t)
    label3_t = gen_loop_batch(label3_t)
    label4_t = gen_loop_batch(label4_t)
    label5_t = gen_loop_batch(label5_t)
    label6_t = gen_loop_batch(label6_t)
    label7_t = gen_loop_batch(label7_t)
    label8_t = gen_loop_batch(label8_t)
    label9_t = gen_loop_batch(label9_t)
    label10_t = gen_loop_batch(label10_t)

    # chair
    data1_c, label1_c = chair_gen1(3500)
    data2_c, label2_c = chair_gen2(1000)
    data5_c, label5_c = chair_gen3(250)
    data6_c, label6_c = chair_gen4(250)

    label1_c = gen_loop_batch(label1_c)
    label2_c = gen_loop_batch(label2_c)
    label5_c = gen_loop_batch(label5_c)
    label6_c = gen_loop_batch(label6_c)

    val_shape = np.vstack((data1_t, data2_t, data3_t, data4_t, data5_t,
                           data6_t, data7_t, data8_t, data9_t, data10_t,
                           data1_c, data2_c, data5_c, data6_c))
    val_prog = np.vstack((label1_t, label2_t, label3_t, label4_t, label5_t,
                          label6_t, label7_t, label8_t, label9_t, label10_t,
                          label1_c, label2_c, label5_c, label6_c))

    return train_shape, train_prog, val_shape, val_prog


if __name__ == '__main__':

    print('==> synthesizing (shape, program) pairs')
    train_x, train_y, val_x, val_y = synthesize_data()
    print('Done')

    if not os.path.isdir('./data'):
        os.makedirs('./data')
    train_file = './data/train_shapes.h5'
    val_file = './data/val_shapes.h5'

    print('==> saving data')

    f_train = h5py.File(train_file, 'w')
    f_train['data'] = train_x
    f_train['label'] = train_y
    f_train.close()

    f_val = h5py.File(val_file, 'w')
    f_val['data'] = val_x
    f_val['label'] = val_y
    f_val.close()

    print('Done')
