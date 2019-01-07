from .utils import *
import math
import os
from .label_config import max_step, max_param
from misc import get_distance_to_center

###################################
# generate tables with sideboards
# might have second layer
# max steps: 4
###################################


def generate_single(d):

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []

    # sample tabletop thickness
    p = np.random.rand()
    if p < 0.75:
        top_t = np.random.randint(1, 3)
    else:
        top_t = np.random.randint(3, 4)
    # sample board height
    board_h = 2 * np.random.randint(4, 13) - top_t

    total_height = board_h + top_t
    board_start = - int(total_height / 2)
    tabletop_start = board_start + board_h

    # sample the rectangle tabletop
    top_r1 = np.random.randint(4, 13)
    top_r2 = np.random.randint(10, 13)
    if top_r1 > top_r2:
        tmp = top_r1
        top_r1 = top_r2
        top_r2 = tmp
    if top_r1 == top_r2:
        top_r1 -= 1
    data, step = draw_rectangle_top(data, tabletop_start, 0, 0, top_t, top_r1, top_r2)
    steps.append(step)

    # sample the sideboards
    p = np.random.rand()
    if p < 0.75:
        board_r1 = top_r1
    else:
        board_r1 = np.random.randint(max(3, top_r1-3), top_r1)
    p = np.random.rand()
    if p < 0.75:
        board_r2 = np.random.randint(1, 3)
    else:
        board_r2 = np.random.randint(3, 4)
    p = np.random.rand()
    if p < 0.75:
        s2 = top_r2 - board_r2
    else:
        s2 = np.random.randint(top_r2-board_r2-4, top_r2-board_r2)
    data, step = draw_sideboard(data, board_start, 0, -s2 - board_r2, board_h, board_r1, board_r2)
    steps.append(step)
    data, step = draw_sideboard(data, board_start, 0, s2, board_h, board_r1, board_r2)
    steps.append(step)

    # sample the second layer
    p = np.random.rand()
    if p < 0.5:
        p1 = np.random.rand()
        if p1 < 0.5:
            layer_t = top_t
        else:
            q = np.random.rand()
            if q < 0.8:
                layer_t = np.random.randint(1, 3)
            else:
                layer_t = np.random.randint(3, 4)
        p2 = np.random.rand()
        if p2 < 0.5:
            layer_r1 = board_r1
            layer_r2 = s2 + board_r2
        else:
            layer_r1 = np.random.randint(max(3, board_r1-3), max(4, board_r1))
            layer_r2 = s2 + board_r2
        layer_start = np.random.randint(board_start, tabletop_start - layer_t)
        data, step = draw_middle_rect_layer(data, layer_start, 0, 0, layer_t, layer_r1, layer_r2)
        steps.append(step)

    return data, steps


def generate_batch(num):
    data = np.zeros((num, 32, 32, 32), dtype=np.uint8)
    label = np.zeros((num, max_step, max_param), dtype=np.int32)

    d = get_distance_to_center()

    for i in range(num):
        x, y = generate_single(d)
        data[i, ...] = x

        for k1 in range(len(y)):
            label[i, k1, 0:len(y[k1])] = y[k1]

    return data, label


def check_max_steps():

    d = get_distance_to_center()

    step = 0
    for i in range(200):
        x, y = generate_single(d)
        if len(y) > step:
            step = len(y)
    print("Maximum Steps: " + str(step) + " " + os.path.basename(__file__))

    return step

