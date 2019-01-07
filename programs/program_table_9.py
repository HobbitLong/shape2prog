from .utils import *
import math
import os
from .label_config import max_step, max_param
from misc import get_distance_to_center

###################################
# generate working desks
# max steps: 6
###################################


def generate_single(d):

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []

    # sample tabletop thickness
    p = np.random.rand()
    if p < 0.75:
        top_t = np.random.randint(1, 3)
    else:
        top_t = 3
    # sample board height
    board_h = 2 * np.random.randint(5, 10) - top_t

    total_height = board_h + top_t
    board_start = - int(total_height / 2)
    tabletop_start = board_start + board_h

    # sample the rectangle tabletop
    top_r1 = np.random.randint(5, 9)
    top_r2 = np.random.randint(11, 13)
    data, step = draw_rectangle_top(data, tabletop_start, 0, 0, top_t, top_r1, top_r2)
    steps.append(step)

    # sample the sideboards
    p = np.random.rand()
    if p < 0.75:
        board_r1 = top_r1
    else:
        board_r1 = top_r1 - 1
    p = np.random.rand()
    if p < 0.75:
        board_r2 = np.random.randint(1, 3)
    else:
        board_r2 = np.random.randint(3, 4)
    s2 = top_r2 - board_r2
    data, step = draw_sideboard(data, board_start, 0, -s2 - board_r2, board_h, board_r1, board_r2)
    steps.append(step)
    data, step = draw_sideboard(data, board_start, 0, s2, board_h, board_r1, board_r2)
    steps.append(step)

    # sample the vertboard
    p = np.random.rand()
    if p < 0.7:
        v_r1 = np.random.randint(1, 3)
        v_r2 = 2 * top_r2
        v_start = board_start + np.random.randint(0, 5)
        v_t = tabletop_start - v_start
        v_s1 = top_r1 - v_r1 + np.random.randint(-1, 1)
        v_s2 = - top_r2
        data, step = draw_vertboard(data, v_start, v_s1, v_s2, v_t, v_r1, v_r2)
        steps.append(step)

    # sample the locker
    l_h = np.random.randint(board_h - 5, board_h + 1)
    l_r1 = np.random.randint(5, 2*board_r1)
    l_r2 = np.random.randint(4, 8)
    l_start = board_start + (board_h - l_h)
    l_s1 = - board_r1 + np.random.randint(0, min(2, 2*board_r1 - l_r1))

    p = np.random.rand()
    if p < 0.5:
        l_s2 = s2 - l_r2
        data, step = draw_locker(data, l_start, l_s1, l_s2, l_h, l_r1, l_r2)
        steps.append(step)
    elif p < 0.7:
        l_s2 = - s2
        data, step = draw_locker(data, l_start, l_s1, l_s2, l_h, l_r1, l_r2)
        steps.append(step)
    else:
        l_s2 = - s2
        data, step = draw_locker(data, l_start, l_s1, l_s2, l_h, l_r1, l_r2)
        steps.append(step)
        l_s2 = s2 - l_r2
        data, step = draw_locker(data, l_start, l_s1, l_s2, l_h, l_r1, l_r2)
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

