from .utils import *
import math
import os
from .label_config import max_step, max_param
from misc import get_distance_to_center

###################################
# generate tables with  2 vertical legs
# tabletop can be: square, circle, rectangle
# max steps: 5
###################################


def generate_single(d):

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []

    # sample tabletop thickness
    p = np.random.rand()
    if p < 0.8:
        top_t = np.random.randint(1, 3)
    else:
        top_t = np.random.randint(3, 4)
    # sample leg height
    leg_h = 2 * np.random.randint(4, 13) - top_t

    total_height = leg_h + top_t
    leg_start = - int(total_height / 2)
    tabletop_start = leg_start + leg_h

    # sample the tabletop
    p = np.random.rand()
    top_type = -1
    if p < 0.6:
        # rectangle tabletop
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
        top_type = 0
    elif p < 0.8:
        # square tabletop
        q = np.random.rand()
        if q < 0.75:
            top_r = np.random.randint(11, 13)
        elif q < 0.95:
            top_r = np.random.randint(9, 11)
        else:
            top_r = 8
        data, step = draw_square_top(data, tabletop_start, 0, 0, top_t, top_r)
        steps.append(step)
        top_r1 = top_r
        top_r2 = top_r
        top_type = 1
    else:
        # circle tabletop
        q = np.random.rand()
        if q < 0.75:
            top_r = np.random.randint(11, 13)
        elif q < 0.95:
            top_r = np.random.randint(9, 11)
        else:
            top_r = 8
        data, step = draw_circle_top(data, tabletop_start, 0, 0, top_t, top_r)
        steps.append(step)
        top_r1 = top_r
        top_r2 = top_r
        top_type = 2

    p = np.random.rand()
    if p < 0.45:
        leg_w = 2
        leg_l = 2
    elif p < 0.6:
        leg_w = 1
        leg_l = 1
    elif p < 0.75:
        leg_w = 1
        leg_l = 2
    elif p < 0.9:
        leg_w = 2
        leg_l = 1
    else:
        leg_w = 3
        leg_l = 3

    s1 = np.random.randint(-1, 1)
    s2 = np.random.randint(top_r2-5, top_r2-leg_l)
    data, step = draw_vertical_leg(data, leg_start, s1-leg_w, -s2-leg_l, leg_h, leg_w, leg_l)
    steps.append(step)
    data, step = draw_vertical_leg(data, leg_start, s1-leg_w, s2, leg_h, leg_w, leg_l)
    steps.append(step)

    r1 = np.random.randint(max(top_r1-5, 3), top_r1)
    heigth = np.random.randint(1, 3)
    data, step = draw_horizontal_bar(data, leg_start, -r1, -s2 - leg_l, heigth, 2 * r1, leg_l)
    steps.append(step)
    data, step = draw_horizontal_bar(data, leg_start, -r1, s2, heigth, 2 * r1, leg_l)
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

