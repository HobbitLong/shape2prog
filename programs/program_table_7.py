from .utils import *
import math
import os
from .label_config import max_step, max_param
from misc import get_distance_to_center

###################################
# generate tables with 4 vertical legs
# tabletop can be: square, circle, rectangle
# horizontal bars connect legs
# max steps: 9
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
        top_type = 2

    # sample legs
    if top_type == 0:
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
        p = np.random.rand()
        if p < 0.5:
            shrink_w = 0
            shrink_l = 0
        else:
            shrink_w = np.random.randint(0, min(2, top_r1-leg_w))
            shrink_l = np.random.randint(0, min(4, top_r2-leg_l))
        s1 = top_r1 - shrink_w - leg_w
        s2 = top_r2 - shrink_l - leg_l
        data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, s1, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, s1, s2, leg_h + top_t, leg_w, leg_l)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w, s2, leg_h + top_t, leg_w, leg_l)
        steps.append(step)
    if top_type == 1:
        p = np.random.rand()
        if p < 0.6:
            leg_w = 2
            leg_l = 2
        elif p < 0.85:
            leg_w = 1
            leg_l = 1
        else:
            leg_w = 3
            leg_l = 3
        p = np.random.rand()
        if p < 0.7:
            shrink_w = 0
            shrink_l = 0
        else:
            shrink_w = np.random.randint(1, 3)
            shrink_l = shrink_w
        s1 = top_r - shrink_w - leg_w
        s2 = top_r - shrink_l - leg_l
        data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, s1, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, s1, s2, leg_h + top_t, leg_w, leg_l)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w, s2, leg_h + top_t, leg_w, leg_l)
        steps.append(step)
    if top_type == 2:
        p = np.random.rand()
        if p < 0.6:
            leg_w = 2
            leg_l = 2
        elif p < 0.85:
            leg_w = 1
            leg_l = 1
        else:
            leg_w = 3
            leg_l = 3
        p = np.random.rand()
        # if p < 0.7:
        if p < 1.1:
            shrink_w = 0
            shrink_l = 0
        else:
            shrink_w = np.random.randint(1, 3)
            shrink_l = shrink_w
        s1 = int(round(top_r / math.sqrt(2))) - shrink_w - leg_w
        s2 = int(round(top_r / math.sqrt(2))) - shrink_l - leg_l
        data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, s1, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, s1, s2, leg_h + top_t, leg_w, leg_l)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w, s2, leg_h + top_t, leg_w, leg_l)
        steps.append(step)

    h_bar_t = np.random.randint(1, 3)
    h_bar_start = leg_start + np.random.randint(0, min(5, leg_h-h_bar_t))

    p = np.random.rand()
    if p < 0.7:
        data, step = draw_horizontal_bar(data, h_bar_start, -s1 - leg_w, -s2 - leg_l, h_bar_t, 2 * (s1 + leg_w), leg_l)
        steps.append(step)
        data, step = draw_horizontal_bar(data, h_bar_start, -s1 - leg_w, s2, h_bar_t, 2 * (s1 + leg_w), leg_l)
        steps.append(step)
        # third single bar
        start = np.random.randint(-s1-leg_w, s1 + 1)
        q = np.random.rand()
        if q < 0.5:
            width = leg_w
        else:
            width = np.random.randint(1, min(4, 2*s1))
        data, step = draw_horizontal_bar(data, h_bar_start, start, -s2 - leg_l, h_bar_t, width, 2 * (s2 + leg_l))
        steps.append(step)
    else:
        data, step = draw_horizontal_bar(data, h_bar_start, -s1 - leg_w, -s2 - leg_l, h_bar_t, 2 * (s1 + leg_w), leg_l)
        steps.append(step)
        data, step = draw_horizontal_bar(data, h_bar_start, -s1 - leg_w, s2, h_bar_t, 2 * (s1 + leg_w), leg_l)
        steps.append(step)
        data, step = draw_horizontal_bar(data, h_bar_start, -s1 - leg_w, -s2 - leg_l, h_bar_t, leg_w, 2 * (s2 + leg_l))
        steps.append(step)
        data, step = draw_horizontal_bar(data, h_bar_start, s1, -s2 - leg_l, h_bar_t, leg_w, 2 * (s2 + leg_l))
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

