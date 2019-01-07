from .utils import *
import math
import os
from .label_config import max_step, max_param
from misc import get_distance_to_center
from .complex_base import draw_new_base

###################################
# generate club chairs
# seat-top can be: square, circle, rectangle
# supports can be: circle, square
# bottom bases can be: circle, square, cross
# back can be tilted
# max steps: 9
###################################


def generate_single(d):
    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []

    p = np.random.rand()
    if p < 0.8:
        top_t = np.random.choice([2,3,4],1)[0]
    else:
        top_t = np.random.choice([3,4,5],1)[0]
    # leg_h = np.random.randint(7, 10) - top_t
    leg_h = np.random.randint(top_t+4, top_t+7)-top_t
    total_height = leg_h + 1

    entire_height = np.random.choice([22, 23, 24, 25, 26], 1)[0]
    back_height =  entire_height - total_height
    leg_start = -int(entire_height/2) - top_t + 1
    seattop_start = leg_start + leg_h

    # leg_start = -total_height
    # seattop_start = leg_start + leg_h

    tilt_amount = np.random.choice([0,1,2,3,4], 1)[0]

    beam_offset = np.random.choice([-1,0,1], 1)[0]

    seattop_offset = -int(np.rint(tilt_amount/2))

    if tilt_amount!=0:
        back_thickness = np.random.choice([2,3,3], 1)[0]
    else:
        back_thickness = np.random.choice([2,3], 1)[0]

    # back_height = total_height

    # sample the seattop
    p = np.random.rand()
    top_type = -1

    if p < 0.5:
        # rectangle seattop
        top_r2 = np.random.randint(6, 12)
        top_r1 = top_r2 - np.random.choice([1, 2],1)[0]
        top_r = top_r1
        # data, step = draw_rectangle_top(data, seattop_start, top_r1, top_r2, top_t)
        data, step = draw_rectangle_top(data, seattop_start, seattop_offset, 0, top_t, top_r1, top_r2)
        steps.append(step)
        top_type = 0
    else:
        # square seattop
        q = np.random.rand()
        if q < 0.75:
            top_r = np.random.randint(6, 9)
        elif q < 0.95:
            top_r = np.random.randint(10, 11)
        else:
            top_r = 11
        # data, step = draw_square_top(data, seattop_start, top_r, top_t)
        data, step = draw_square_top(data, seattop_start, seattop_offset, 0, top_t, top_r)
        steps.append(step)
        top_type = 1
    # else:
    #     # circle seattop
    #     q = np.random.rand()
    #     if q < 0.75:
    #         top_r = np.random.randint(6, 9)
    #     elif q < 0.95:
    #         top_r = np.random.randint(10, 11)
    #     else:
    #         top_r = 11
    #     # data, step = draw_circle_top(data, seattop_start, top_r, top_t, d)
    #     data, step = draw_circle_top(data, seattop_start, seattop_offset, 0, top_t, top_r)
    #     steps.append(step)
    #     top_type = 2

    p = np.random.rand()
    if p < 0.4:
        # sample circle support
        q = np.random.rand()
        if q < 0.8:
            support_r = np.random.randint(1, 3)
        else:
            support_r = np.random.randint(3, 4)
        # data, step = draw_circle_support(data, leg_start, support_r, leg_h + top_t, d)
        data, step = draw_circle_support(data, leg_start, seattop_offset + beam_offset, 0, leg_h + top_t, support_r)
        steps.append(step)
    else:
        # sample square support
        # q = np.random.rand()
        # if q < 0.75:
        #     support_r = np.random.randint(2, 3)
        # else:
        #     support_r = np.random.randint(3, 4)
        support_r = np.random.randint(max(4, top_r-5), top_r+1)
        support_r = max(support_r, np.random.randint(max(4, top_r-5), top_r+1))
        # data, step = draw_square_support(data, leg_start, support_r, leg_h + top_t)
        data, step = draw_square_support(data, leg_start, seattop_offset + beam_offset, 0, leg_h + top_t, support_r)
        steps.append(step)

    # sample bottom base
    p = np.random.rand()
    if p < 0.75:
        base_h = 1
    else:
        base_h = 2
    p = np.random.rand()
    if p<0.5:
        pass
    if p < 0.5:
        # sample circle base
        base_r = np.random.randint(5, min(5 + 2, top_r + 4, 10))

        # data, step = draw_circle_base(data, leg_start, base_r, base_h, d)
        data, step = draw_circle_base(data, leg_start, seattop_offset + beam_offset, 0, base_h, base_r)
        steps.append(step)
    elif p < 0.8:
        # # sample cross base
        # base_r = np.random.randint(5, min(5 + 2, top_r + 4, 10))
        # for angle in range(4):
        #     # data, step = draw_cross_base(data, leg_start, base_r, base_h, angle)
        #     data, step = draw_cross_base(data, leg_start, seattop_offset + beam_offset, 0, base_h, base_r, angle)
        #     steps.append(step)
        # sample cross base
        base_r = np.random.randint(7, 12)
        count = np.random.choice([3, 3, 4, 4, 5, 6])
        leg_thickness = 1
        leg_end_offset = -np.random.choice([0, 1, 2], 1)[0]
        current_angle = np.random.choice([90, 135], 1)[0]
        x1 = leg_start-leg_thickness
        y1 = seattop_offset + beam_offset
        z1 = 0
        x2 = x1 +leg_end_offset
        y2 = np.rint(-base_r * np.sin(np.deg2rad(current_angle))) + y1
        z2 = np.rint(base_r * np.cos(np.deg2rad(current_angle))) + z1
        data, step = draw_new_base(data, x1, y1, z1, x2, y2, z2, count)
        for step_i in step:
            steps.append(step_i)
        # base_r = np.random.randint(7, 12)
        # base_r = min(base_r, top_r)
        # count = np.random.choice([3, 3, 4, 4, 5])
        # leg_thickness = 1
        # leg_end_offset = -np.random.choice([1, 2], 1)[0] - leg_thickness
        # draw_new_base(data, steps, leg_start-leg_thickness, seattop_offset + beam_offset , 0, leg_start+leg_end_offset, base_r, leg_thickness, count)
    else:
        # sample square base
        base_r = np.random.randint(5, min(5 + 2, top_r + 4, 10))
        # data, step = draw_square_base(data, leg_start, base_r, base_h)
        data, step = draw_square_base(data, leg_start, seattop_offset + beam_offset, 0, base_h, base_r)
        steps.append(step)

    if top_type == 0:
        s1 = top_r1
        s2 = top_r2
        data, step = draw_tilt_back(data, leg_start + leg_h + top_t, +s1 + np.random.choice([-1, 0], 1)[0] + seattop_offset, -s2, back_height, back_thickness, 2 * s2, tilt_amount)
        steps.append(step)

    if top_type == 1:
        s1 = top_r
        s2 = top_r
        data, step = draw_tilt_back(data, leg_start + leg_h + top_t, +s1 + np.random.choice([-1, 0], 1)[0] + seattop_offset, -s2, back_height, back_thickness, 2 * s2, tilt_amount)
        steps.append(step)

    if top_type == 2:
        s1 = int(round(top_r / math.sqrt(2)))
        s2 = int(round(top_r / math.sqrt(2)))
        data, step = draw_tilt_back(data, leg_start + leg_h + top_t, +s1 + np.random.choice([-1, 0], 1)[0] + seattop_offset, -s2, back_height, back_thickness, 2 * s2, tilt_amount)
        steps.append(step)

    arm_rest_vert_offset = np.random.choice([0,0,1,2,3], 1)[0]
    arm_rest_thickness = np.random.choice([3,4,5], 1)[0]
    arm_front_shift = -np.random.choice([0,1,2], 1)[0]
    arm_max_height = leg_h + top_t + back_height - np.random.choice([1,2], 1)[0] - arm_rest_vert_offset
    data, step = draw_sideboard(data, leg_start+arm_rest_vert_offset, seattop_offset+arm_front_shift+tilt_amount, s2, arm_max_height, top_r, arm_rest_thickness)
    steps.append(step)
    data, step = draw_sideboard(data, leg_start+arm_rest_vert_offset, seattop_offset+arm_front_shift+tilt_amount, -s2-arm_rest_thickness, arm_max_height,top_r, arm_rest_thickness)
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


from pprint import pprint


def check_max_steps():

    d = get_distance_to_center()

    step = 0
    for i in range(200):
        x, y = generate_single(d)
        if len(y) > step:
            step = len(y)
    print("Maximum Steps: " + str(step) + " " + os.path.basename(__file__))

    return step
