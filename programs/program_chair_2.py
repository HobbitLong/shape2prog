from .utils import *
from .complex_base import draw_new_base
import math
import os
from .label_config import max_step, max_param
# from misc import get_distance_to_center

def get_distance_to_center():
    x = np.arange(32)
    y = np.arange(32)
    xx, yy = np.meshgrid(x, y)
    xx = xx + 0.5
    yy = yy + 0.5
    d = np.sqrt(np.square(xx - int(32 / 2)) + np.square(yy - int(32 / 2)))
    return d


def generate_single(d):
    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []

    p = np.random.rand()
    if p < 0.8:
        top_t = np.random.choice([1,2,3,4],1)[0]
    else:
        top_t = np.random.choice([3,4,5],1)[0]
    # leg_h = np.random.randint(7, 10) - top_t

    club_chair = np.random.choice([0,0,0,1], 1)[0]
    club_chair = 0
    # sample the seattop
    if club_chair:
        top_type = np.random.choice([0,1], 1)[0]
    else:
        top_type = np.random.choice([0, 1, 2], 1)[0]
        # top_type = 2
    # top_type = 0

    if club_chair:
        leg_h = np.random.randint(5, 12) - top_t
    else:
        leg_h = np.random.randint(8, 14) - top_t


    total_height = leg_h + top_t
    entire_height = np.random.choice([22, 23, 24, 25, 26], 1)[0]
    back_height =  entire_height - total_height
    leg_start = -int(entire_height/2)
    seattop_start = leg_start + leg_h

    # leg_start = -total_height

    # seattop_start = leg_start + leg_h


    back_type = np.random.choice([0, 1], 1)[0]
    # back_type = 0

    back_tilt_amount = np.random.choice([0,1,2,3,4], 1)[0]

    beam_offset = np.random.choice([-1,0,1], 1)[0]

    if back_tilt_amount!=0:
        back_thickness = np.random.choice([2,3,3], 1)[0]
    else:
        back_thickness = np.random.choice([1,2,3], 1)[0]

    seattop_offset = -int(np.rint(back_tilt_amount/2))

    # back_height = total_height

    top_r = np.random.randint(6, 12)
    top_r2 = top_r
    top_r1 = top_r2 - np.random.choice([1, 2],1)[0]
    if top_type == 0:
        data, step = draw_rectangle_top(data, seattop_start, seattop_offset, 0, top_t, top_r1, top_r2)
        steps.append(step)
    elif top_type == 1:
        data, step = draw_square_top(data, seattop_start, seattop_offset, 0, top_t, top_r)
        steps.append(step)
    elif top_type == 2:
        data, step = draw_circle_top(data, seattop_start, seattop_offset, 0, top_t, top_r)
        steps.append(step)

    if club_chair:
        leg_type = np.random.choice([0, 1, 2], 1)[0]
        if leg_type==0:
            # sample 4 legs
            p = np.random.rand()
            if p < 0.45:
                leg_w = 1
                leg_l = 1
            elif p < 0.725:
                leg_w = 1
                leg_l = 2
            else:
                leg_w = 2
                leg_l = 1
            p = np.random.rand()
            if p < 0.5:
                shrink_w = 0
                shrink_l = 0
            else:
                shrink_w = np.random.randint(0, min(1, top_r1-leg_w))
                shrink_l = np.random.randint(0, min(2, top_r2-leg_l))
            if top_type == 0:
                s1 = top_r1 - shrink_w - leg_w
                s2 = top_r2 - shrink_l - leg_l
            elif top_type == 1:
                s1 = top_r - shrink_w - leg_w
                s2 = top_r - shrink_l - leg_l
            data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w + seattop_offset, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
            data, step = draw_vertical_leg(data, leg_start, +s1  + seattop_offset, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
            data, step = draw_vertical_leg(data, leg_start, +s1 + seattop_offset, +s2, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
            data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w + seattop_offset, +s2, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
        elif leg_type==1:
            # sample circle support
            support_r = np.random.randint(1, 4)
            data, step = draw_circle_support(data, leg_start, seattop_offset + beam_offset, 0, leg_h + top_t, support_r)
            steps.append(step)
        elif leg_type==2:
            # sample square support
            support_r = np.random.randint(3, 5)
            # data, step = draw_square_support(data, leg_start, support_r, leg_h + top_t)
            data, step = draw_square_support(data, leg_start, seattop_offset + beam_offset, 0, leg_h + top_t, support_r)
            steps.append(step)
    elif not club_chair:
        leg_type = np.random.choice([1], 1)[0]
        # leg_type = 1
        p = np.random.rand()
        if p < 0.45:
            leg_w = 1
            leg_l = 1
        elif p < 0.725:
            leg_w = 1
            leg_l = 2
        else:
            leg_w = 2
            leg_l = 1
        p = np.random.rand()
        if p < 0.5:
            shrink_w = 0
            shrink_l = 0
        else:
            shrink_w = np.random.randint(0, min(1, top_r1-leg_w))
            shrink_l = np.random.randint(0, min(2, top_r2-leg_l))
        if top_type == 0:
            s1 = top_r1 - shrink_w - leg_w
            s2 = top_r2 - shrink_l - leg_l
        elif top_type == 1:
            s1 = top_r - shrink_w - leg_w
            s2 = top_r - shrink_l - leg_l
        elif top_type == 2:
            s1 = int(round(top_r / math.sqrt(2))) - shrink_w - leg_w
            s2 = int(round(top_r / math.sqrt(2))) - shrink_l - leg_l

        if leg_type==0:
            # sample 4 legs
            data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w + seattop_offset, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
            data, step = draw_vertical_leg(data, leg_start, +s1  + seattop_offset, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
            data, step = draw_vertical_leg(data, leg_start, +s1 + seattop_offset, +s2, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
            data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w + seattop_offset, +s2, leg_h + top_t, leg_w, leg_l)
            steps.append(step)

        elif leg_type==1:
            # sample circle support
            support_r = np.random.randint(1, 3)
            data, step = draw_circle_support(data, leg_start, seattop_offset + beam_offset, 0, leg_h + top_t, support_r)
            steps.append(step)

        elif leg_type==2:
            # sample square support
            support_r = 3
            # data, step = draw_square_support(data, leg_start, support_r, leg_h + top_t)
            data, step = draw_square_support(data, leg_start, seattop_offset + beam_offset, 0, leg_h + top_t, support_r)
            steps.append(step)

        elif leg_type==3:
            h_bar_t = np.random.randint(1, 3)
            data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w + seattop_offset, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
            data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w + seattop_offset, +s2, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
            data, step = draw_horizontal_bar(data, leg_start, -s1 - leg_w + seattop_offset, -s2 - leg_l, h_bar_t, 2 * s1 + leg_w, leg_l)
            steps.append(step)
            data, step = draw_horizontal_bar(data, leg_start, -s1 - leg_w + seattop_offset, s2, h_bar_t, 2 * s1 + leg_w, leg_l)
            steps.append(step)
            data, step = draw_horizontal_bar(data, leg_start, s1 + seattop_offset, -s2 - leg_l, h_bar_t, leg_w, 2 * (s2 + leg_l))
            steps.append(step)

        elif leg_type==4:
            h_bar_t = np.random.randint(1, 3)
            data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w + seattop_offset, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
            data, step = draw_vertical_leg(data, leg_start, +s1  + seattop_offset, -s2 - leg_l, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
            data, step = draw_vertical_leg(data, leg_start, +s1  + seattop_offset, +s2, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
            data, step = draw_vertical_leg(data, leg_start, -s1 - leg_w + seattop_offset, +s2, leg_h + top_t, leg_w, leg_l)
            steps.append(step)
            data, step = draw_horizontal_bar(data, leg_start, -s1 - leg_w + seattop_offset, -s2 - leg_l, h_bar_t, 2 * (s1 + leg_w), leg_l)
            steps.append(step)
            data, step = draw_horizontal_bar(data, leg_start, -s1 - leg_w + seattop_offset, s2, h_bar_t, 2 * (s1 + leg_w), leg_l)
            steps.append(step)

        elif leg_type==5:
            data, step = draw_vertboard(data, leg_start, -top_r1 +seattop_offset, -top_r2, leg_h + top_t, leg_w, top_r2*2)
            steps.append(step)
            data, step = draw_square_base(data, leg_start, seattop_offset, 0, np.random.choice([1,2], 1)[0], min(top_r1, top_r2))
            steps.append(step)

        elif leg_type==6:
            p = np.random.rand()
            if p < 0.75:
                board_r2 = np.random.randint(1, 3)
            else:
                board_r2 = 3
            s2 = top_r2 - board_r2
            data, step = draw_sideboard(data, leg_start, seattop_offset, -s2-board_r2, leg_h + top_t, top_r1, board_r2)
            steps.append(step)
            data, step = draw_sideboard(data, leg_start, seattop_offset, s2, leg_h + top_t, top_r1, board_r2)
            steps.append(step)

        if leg_type in [1, 2]:
            base_type = np.random.choice([3], 1)[0]
            base_r = np.random.randint(support_r+2, max(support_r+4, top_r2+1))
            p = np.random.rand()
            if p < 0.75:
                base_h = 1
            else:
                base_h = 2
            if base_type==0:
                pass
            elif base_type==1:
                data, step = draw_circle_base(data, leg_start, seattop_offset + beam_offset, 0, base_h, base_r)
                steps.append(step)
            elif base_type==2:
                data, step = draw_square_base(data, leg_start, seattop_offset + beam_offset, 0, base_h, base_r)
                steps.append(step)
            elif base_type==3:
                # sample cross base
                base_r = np.random.randint(7, 12)
                count = np.random.choice([3, 4, 4, 5, 5, 5, 5, 6])
                leg_thickness = 1
                leg_end_offset = -np.random.choice([0, 1, 2, 3, 4], 1)[0]
                current_angle = 90 + np.random.randint(0, int(360/count/2.0), 1)[0]
                x1 = leg_start-leg_thickness
                y1 = seattop_offset + beam_offset
                z1 = 0
                x2 = x1 + leg_end_offset
                y2 = np.rint(-base_r * np.sin(np.deg2rad(current_angle))) + y1
                z2 = np.rint(base_r * np.cos(np.deg2rad(current_angle))) + z1
                data, step = draw_new_base(data, x1, y1, z1, x2, y2, z2, count)
                for step_i in step:
                    steps.append(step_i)
                # count = np.random.choice([3, 3, 4, 4, 5])
                # leg_thickness = 1
                # leg_end_offset = -np.random.choice([1, 2], 1)[0] - leg_thickness
                # draw_new_base(data, steps, leg_start-leg_thickness, seattop_offset + beam_offset , 0, leg_start+leg_end_offset, base_r, leg_thickness, count)

                # for angle in range(4):
                #     data, step = draw_cross_base(data, leg_start, 0, 0, base_h, base_r, angle)
                #     steps.append(step)


        if top_type == 0:
            s1 = top_r1
            s2 = top_r2
        elif top_type == 1:
            s1 = top_r
            s2 = top_r
        elif top_type == 2:
            s1 = int(round(top_r / math.sqrt(2)))
            s2 = int(round(top_r / math.sqrt(2)))

        if back_type==0:
            back_offset_1 = +s1 + np.random.choice([-1, 0], 1)[0] + seattop_offset
            back_offset_2 = back_offset_1
            back_h_1 = 1
            back_width_half = s2
            data, step = draw_tilt_back(data, leg_start + leg_h + top_t, back_offset_1, -s2, back_height, back_thickness, 2* back_width_half, back_tilt_amount)
            steps.append(step)

        elif back_type==1:
            back_h_1 = np.random.randint(3,5)
            back_h_2 = back_height-back_h_1
            scale_factor_1 = np.random.rand() * 0.5
            scale_factor_2 = np.random.rand()*0.3 + 0.7
            if top_type==2:
                scale_factor_2 = scale_factor_2 * 1.3

            back_offset_1 = +s1 + np.random.choice([0, 1], 1)[0] + seattop_offset - back_thickness
            back_offset_2 = +s1 + np.random.choice([0, 1], 1)[0] + seattop_offset - back_thickness

            back_support_width = max(2 * int(s2 * scale_factor_1),2)
            back_width_half = int(s2 * scale_factor_2)

            data, step = draw_back_support(data, leg_start + leg_h + top_t, back_offset_1, -max(int(s2 * scale_factor_1), 1), back_h_1, np.random.choice([1, 2], 1)[0], back_support_width)
            steps.append(step)
            data, step = draw_tilt_back(data, leg_start + leg_h + top_t + back_h_1, back_offset_2, -int(s2*scale_factor_2), back_h_2, back_thickness, 2*back_width_half, back_tilt_amount)
            steps.append(step)

        arm_type = np.random.choice([0, 1, 2, 3, 3, 4, 4], 1)[0]
        # arm_type = 2
        # arm_type = 4
        # from pprint import pprint
        # pprint(steps)

        if arm_type == 0:
            pass

        elif arm_type == 1 or arm_type == 2 or arm_type == 3 or arm_type == 4:

            arm_scale_factor = np.random.rand()
            side_length = int(s1 * max(0.7, arm_scale_factor))
            side_offset = back_offset_1 - side_length + np.random.choice([-2, -1, 0, 1], 1)[0]
            side_height = back_h_1+np.random.randint(1, 3,1)[0]

            p = np.random.rand()
            if p < 0.45:
                arm_beam_w = 1
                arm_beam_l = 1
            elif p < 0.725:
                arm_beam_w = 1
                arm_beam_l = 2
            else:
                arm_beam_w = 2
                arm_beam_l = 1

            shrink_l = np.random.choice([-1, 0], 1)[0]
            s2 = back_width_half - shrink_l - arm_beam_l

            if arm_type == 1:

                data, step = draw_sideboard(data, leg_start + leg_h + top_t, side_offset, -s2 - arm_beam_l, side_height, side_length, arm_beam_l)
                steps.append(step)
                data, step = draw_sideboard(data, leg_start + leg_h + top_t, side_offset, s2, side_height, side_length, arm_beam_l)
                steps.append(step)

            elif arm_type == 2:

                front_back_loc = int(side_length/2)
                p = np.random.rand()
                if p < 0.45:
                    arm_beam_w = 1
                    arm_beam_l = 1
                elif p < 0.725:
                    arm_beam_w = 1
                    arm_beam_l = 2
                else:
                    arm_beam_w = 2
                    arm_beam_l = 1

                data, step = draw_chair_beam(data, leg_start + leg_h + top_t, -front_back_loc - arm_beam_w + seattop_offset, -s2 - arm_beam_l, side_height, arm_beam_w, arm_beam_l)
                steps.append(step)
                data, step = draw_chair_beam(data, leg_start + leg_h + top_t, +front_back_loc + seattop_offset, -s2 - arm_beam_l, side_height, arm_beam_w, arm_beam_l)
                steps.append(step)
                data, step = draw_chair_beam(data, leg_start + leg_h + top_t, +front_back_loc + seattop_offset, +s2, side_height, arm_beam_w, arm_beam_l)
                steps.append(step)
                data, step = draw_chair_beam(data, leg_start + leg_h + top_t, -front_back_loc - arm_beam_w + seattop_offset, +s2, side_height, arm_beam_w, arm_beam_l)
                steps.append(step)

                h_bar_t = np.random.choice([1,2], 1)[0]
                data, step = draw_horizontal_bar(data, leg_start + leg_h + top_t + side_height, -front_back_loc - arm_beam_w + seattop_offset, -s2 - arm_beam_l, h_bar_t, 2 * front_back_loc + 2 * arm_beam_w, arm_beam_l)
                steps.append(step)
                data, step = draw_horizontal_bar(data, leg_start + leg_h + top_t + side_height, -front_back_loc - arm_beam_w + seattop_offset, +s2, h_bar_t, 2 * front_back_loc + 2 * arm_beam_w, arm_beam_l)
                steps.append(step)

            elif arm_type == 3 or arm_type == 4:
                front_back_loc = np.random.randint(arm_beam_w, top_r1) - arm_beam_w
                # side_length = top_r1 + front_back_loc
                side_length = back_offset_2 + front_back_loc + arm_beam_w - seattop_offset
                if arm_type == 3:
                    data, step = draw_chair_beam(data, leg_start + leg_h + top_t, -front_back_loc - arm_beam_w + seattop_offset, -s2 - arm_beam_l, side_height, arm_beam_w, arm_beam_l)
                    steps.append(step)
                    data, step = draw_chair_beam(data, leg_start + leg_h + top_t, -front_back_loc - arm_beam_w + seattop_offset, +s2, side_height, arm_beam_w, arm_beam_l)
                    steps.append(step)

                h_bar_t = np.random.choice([1,2], 1)[0]
                data, step = draw_horizontal_bar(data, leg_start + leg_h + top_t + side_height, -front_back_loc - arm_beam_w + seattop_offset, -s2 - arm_beam_l, h_bar_t, side_length + arm_beam_w, arm_beam_l)
                steps.append(step)
                data, step = draw_horizontal_bar(data, leg_start + leg_h + top_t + side_height, -front_back_loc - arm_beam_w + seattop_offset, +s2, h_bar_t, side_length + arm_beam_w, arm_beam_l)
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