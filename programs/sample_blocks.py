from __future__ import print_function
from __future__ import division

from .utils import *
from .label_config import max_param, for_step
from .loop_gen import gen_loop
from .complex_base import draw_new_base


def sample_batch(num, primitive_type):
    data = np.zeros((num, 32, 32, 32), dtype=np.uint8)
    label = np.zeros((num, for_step, max_param), dtype=np.int32)
    for i in range(num):
        if primitive_type == 1:
            d, s = sample_vertical_leg()
        elif primitive_type == 2:
            d, s = sample_rectangle_tabletop()
        elif primitive_type == 3:
            d, s = sample_square_tabletop()
        elif primitive_type == 4:
            d, s = sample_circle_tabletop()
        elif primitive_type == 5:
            d, s = sample_second_layer()
        elif primitive_type == 6:
            d, s = sample_circle_support()
        elif primitive_type == 7:
            d, s = sample_square_support()
        elif primitive_type == 8:
            d, s = sample_circle_base()
        elif primitive_type == 9:
            d, s = sample_square_base()
        elif primitive_type == 10:
            d, s = sample_cross_base()
        elif primitive_type == 11:
            d, s = sample_side_board()
        elif primitive_type == 12:
            d, s = sample_horizontal_bar()
        elif primitive_type == 13:
            d, s = sample_vertboard()
        elif primitive_type == 14:
            d, s = sample_locker()

        elif primitive_type == 15:
            d, s = sample_vertical_leg_2()
        elif primitive_type == 16:
            d, s = sample_circle_support_2()
        elif primitive_type == 17:
            d, s = sample_square_support_2()
        elif primitive_type == 18:
            d, s = sample_circle_base_2()
        elif primitive_type == 19:
            d, s = sample_square_base_2()
        elif primitive_type == 20:
            d, s = sample_side_board_2()
        elif primitive_type == 21:
            d, s = sample_horizontal_bar_2()
        elif primitive_type == 22:
            d, s = sample_vertboard_2()
        elif primitive_type == 23:
            d, s = sample_rectangle_seattop()
        elif primitive_type == 24:
            d, s = sample_square_seattop()
        elif primitive_type == 25:
            d, s = sample_circle_seattop()
        elif primitive_type == 26:
            d, s = sample_tilt_back()
        elif primitive_type == 27:
            d, s = sample_chair_beam()
        elif primitive_type == 28:
            d, s = sample_new_line()
        elif primitive_type == 29:
            d, s = sample_back_support()

        elif primitive_type == 101:   # 30
            d, s = sample_leg_loop_double()
        elif primitive_type == 102:   # 31
            d, s = sample_leg_loop_single()
        elif primitive_type == 103:   # 32
            d, s = sample_horizontal_bar_loop_single()
        elif primitive_type == 104:   # 33
            d, s = sample_sideboard_loop_single()
        elif primitive_type == 105:   # 34
            d, s = sample_locker_loop_single()
        elif primitive_type == 106:   # 35
            d, s = sample_cross_base_loop_single()
        elif primitive_type == 107:   # 36
            d, s = sample_line_loop_single()
        elif primitive_type == 108:   # 37
            d, s = sample_second_layer_loop_single()

        elif primitive_type == 109:   # 38
            d, s = sample_leg_loop_double_2()
        elif primitive_type == 110:   # 39
            d, s = sample_leg_loop_single_2()
        elif primitive_type == 111:   # 40
            d, s = sample_horizontal_bar_loop_single_2()
        elif primitive_type == 112:   # 41
            d, s = sample_sideboard_loop_single_2()
        elif primitive_type == 113:   # 42
            d, s = sample_chair_beam_double()
        elif primitive_type == 114:   # 43
            d, s = sample_chair_beam_single()
        elif primitive_type == 115:   # 44
            d, s = sample_new_base_single()

        elif primitive_type == 0:   # 45
            return data, label
        else:
            raise NotImplementedError('type is not implemented: ', primitive_type)
        data[i] = d

        if len(s) == 1:
            n_step = 1
        elif len(s) == 3:
            n_step = 2
        elif len(s) == 5:
            n_step = 3
        else:
            raise NotImplementedError('length of steps is not implemented: ', len(s))

        label[i, :n_step, :s.shape[1]] = s[:n_step]
    return data, label


# ======================
# ===== none-loops =====
# ======================


# ======1: vertical leg======
def sample_vertical_leg():
    # sample the shape
    leg_w = np.random.randint(1, 3)
    leg_l = np.random.randint(1, 3)
    p = np.random.rand()
    if p < 0.2:
        leg_w += 1
        leg_l += 1
    leg_h = np.random.randint(8, 25)
    leg_start = - int(leg_h / 2)
    q = np.random.rand()
    if q < 0.5:
        leg_h -= 1

    # sample the position
    leg_s1 = np.random.randint(-12, 11)
    leg_s2 = np.random.randint(-12, 11)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_vertical_leg(data, leg_start, leg_s1, leg_s2, leg_h, leg_w, leg_l)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======2: rectangle tabletop======
def sample_rectangle_tabletop():
    top_r1 = np.random.randint(4, 13)
    top_r2 = np.random.randint(8, 13)

    p = np.random.rand()
    if p < 0.6:
        r1 = min(top_r1, top_r2)
        r2 = max(top_r1, top_r2)
    elif p < 0.8:
        r1 = max(top_r1, top_r2)
        r2 = min(top_r1, top_r2)
    else:
        r1 = np.random.randint(10, 13)
        r2 = np.random.randint(4, 10)

    top_t = np.random.randint(1, 5)

    h = np.random.randint(4, 13) - top_t
    c1 = np.random.randint(-2, 3)
    c2 = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_rectangle_top(data, h, c1, c2, top_t, r1, r2)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======3: square tabletop======
def sample_square_tabletop():
    top_r = np.random.randint(6, 13)
    top_t = np.random.randint(1, 5)

    h = np.random.randint(4, 13) - top_t
    c1 = np.random.randint(-2, 3)
    c2 = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_square_top(data, h, c1, c2, top_t, top_r)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======4: circle tabletop======
def sample_circle_tabletop():
    top_r = np.random.randint(6, 13)
    top_t = np.random.randint(1, 5)

    h = np.random.randint(4, 13) - top_t
    c1 = np.random.randint(-2, 3)
    c2 = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_circle_top(data, h, c1, c2, top_t, top_r)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======5: second layer======
def sample_second_layer():
    top_r1 = np.random.randint(2, 13)
    top_r2 = np.random.randint(8, 13)

    p = np.random.rand()
    if p < 0.6:
        r1 = min(top_r1, top_r2)
        r2 = max(top_r1, top_r2)
    elif p < 0.8:
        r1 = max(top_r1, top_r2)
        r2 = min(top_r1, top_r2)
    else:
        r1 = np.random.randint(10, 13)
        r2 = np.random.randint(4, 10)

    p = np.random.rand()
    if p < 0.7:
        t = np.random.randint(1, 4)
    else:
        t = np.random.randint(4, 8)

    h = np.random.randint(-12, 10) - t

    c1 = np.random.randint(-2, 3)
    c2 = 0

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_middle_rect_layer(data, h, c1, c2, t, r1, r2)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======6: circle support======
def sample_circle_support():
    r = np.random.randint(1, 9)
    t = np.random.randint(6, 25)
    h = - int(t / 2)
    p = 2 * np.random.rand()
    t = t - round(p)

    c1 = np.random.randint(-2, 3)
    c2 = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_circle_support(data, h, c1, c2, t, r)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======7: square support======
def sample_square_support():
    r = np.random.randint(3, 9)
    t = np.random.randint(6, 25)
    h = - int(t/2)
    p = 2 * np.random.rand()
    t = t - round(p)

    c1 = np.random.randint(-2, 3)
    c2 = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_square_support(data, h, c1, c2, t, r)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======8: circle base======
def sample_circle_base():
    r = np.random.randint(4, 11)
    t = np.random.randint(1, 3)

    h = np.random.randint(-13, -3)
    c1 = np.random.randint(-2, 3)
    c2 = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_circle_base(data, h, c1, c2, t, r)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======9: square base======
def sample_square_base():
    r = np.random.randint(4, 11)
    t = np.random.randint(1, 3)

    h = np.random.randint(-12, -3)
    c1 = np.random.randint(-2, 3)
    c2 = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_square_base(data, h, c1, c2, t, r)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======10: cross base======
def sample_cross_base():
    r = np.random.randint(4, 13)
    t = np.random.randint(1, 3)

    h = np.random.randint(-12, -3)
    angle = np.random.randint(0, 4)
    c1 = np.random.randint(-1, 2)
    c2 = np.random.randint(-1, 2)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_cross_base(data, h, c1, c2, t, r, angle)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======11: side board======
def sample_side_board():
    r1 = np.random.randint(3, 13)
    r2 = np.random.randint(1, 4)
    t = np.random.randint(8, 25)

    s1 = np.random.randint(-2, 3)
    s2 = np.random.randint(-12, 11)
    h = - int(t/2)
    h = h + np.random.randint(-2, 3)
    delta = np.random.randint(0, 4)
    t = t - delta

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_sideboard(data, h, s1, s2, t, r1, r2)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======12: horizontal bar======
def sample_horizontal_bar():
    h_start = np.random.randint(-12, 8)
    t = np.random.randint(1, 3)

    length = np.random.randint(6, 25)
    width = np.random.randint(1, 4)

    shift = np.random.randint(-2, 3)

    p = np.random.rand()
    if p < 0.5:
        s1 = - int(length / 2) + shift
        s2 = np.random.randint(-12, 12)
        r1 = length
        r2 = width
    else:
        s1 = np.random.randint(-12, 12)
        s2 = - int(length / 2) + shift
        r1 = width
        r2 = length

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_horizontal_bar(data, h_start, s1, s2, t, r1, r2)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======13: vertboard======
def sample_vertboard():

    p = np.random.rand()
    if p < 0.5:
        h = np.random.randint(4, 13)
        h_start = - h + np.random.randint(0, max(5, round(1.5*h)))
        t = h - h_start + np.random.randint(-3, 1)
        t = max(t, 2)

        r1 = np.random.randint(1, 4)
        r2 = 2 * np.random.randint(5, 13)
        s1 = np.random.randint(-12, 13)
        s2 = -int(r2 / 2)

        shift = np.random.randint(-2, 3)

        s2 = s2 + shift

        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        data, step = draw_vertboard(data, h_start, s1, s2, t, r1, r2)
        steps.append(step)

        steps = np.asarray(steps)

    else:
        # sample the shape
        t = np.random.randint(4, 25)
        r1 = np.random.randint(1, 4)
        r2 = 2 * np.random.randint(5, 13)

        # sample the position
        s1 = np.random.randint(-12, 12 - r1 + 1)
        s2 = np.random.randint(-12, 12 - r2 + 1)
        h = np.random.randint(-12, 12 - t + 1)

        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        data, step = draw_vertboard(data, h, s1, s2, t, r1, r2)
        steps.append(step)

        steps = np.asarray(steps)

    return data, steps


# ======14: sample locker======
def sample_locker():
    t = np.random.randint(8, 25)
    shift = np.random.randint(0, 1 + 24 - t)
    h_start = - 12 + shift

    r1 = np.random.randint(4, 25)
    r2 = np.random.randint(4, 20)
    s1 = np.random.randint(-12, 13 - r1)
    s2 = np.random.randint(-12, 13 - r2)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_locker(data, h_start, s1, s2, t, r1, r2)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======15: vertical leg======
def sample_vertical_leg_2():
    # sample the shape
    leg_w = np.random.randint(1, 3)
    leg_l = np.random.randint(1, 3)
    p = np.random.rand()
    if p < 0.2:
        leg_w += 1
        leg_l += 1
    leg_h = np.random.randint(3, 16)
    p = np.random.rand()
    if p < 0.3:
        leg_start = -12
    else:
        leg_start = np.random.randint(-12, -3)

    # sample the position
    leg_s1 = np.random.randint(-12, 11)
    leg_s2 = np.random.randint(-12, 11)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_vertical_leg(data, leg_start, leg_s1, leg_s2, leg_h, leg_w, leg_l)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======16: circle support======
def sample_circle_support_2():
    r = np.random.randint(1, 5)
    t = np.random.randint(4, 14)

    p = np.random.rand()
    if p < 0.5:
        h = -12 + np.random.choice([0, 1, 2, 3, 4], 1)[0]
    else:
        h = -t + np.random.choice([0, 1, 2, 3, 4], 1)[0]

    c1 = np.random.randint(-4, 3)
    c2 = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_circle_support(data, h, c1, c2, t, r)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======17: square support======
def sample_square_support_2():
    r = np.random.randint(3, 11)
    t = np.random.randint(4, 14)
    p = np.random.rand()
    if p < 0.5:
        h = -12 + np.random.choice([0, 1, 2, 3, 4], 1)[0]
    else:
        h = -t + np.random.choice([0, 1, 2, 3, 4], 1)[0]

    c1 = np.random.randint(-4, 3)
    c2 = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_square_support(data, h, c1, c2, t, r)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======18: circle base======
def sample_circle_base_2():
    r = np.random.randint(4, 11)
    t = np.random.randint(1, 3)

    h = np.random.randint(-13, -3)
    c1 = np.random.randint(-4, 3)
    c2 = np.random.randint(-4, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_circle_base(data, h, c1, c2, t, r)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======19: square base======
def sample_square_base_2():
    r = np.random.randint(4, 11)
    t = np.random.randint(1, 3)

    h = np.random.randint(-13, -3)
    c1 = np.random.randint(-4, 3)
    c2 = np.random.randint(-4, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_square_base(data, h, c1, c2, t, r)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======20: side board======
def sample_side_board_2():

    p = np.random.rand()

    if p < 0.5:
        r1 = np.random.randint(3, 13)
        r2 = np.random.randint(1, 4)
        t = np.random.randint(5, 13)

        s1 = np.random.randint(-4, 3)
        s2 = np.random.randint(-12, 11)
        h = -t + np.random.randint(-2, 7)

        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        data, step = draw_sideboard(data, h, s1, s2, t, r1, r2)
        steps.append(step)

        steps = np.asarray(steps)

        return data, steps
    else:
        r1 = int(np.random.randint(5, 13) * max(np.random.rand(), 0.6))
        r2 = np.random.randint(1, 4)
        t = np.random.randint(4, 7)

        s1 = np.random.randint(-4, 3)
        s2 = np.random.randint(-7, 6)
        h = np.random.choice([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4], 1)[0]

        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        data, step = draw_sideboard(data, h, s1, s2, t, r1, r2)
        steps.append(step)

        steps = np.asarray(steps)

        return data, steps


# ======21: horizontal bar======
def sample_horizontal_bar_2():
    h_start = np.random.randint(-13, 8)
    t = np.random.randint(1, 3)

    length = np.random.randint(8, 25)
    width = np.random.randint(1, 4)

    shift = np.random.randint(-3, 4)

    p = np.random.rand()
    if p < 0.5:
        s1 = - int(length / 2) + shift
        s2 = np.random.randint(-12, 12)
        r1 = length
        r2 = width
    else:
        s1 = np.random.randint(-12, 12)
        s2 = - int(length / 2) + shift
        r1 = width
        r2 = length

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_horizontal_bar(data, h_start, s1, s2, t, r1, r2)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======22: vertboard======
def sample_vertboard_2():

    q = np.random.rand()
    if q < 0.6:
        h = np.random.randint(4, 13)
        p = np.random.rand()
        if p < 0.5:
            h_start = - h + np.random.choice([0, 1, 2, 3, 4], 1)[0]
        else:
            h_start = -12 + np.random.randint(0, 8)

        r1 = np.random.randint(1, 4)
        r2 = 2 * np.random.randint(5, 13)
        s1 = np.random.randint(-12, 8)
        s2 = -int(r2 / 2)

        shift = 0
        s2 = s2 + shift
        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        data, step = draw_vertboard(data, h_start, s1, s2, h, r1, r2)
        steps.append(step)
        steps = np.asarray(steps)

    else:
        h = np.random.randint(8, 21)
        r1 = np.random.randint(1, 4)
        r2 = 2 * np.random.randint(4, 10)

        h_start = np.random.randint(-12, 12 - h + 1)
        s1 = np.random.randint(-12, -7)
        s2 = -int(r2 / 2)

        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []

        data, step = draw_vertboard(data, h_start, s1, s2, h, r1, r2)
        steps.append(step)
        data, step = draw_vertboard(data, h_start, -s1 - r1, s2, h, r1, r2)
        steps.append(step)

        steps = gen_loop(steps)
        steps = np.asarray(steps)

    return data, steps


# ======23: rectangle_seattop======
def sample_rectangle_seattop():

    top_r1 = np.random.randint(3, 13)
    top_r2 = np.random.randint(8, 13)
    p = np.random.rand()
    if p < 0.6:
        r1 = min(top_r1, top_r2)
        r2 = max(top_r1, top_r2)
    else:
        r1 = max(top_r1, top_r2)
        r2 = min(top_r1, top_r2)
    top_r1 = r1
    top_r2 = r2

    top_t = np.random.randint(1, 5)
    h = np.random.randint(-5, 6) - top_t
    c1 = np.random.randint(-4, 4)
    c2 = np.random.randint(-1, 2)
    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_rectangle_top(data, h, c1, c2, top_t, top_r1, top_r2)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======24: square_seattop======
def sample_square_seattop():
    top_r = np.random.randint(5, 13)
    top_t = np.random.randint(1, 4)
    h = np.random.randint(-5, 6) - top_t
    c1 = np.random.randint(-4, 4)
    c2 = np.random.randint(-1, 2)
    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_square_top(data, h, c1, c2, top_t, top_r)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======25: circle_seattop======
def sample_circle_seattop():
    top_r = np.random.randint(5, 13)
    top_t = np.random.randint(1, 4)
    h = np.random.randint(-5, 6) - top_t
    c1 = np.random.randint(-4, 4)
    c2 = np.random.randint(-1, 2)
    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_circle_top(data, h, c1, c2, top_t, top_r)
    steps.append(step)

    steps = np.asarray(steps)

    return data, steps


# ======26: tilt_back======
def sample_tilt_back():

    q = np.random.rand()

    if q < 0.6:
        h = np.random.randint(-7, 5)
        p = np.random.rand()
        if p < 0.5:
            back_height = 13 - h - np.random.choice([0, 0, 1, 1, 2])
        else:
            back_height = np.random.randint(2, min(12 - h, 8))

        back_thickness = np.random.randint(1, 5)

        p = np.random.rand()
        if p < 0.3:
            tilt_amount = 0
        else:
            tilt_amount = np.random.randint(1, 5)

        offset = tilt_amount + back_thickness
        c1 = np.random.randint(4, 15) - offset
        width = np.random.randint(4, 13)
        c2 = - width
        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        data, step = draw_tilt_back(data, h, c1, c2, back_height, back_thickness, 2 * width, tilt_amount)
        steps.append(step)
        steps = np.asarray(steps)

    else:

        h = np.random.randint(-4, 5)
        back_height = np.random.randint(2, 8)
        back_thickness = np.random.randint(1, 4)

        p = np.random.rand()
        if p < 0.5:
            tilt_amount = 0
        else:
            tilt_amount = np.random.randint(1, 4)

        offset = tilt_amount + back_thickness
        c1 = np.random.randint(3, 10) - offset
        p = np.random.rand()
        if p < 0.:
            width = np.random.randint(10, 13)
        else:
            width = np.random.randint(6, 13)
        c2 = - width
        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        data, step = draw_tilt_back(data, h, c1, c2, back_height, back_thickness, 2 * width, tilt_amount)
        steps.append(step)
        steps = np.asarray(steps)

    return data, steps


# ======27: chair_beam======
def sample_chair_beam():
    # sample the shape
    leg_w = np.random.randint(1, 3)
    leg_l = np.random.randint(1, 3)
    p = np.random.rand()
    if p < 0.2:
        leg_w += 1
        leg_l += 1
    beam_h = np.random.randint(2, 5) + np.random.randint(0, 4)
    beam_start = np.random.randint(-5, 5)

    s1 = np.random.randint(-10, 11)
    s2 = np.random.randint(-12, 11)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_chair_beam(data, beam_start, s1, s2, beam_h, leg_w, leg_l)
    steps.append(step)
    steps = np.asarray(steps)
    return data, steps


# ======28: new_line======
def sample_new_line():
    base_r = np.random.randint(5, 12)
    leg_end_offset = -np.random.choice([0, 1, 2], 1)[0]
    current_angle = np.random.randint(0, 350, 1)[0]

    leg_h = np.random.randint(6, 14)
    leg_start = -leg_h + np.random.choice([0, 1, 2, 3, 4], 1)[0]

    origin_1 = np.random.randint(-3, 4, 1)[0]
    origin_2 = np.random.randint(-3, 4, 1)[0]

    leg_end = leg_start + leg_end_offset

    y2 = int(np.rint(-base_r * np.sin(np.deg2rad(current_angle)))) + origin_1
    z2 = int(np.rint(base_r * np.cos(np.deg2rad(current_angle)))) + origin_2
    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_line(data, leg_start, origin_1, origin_2, leg_end, y2, z2, 1)
    steps.append(step)
    steps = np.asarray(steps)
    return data, steps


# ======29: back_support======
def sample_back_support():
    # sample the shape
    support_h = np.random.randint(3, 6, 1)[0]
    support_start = np.random.choice([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3], 1)[0]

    s1 = np.random.randint(-3, 11)
    s2 = np.random.randint(-5, 0)
    length = -s2 * 2

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    support_w = np.random.choice([1,2,3])
    data, step = draw_back_support(data, support_start, s1, s2, support_h, support_w, length)
    steps.append(step)
    steps = np.asarray(steps)
    return data, steps


# ======================
# ===== loops =====
# ======================

# ======101======
def sample_leg_loop_double():

    leg_h = np.random.randint(8, 25)
    leg_start = - int(leg_h / 2)
    q = 2 * np.random.rand()
    leg_h = leg_h - round(q)

    l1 = np.random.randint(1, 3)
    l2 = np.random.randint(1, 3)
    t = np.random.rand()
    if t < 0.2:
        l1 += 1
        l2 += 1

    x_min = np.random.randint(-12, -4)
    y_min = np.random.randint(-12, -4)
    x_max = - x_min - l1
    y_max = - y_min - l2
    s_x = np.random.randint(-4, 4)
    s_y = 0

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_vertical_leg(data, leg_start, x_min + s_x, y_min + s_y, leg_h, l1, l2)
    steps.append(step)
    data, step = draw_vertical_leg(data, leg_start, x_max + s_x, y_min + s_y, leg_h, l1, l2)
    steps.append(step)
    data, step = draw_vertical_leg(data, leg_start, x_max + s_x, y_max + s_y, leg_h, l1, l2)
    steps.append(step)
    data, step = draw_vertical_leg(data, leg_start, x_min + s_x, y_max + s_y, leg_h, l1, l2)
    steps.append(step)

    steps = gen_loop(steps)
    steps = np.asarray(steps)

    return data, steps


# ======102======
def sample_leg_loop_single():

    leg_h = np.random.randint(8, 26)
    leg_start = - int(leg_h / 2)
    q = 2 * np.random.rand()
    leg_h = leg_h - round(q)

    l1 = np.random.randint(1, 3)
    l2 = np.random.randint(1, 3)
    t = np.random.rand()
    if t < 0.2:
        l1 += 1
        l2 += 1

    p = np.random.rand()
    if p < 0.5:
        x_min = np.random.randint(-2, 3)
        y_min = np.random.randint(-12, -4)
        y_max = - y_min - l2
        s_x = np.random.randint(-3, 4)
        s_y = 0

        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        data, step = draw_vertical_leg(data, leg_start, x_min + s_x, y_min + s_y, leg_h, l1, l2)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, x_min + s_x, y_max + s_y, leg_h, l1, l2)
        steps.append(step)
        steps = gen_loop(steps)
        steps = np.asarray(steps)

    else:
        x_min = np.random.randint(-12, -4)
        y_min = np.random.randint(-2, 3)
        x_max = - x_min - l1
        s_x = 0
        s_y = np.random.randint(-3, 4)

        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        data, step = draw_vertical_leg(data, leg_start, x_min + s_x, y_min + s_y, leg_h, l1, l2)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, x_max + s_x, y_min + s_y, leg_h, l1, l2)
        steps.append(step)
        steps = gen_loop(steps)
        steps = np.asarray(steps)

    return data, steps


# ======103======
def sample_horizontal_bar_loop_single():

    h_start = np.random.randint(-12, 8)
    h_t = np.random.randint(1, 3)

    l1 = np.random.randint(1, 3)
    l2 = np.random.randint(1, 3)
    x_min = np.random.randint(-12, -3)
    y_min = np.random.randint(-12, -3)
    x_max = - x_min - l1
    y_max = - y_min - l2
    residual = (x_min + 12)
    residual = min(4, residual)
    s_x = np.random.randint(-residual, residual + 1)
    s_y = 0
    x_min = x_min + s_x
    x_max = x_max + s_x
    y_min = y_min + s_y
    y_max = y_max + s_y

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []

    q = np.random.rand()
    if q < 0.5:
        data, step = draw_horizontal_bar(data, h_start, x_min, y_min, h_t, l1, y_max - y_min + l2)
        steps.append(step)
        data, step = draw_horizontal_bar(data, h_start, x_max, y_min, h_t, l1, y_max - y_min + l2)
        steps.append(step)
    else:
        data, step = draw_horizontal_bar(data, h_start, x_min, y_min, h_t, x_max - x_min + l1, l2)
        steps.append(step)
        data, step = draw_horizontal_bar(data, h_start, x_min, y_max, h_t, x_max - x_min + l1, l2)
        steps.append(step)

    steps = gen_loop(steps)
    steps = np.asarray(steps)

    return data, steps


# ======104======
def sample_sideboard_loop_single():
    r1 = np.random.randint(2, 13)
    r2 = np.random.randint(1, 4)
    t = np.random.randint(8, 25)

    s1 = np.random.randint(-2, 3)
    s2 = np.random.randint(-12, -3)
    h = - int(t / 2)
    p = 2 * np.random.rand()
    t = t - round(p)

    shift = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_sideboard(data, h, s1, s2 + shift, t, r1, r2)
    steps.append(step)
    data, step = draw_sideboard(data, h, s1, -s2-r2 + shift, t, r1, r2)
    steps.append(step)

    steps = gen_loop(steps)
    steps = np.asarray(steps)

    return data, steps


# ======105======
def sample_locker_loop_single():
    t = np.random.randint(8, 25)
    shift = np.random.randint(0, 1 + 24 - t)
    h_start = - 12 + shift

    r1 = np.random.randint(5, 15)
    r2 = np.random.randint(5, 11)
    s1 = np.random.randint(-12, 13 - r1)
    s2 = np.random.randint(-12, -r2)

    shift = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_locker(data, h_start, s1, s2+shift, t, r1, r2)
    steps.append(step)
    data, step = draw_locker(data, h_start, s1, -s2-r2+shift, t, r1, r2)
    steps.append(step)

    steps = gen_loop(steps)
    steps = np.asarray(steps)

    return data, steps


# ======106======
def sample_cross_base_loop_single():
    r = np.random.randint(4, 13)
    t = np.random.randint(1, 3)

    h = np.random.randint(-12, -3)
    c1 = np.random.randint(-2, 3)
    c2 = 0

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    for i in range(4):
        data, step = draw_cross_base(data, h, c1, c2, t, r, angle=i)
        steps.append(step)

    steps = gen_loop(steps)
    steps = np.asarray(steps)

    return data, steps


# ======107======
def sample_line_loop_single():

    r = np.random.randint(4, 10)
    h = np.random.randint(-12, -5)

    c1 = np.random.randint(-3, 1)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    h_ = h + np.random.randint(-3, 1)
    data, step = draw_new_base(data, h, c1, 0, h_, -r + c1, -r, 4)
    steps = step

    steps = gen_loop(steps)
    steps = np.asarray(steps)

    return data, steps


# ======108======
def sample_second_layer_loop_single():
    top_r1 = np.random.randint(2, 13)
    top_r2 = np.random.randint(8, 13)

    p = np.random.rand()
    if p < 0.6:
        r1 = min(top_r1, top_r2)
        r2 = max(top_r1, top_r2)
    else:
        r1 = max(top_r1, top_r2)
        r2 = min(top_r1, top_r2)

    t = np.random.randint(1, 3)
    h1 = np.random.randint(-3, 8) - t
    h2 = np.random.randint(-13, h1 - t - 1)

    c1 = np.random.randint(-2, 3)
    c2 = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_middle_rect_layer(data, h2, c1, c2, t, r1, r2)
    steps.append(step)
    data, step = draw_middle_rect_layer(data, h1, c1, c2, t, r1, r2)
    steps.append(step)

    steps = gen_loop(steps)
    steps = np.asarray(steps)

    return data, steps


# ======109======
def sample_leg_loop_double_2():

    leg_h = np.random.randint(3, 14)
    p = np.random.rand()
    if p < 0.3:
        leg_start = -12
    else:
        leg_start = np.random.randint(-12, -3)

    l1 = np.random.randint(1, 3)
    l2 = np.random.randint(1, 3)
    t = np.random.rand()
    if t < 0.2:
        l1 += 1
        l2 += 1

    q = np.random.rand()
    if q < 0.6:
        x_min = np.random.randint(-12, -3)
        y_min = np.random.randint(-12, -4)
    else:
        x_min = np.random.randint(-6, -2)
        y_min = np.random.randint(-12, -8)
    x_max = - x_min - l1
    y_max = - y_min - l2
    s_x = np.random.randint(-4, 2)
    s_y = 0

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_vertical_leg(data, leg_start, x_min + s_x, y_min + s_y, leg_h, l1, l2)
    steps.append(step)
    data, step = draw_vertical_leg(data, leg_start, x_max + s_x, y_min + s_y, leg_h, l1, l2)
    steps.append(step)
    data, step = draw_vertical_leg(data, leg_start, x_max + s_x, y_max + s_y, leg_h, l1, l2)
    steps.append(step)
    data, step = draw_vertical_leg(data, leg_start, x_min + s_x, y_max + s_y, leg_h, l1, l2)
    steps.append(step)

    steps = gen_loop(steps)
    steps = np.asarray(steps)

    return data, steps


# ======110======
def sample_leg_loop_single_2():

    leg_h = np.random.randint(3, 14)
    p = np.random.rand()
    if p < 0.3:
        leg_start = -12
    else:
        leg_start = np.random.randint(-12, -3)

    l1 = np.random.randint(1, 3)
    l2 = np.random.randint(1, 3)
    t = np.random.rand()
    if t < 0.2:
        l1 += 1
        l2 += 1

    r = np.random.rand()

    if r < 0.5:
        x_min = np.random.randint(-12, -4)
        y_min = np.random.randint(-2, 0)
        x_max = - x_min - l1
        s_x = np.random.randint(-4, 3)
        s_y = 0

        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        data, step = draw_vertical_leg(data, leg_start, x_min + s_x, y_min + s_y, leg_h, l1, l2)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, x_max + s_x, y_min + s_y, leg_h, l1, l2)
        steps.append(step)
        steps = gen_loop(steps)
        steps = np.asarray(steps)

    else:

        x_min = 0
        y_min = np.random.randint(-12, -5)
        y_max = - y_min - l2
        s_x = np.random.randint(-4, 3)
        s_y = np.random.randint(-1, 2)

        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        data, step = draw_vertical_leg(data, leg_start, x_min + s_x, y_min + s_y, leg_h, l1, l2)
        steps.append(step)
        data, step = draw_vertical_leg(data, leg_start, x_min + s_x, y_max + s_y, leg_h, l1, l2)
        steps.append(step)
        steps = gen_loop(steps)
        steps = np.asarray(steps)

    return data, steps


# ======111======
def sample_horizontal_bar_loop_single_2():

    h_start = np.random.randint(-13, 8)
    h_t = np.random.randint(1, 3)

    l1 = np.random.randint(1, 3)
    l2 = np.random.randint(1, 3)

    r = np.random.rand()
    if r < 0.6:
        x_min = np.random.randint(-12, -3)
        y_min = np.random.randint(-12, -3)
        x_max = - x_min - l1
        y_max = - y_min - l2
        s_x = np.random.randint(-4, 2)
        s_y = 0
        x_min = x_min + s_x
        x_max = x_max + s_x
        y_min = y_min + s_y
        y_max = y_max + s_y

        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        q = np.random.rand()
        if q < 0.5:
            data, step = draw_horizontal_bar(data, h_start, x_min, y_min, h_t, l1, y_max - y_min + l2)
            steps.append(step)
            data, step = draw_horizontal_bar(data, h_start, x_max, y_min, h_t, l1, y_max - y_min + l2)
            steps.append(step)
        else:
            data, step = draw_horizontal_bar(data, h_start, x_min, y_min, h_t, x_max - x_min + l1, l2)
            steps.append(step)
            data, step = draw_horizontal_bar(data, h_start, x_min, y_max, h_t, x_max - x_min + l1, l2)
            steps.append(step)
        steps = gen_loop(steps)
        steps = np.asarray(steps)

    else:
        x_min = np.random.randint(-8, -3)
        y_min = np.random.randint(-12, -8)
        x_max = - x_min - l1
        y_max = - y_min - l2
        s_x = np.random.randint(-3, 2)
        s_y = 0
        x_min = x_min + s_x
        x_max = x_max + s_x
        y_min = y_min + s_y
        y_max = y_max + s_y

        data = np.zeros((32, 32, 32), dtype=np.uint8)
        steps = []
        data, step = draw_horizontal_bar(data, h_start, x_min, y_min, h_t, x_max - x_min + l1, l2)
        steps.append(step)
        data, step = draw_horizontal_bar(data, h_start, x_min, y_max, h_t, x_max - x_min + l1, l2)
        steps.append(step)
        steps = gen_loop(steps)
        steps = np.asarray(steps)

    return data, steps


# ======112======
def sample_sideboard_loop_single_2():
    q = np.random.rand()
    if q < 0.5:
        t = np.random.randint(5, 14)
        h = -t + np.random.choice([0, 1, 2, 3, 4], 1)[0]
    else:
        t = np.random.randint(2, 6)
        h = np.random.randint(-6, 5)

    q = np.random.rand()
    if q < 0.6:
        r1 = np.random.randint(3, 13)
        r2 = np.random.randint(1, 4)
        s1 = np.random.randint(-4, 4)
        s2 = np.random.randint(-12, -3)
    else:
        r1 = np.random.randint(3, 6)
        r2 = np.random.randint(1, 4)
        s1 = np.random.randint(-3, 3)
        s2 = np.random.randint(-12, -9)

    shift = np.random.randint(-2, 3)

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_sideboard(data, h, s1, s2 + shift, t, r1, r2)
    steps.append(step)
    data, step = draw_sideboard(data, h, s1, -s2-r2 + shift, t, r1, r2)
    steps.append(step)

    steps = gen_loop(steps)
    steps = np.asarray(steps)

    return data, steps


# ======113======
def sample_chair_beam_double():
    beam_h = np.random.randint(2, 5) + np.random.randint(0, 4)
    beam_start = np.random.randint(-6, 5)

    l1 = np.random.randint(1, 3)
    l2 = np.random.randint(1, 3)
    t = np.random.rand()
    if t < 0.2:
        l1 += 1
        l2 += 1

    p = np.random.rand()
    if p < 0.6:
        x_min = np.random.randint(-12, -3)
        y_min = np.random.randint(-12, -5)
    else:
        x_min = np.random.randint(-6, -2)
        y_min = np.random.randint(-12, -8)
    x_max = - x_min - l1
    y_max = - y_min - l2
    s_x = np.random.randint(-4, 3)
    s_y = 0

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_chair_beam(data, beam_start, x_min + s_x, y_min + s_y, beam_h, l1, l2)
    steps.append(step)
    data, step = draw_chair_beam(data, beam_start, x_max + s_x, y_min + s_y, beam_h, l1, l2)
    steps.append(step)
    data, step = draw_chair_beam(data, beam_start, x_max + s_x, y_max + s_y, beam_h, l1, l2)
    steps.append(step)
    data, step = draw_chair_beam(data, beam_start, x_min + s_x, y_max + s_y, beam_h, l1, l2)
    steps.append(step)

    steps = gen_loop(steps)
    steps = np.asarray(steps)

    return data, steps


# ======114======
def sample_chair_beam_single():
    beam_h = np.random.randint(2, 5) + np.random.randint(0, 4)
    beam_start = np.random.randint(-6, 5)

    l1 = np.random.randint(1, 3)
    l2 = np.random.randint(1, 3)
    t = np.random.rand()
    if t < 0.2:
        l1 += 1
        l2 += 1

    p = np.random.rand()
    if p < 0.6:
        x_min = np.random.randint(-12, -3)
        y_min = np.random.randint(-12, -5)
    else:
        x_min = np.random.randint(-6, -2)
        y_min = np.random.randint(-12, -8)
    x_max = - x_min - l1
    y_max = - y_min - l2
    s_x = np.random.randint(-3, 4)
    s_y = 0

    data = np.zeros((32, 32, 32), dtype=np.uint8)
    steps = []
    data, step = draw_chair_beam(data, beam_start, x_min + s_x, y_min + s_y, beam_h, l1, l2)
    steps.append(step)
    data, step = draw_chair_beam(data, beam_start, x_min + s_x, y_max + s_y, beam_h, l1, l2)
    steps.append(step)

    steps = gen_loop(steps)
    steps = np.asarray(steps)

    return data, steps


# ======115======
def sample_new_base_single():
    # sample cross base
    base_r = np.random.randint(6, 13)
    count = np.random.choice([3, 3, 4, 4, 5, 5, 5, 5, 6])
    leg_thickness = 1
    leg_end_offset = -np.random.choice([0, 1, 2, 3, 4], 1)[0]
    current_angle = 90 + np.random.randint(0, int(360 / count / 2.0), 1)[0]
    leg_h = np.random.randint(6, 14)
    leg_start = -leg_h + np.random.choice([0, 1, 2, 3, 4], 1)[0]
    x1 = leg_start-leg_thickness
    y1 = np.random.randint(-3, 4, 1)[0]
    z1 = np.random.randint(-3, 4, 1)[0]
    x2 = x1 + leg_end_offset
    y2 = np.rint(-base_r * np.sin(np.deg2rad(current_angle))) + y1
    z2 = np.rint(base_r * np.cos(np.deg2rad(current_angle))) + z1
    data = np.zeros((32, 32, 32), dtype=np.uint8)

    data, step = draw_new_base(data, x1, y1, z1, x2, y2, z2, count)
    steps = gen_loop(step)
    steps = np.asarray(steps)

    return data, steps
