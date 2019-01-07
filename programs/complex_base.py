from .utils import *
from .loop_gen import decode_loop
from copy import deepcopy
from .loop_gen import translate, rotate, end


def draw_new_base(data, x1, y1, z1, x2, y2, z2, counts):
    if counts not in [3, 4, 5, 6]:
        raise ValueError
    data_copy = deepcopy(data)
    data, step = draw_line(data, int(x1), int(y1), int(z1), int(x2), int(y2), int(z2))
    old_steps = [[rotate, counts], step, [end]]
    data = data_copy
    steps = list(map(list, decode_loop(old_steps)))
    for step in steps:
        data, null_step = draw_line(data, *step[1:])
    return data, old_steps
