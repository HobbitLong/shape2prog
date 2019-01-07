from itertools import groupby
from copy import deepcopy
from numpy import array, int_, ndarray, rint, sin, cos, deg2rad, arctan2, pi
from .label_config import max_param

# translate = "translate"
# rotate = "rotate"
# end = "end"
translate = 19
rotate = 20
end = 21


def pad_list(to_pad, length, filler=0):
    new_a = to_pad + [0] * (length - len(to_pad))
    return new_a


def gen_loop(input_data):
    input_batch = list(map(list, deepcopy(input_data)))

    # We identify unique groups that share the same id and z index
    action_ids = list(map(lambda x: str(x[0]), input_batch))
    action_groups = []
    for k, g in groupby(action_ids):
        action_groups.append(list(g))

    start_idx = 0
    return_sequence = []

    for group in action_groups:

        new_sequence = []
        end_idx = start_idx + len(group)

        # isolate the elements we want to loop over
        elements_in_group = input_batch[start_idx:end_idx]

        # Once isolated, we can change the index to the next group
        start_idx = end_idx

        obj_type = int(group[0].split(";")[0])

        if (obj_type == 1 or obj_type == 16) and len(group) >= 4:
            # translate is times, offset x, offset y, offset z
            new_sequence = [[translate, 2, 0, 0, elements_in_group[-1][3] - elements_in_group[0][3], 0, 0, 0],
                            [translate, 2, 0, elements_in_group[1][2] - elements_in_group[0][2], 0, 0, 0, 0],
                            elements_in_group[0],
                            [end, 0, 0, 0, 0, 0, 0, 0],
                            [end, 0, 0, 0, 0, 0, 0, 0]]

        if (obj_type == 10) and len(group) == 4:
            # rotate is times, rotate_count
            new_sequence = [[rotate, 4, 1, 0, 0, 0, 0],
                            elements_in_group[0],
                            [end, 0, 0, 0, 0, 0, 0, 0]]

        if (obj_type == 17) or (obj_type == rotate) or (obj_type == end):
            pass

        if (obj_type == 12) and len(group) == 4:

            new_sequence = [[translate, 2, 0, 0, elements_in_group[1][3] - elements_in_group[0][3], 0, 0, 0],
                            elements_in_group[0],
                            [end, 0, 0, 0, 0, 0, 0],
                            [translate, 2, 0, elements_in_group[3][2] - elements_in_group[2][2], 0, 0, 0, 0],
                            elements_in_group[2],
                            [end, 0, 0, 0, 0, 0, 0]]

        if (obj_type == 12) and len(group) == 3:

            new_sequence = [[translate, 2, 0, 0, elements_in_group[1][3] - elements_in_group[0][3], 0, 0, 0],
                            elements_in_group[0],
                            [end, 0, 0, 0, 0, 0, 0],
                            elements_in_group[2]]

        elif (obj_type != 0) and len(group) == 2:

            diff_count = 0
            diff_idx = 0

            for param_idx in range(1, len(elements_in_group[0])):
                if elements_in_group[0][param_idx] != elements_in_group[1][param_idx]:
                    diff_count += 1
                    diff_idx = param_idx

            if diff_count == 1:
                if diff_idx in [1, 2, 3]:
                    new_sequence = [[translate, 2, 0, 0, 0, 0, 0],
                                    elements_in_group[0],
                                    [end, 0, 0, 0, 0, 0, 0]]
                    new_sequence[0][diff_idx+1] = elements_in_group[1][diff_idx] - elements_in_group[0][diff_idx]
                elif diff_idx == 6 and obj_type == 10:
                    new_sequence = [[rotate, 2, elements_in_group[1][diff_idx]-elements_in_group[0][diff_idx], 0, 0, 0, 0],
                                    elements_in_group[0],
                                    [end, 0, 0, 0, 0, 0, 0]]

        if len(new_sequence) == 0:
            for i in range(len(elements_in_group)):
                return_sequence.append(list(elements_in_group[i]))

        elif len(new_sequence) != 0:
            for i in range(len(new_sequence)):
                return_sequence.append(new_sequence[i])
        return_sequence = list(map(lambda x:pad_list(x, max_param), return_sequence))
    return array(return_sequence, dtype=int_)


def decode_loop(input_batch):
    def valid_check(input_list):
        s = []
        balanced = True
        index = 0
        while index < len(input_list) and balanced:
            token = input_list[index][0]
            if token in [translate, rotate]:
                s.append(token)
            elif token == end:
                if len(s) == 0:
                    balanced = False
                else:
                    s.pop()
            index += 1
        if not (balanced and len(s) == 0):
            raise IndexError
        return None

    def find_matching_end(batch_input, cur_idx):
        loops_so_far = 0

        if cur_idx == len(batch_input) - 1:
            raise IndexError

        for item_count in range(cur_idx, len(batch_input)):
            if batch_input[item_count][0] in [translate, rotate]:
                loops_so_far += 1
            elif batch_input[item_count][0] == end:
                loops_so_far -= 1

            if loops_so_far == 0:
                return item_count
            if loops_so_far < 0:
                raise IndentationError

        return None

    if type(input_batch) == ndarray:
        accumulator_list = deepcopy(input_batch.tolist())
    else:
        accumulator_list = deepcopy(input_batch)
    valid_check(accumulator_list)

    item_idx = 0
    while 1:
        if item_idx >= len(accumulator_list) - 1:
            return array(accumulator_list, dtype=int_)

        if not accumulator_list[item_idx][0] in [translate, rotate]:
            item_idx += 1
            pass
        else:

            loop_end = find_matching_end(accumulator_list, item_idx)
            if loop_end is not None:

                unrolled_section = accumulator_list[item_idx + 1:loop_end]
                post_loop = accumulator_list[loop_end + 1:]
                pre_loop = accumulator_list[:item_idx]
                for_condition = accumulator_list[item_idx]

                for loop_count in range(0, for_condition[1]):
                    for items_in_loop in unrolled_section:
                        items_in_loop_copy = deepcopy(items_in_loop)
                        if (not items_in_loop[0] in [translate, rotate]) and (items_in_loop[0] != end):

                            if for_condition[0] == rotate:
                                if items_in_loop_copy[0] == 10:
                                    items_in_loop_copy[6] = items_in_loop[6] + for_condition[2] * loop_count
                                elif items_in_loop_copy[0] == 17:
                                    rot_time = min(for_condition[1], 6)
                                    rot_time = max(rot_time, 3)
                                    origin_y, origin_z = items_in_loop[2], items_in_loop[3]
                                    sin_calc = sin(deg2rad(360/rot_time*loop_count))
                                    cos_calc = cos(deg2rad(360/rot_time*loop_count))
                                    y_init_offset = items_in_loop[2] - origin_y
                                    z_init_offset = items_in_loop[3] - origin_z
                                    y_fnal_offset = items_in_loop[5] - origin_y
                                    z_fnal_offset = items_in_loop[6] - origin_z
                                    y_init_new = (cos_calc * y_init_offset - sin_calc * z_init_offset) + origin_y
                                    z_init_new = (sin_calc * y_init_offset + cos_calc * z_init_offset) + origin_z
                                    y_fnal_new = (cos_calc * y_fnal_offset - sin_calc * z_fnal_offset) + origin_y
                                    z_fnal_new = (sin_calc * y_fnal_offset + cos_calc * z_fnal_offset) + origin_z
                                    items_in_loop_copy[2] = rint(y_init_new)
                                    items_in_loop_copy[3] = rint(z_init_new)
                                    items_in_loop_copy[5] = rint(y_fnal_new)
                                    items_in_loop_copy[6] = rint(z_fnal_new)
                            elif for_condition[0] == translate:
                                if for_condition[2] != 0:
                                    items_in_loop_copy[1] = items_in_loop[1] + for_condition[2] * loop_count
                                if for_condition[3] != 0:
                                    items_in_loop_copy[2] = items_in_loop[2] + for_condition[3] * loop_count
                                if for_condition[4] != 0:
                                    items_in_loop_copy[3] = items_in_loop[3] + for_condition[4] * loop_count
                        pre_loop = pre_loop + [list(items_in_loop_copy)]
                accumulator_list = pre_loop + post_loop
            else:
                raise IndexError

