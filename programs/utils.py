import numpy as np
from numpy.linalg import norm


###########################
# all the positions here are relative
# positions to center (16, 16, 16)
###########################


def draw_vertical_leg(data, h, s1, s2, t, r1, r2):
    """
    "Leg", "Cuboid"
    :param (h, s1, s2): position
    :param (t, r1, r2): shape
    """
    data[cut(16 + h):cut(16 + h + t), cut(16 + s1):cut(16 + s1 + r1), cut(16 + s2):cut(16 + s2 + r2)] = 1
    step = np.asarray([1, h, s1, s2, t, r1, r2])
    return data, step


def draw_rectangle_top(data, h, c1, c2, t, r1, r2):
    """
    "Top", "Rectangle"
    :param (h, c1, c2): position
    :param (r1, r2, t): shape
    """
    data[cut(16 + h):cut(16 + h + t), cut(16 + c1 - r1):cut(16 + c1 + r1), cut(16 + c2 - r2):cut(16 + c2 + r2)] = 1
    step = np.asarray([2, h, c1, c2, t, r1, r2])
    return data, step


def draw_square_top(data, h, c1, c2, t, r):
    """
    "Top", "Square"
    :param (h, c1, c2): position
    :param (r, t): shape
    """
    data[cut(16 + h):cut(16 + h + t), cut(16 + c1 - r):cut(16 + c1 + r), cut(16 + c2 - r):cut(16 + c2 + r)] = 1
    step = np.asarray([3, h, c1, c2, t, r])
    return data, step


def draw_circle_top(data, h, c1, c2, t, r):
    """
    "Top", "Circle"
    :param (h, c1, c2): position
    :param (r, t): shape
    """
    data = draw_cylinder(data, h, c1, c2, t, r)
    step = np.asarray([4, h, c1, c2, t, r])
    return data, step


def draw_middle_rect_layer(data, h, c1, c2, t, r1, r2):
    """
    "Layer", "Rectangle"
    :param (h, c1, c2): position
    :param (r1, r2, t): shape
    """
    data[cut(16 + h):cut(16 + h + t), cut(16 + c1 - r1):cut(16 + c1 + r1), cut(16 + c2 - r2):cut(16 + c2 + r2)] = 1
    step = np.asarray([5, h, c1, c2, t, r1, r2])
    return data, step


def draw_circle_support(data, h, c1, c2, t, r):
    """
    "Support", "Cylindar"
    :param (h, c1, c2): position
    :param (r, t): shape
    """
    data = draw_cylinder(data, h, c1, c2, t, r)
    step = np.asarray([6, h, c1, c2, t, r])
    return data, step


def draw_square_support(data, h, c1, c2, t, r):
    """
    "Support", "Cuboid"
    :param (h, c1, c2): position
    :param (r, t): shape
    """
    data[cut(16 + h):cut(16 + h + t), cut(16 + c1 - r):cut(16 + c1 + r), cut(16 + c2 - r):cut(16 + c2 + r)] = 1
    step = np.asarray([7, h, c1, c2, t, r])
    return data, step


def draw_circle_base(data, h, c1, c2, t, r):
    """
    "Base", "Circle"
    :param (h, c1, c2): position
    :param (r, t): shape
    """
    data = draw_cylinder(data, h, c1, c2, t, r)
    step = np.asarray([8, h, c1, c2, t, r])
    return data, step


def draw_square_base(data, h, c1, c2, t, r):
    """
    "Base", "Square"
    :param (h, c1, c2): position
    :param (r, t): shape
    """
    data[cut(16 + h):cut(16 + h + t), cut(16 + c1 - r):cut(16 + c1 + r), cut(16 + c2 - r):cut(16 + c2 + r)] = 1
    step = np.asarray([9, h, c1, c2, t, r])
    return data, step


def draw_cross_base(data, h, c1, c2, t, r, angle):
    """
    "Base", "Cross" ("line")
    :param (h, c1, c2): position, angle: angle position
    :param (r, t): shape
    TODO: extend to multiple angles, more than 3
    """
    angle = round(angle) % 4
    if angle == 0:
        data[cut(16 + h):cut(16 + h + t), cut(16 - 1 + c1):cut(16 + 1 + c1), cut(16 + c2 - r):cut(16 + c2)] = 1
    elif angle == 1:
        data[cut(16 + h):cut(16 + h + t), cut(16 + c1):cut(16 + c1 + r), cut(16 - 1 + c2):cut(16 + 1 + c2)] = 1
    elif angle == 2:
        data[cut(16 + h):cut(16 + h + t), cut(16 - 1 + c1):cut(16 + 1 + c1), cut(16 + c2):cut(16 + c2 + r)] = 1
    elif angle == 3:
        data[cut(16 + h):cut(16 + h + t), cut(16 - r + c1):cut(16 + c1), cut(16 - 1 + c2):cut(16 + 1 + c2)] = 1
    else:
        raise ValueError("The angle type of the cross is wrong")

    step = np.asarray([10, h, c1, c2, t, r, angle])
    return data, step


def draw_sideboard(data, h, s1, s2, t, r1, r2):
    """
    "Sideboard", "Cuboid"
    :param (h, s1, s2): position
    :param (r1, r2, t): shape
    """
    data[cut(16 + h):cut(16 + h + t), cut(16 + s1 - r1):cut(16 + s1 + r1), cut(16 + s2):cut(16 + s2 + r2)] = 1
    step = np.asarray([11, h, s1, s2, t, r1, r2])
    return data, step


def draw_horizontal_bar(data, h, s1, s2, t, r1, r2):
    """
    "Horizontal_Bar", "Cuboid"
    :param (h, s1, s2): position
    :param (r1, r2, t): shape
    """
    data[cut(16 + h):cut(16 + h + t), cut(16 + s1):cut(16 + s1 + r1), cut(16 + s2):cut(16 + s2 + r2)] = 1
    step = np.asarray([12, h, s1, s2, t, r1, r2])
    return data, step


def draw_vertboard(data, h, s1, s2, t, r1, r2):
    """
    "Vertical_board", "Cuboid"
    :param (h, s1, s2): position
    :param (r1, r2, t): shape
    """
    data[cut(16 + h):cut(16 + h + t), cut(16 + s1):cut(16 + s1 + r1), cut(16 + s2):cut(16 + s2 + r2)] = 1
    step = np.asarray([13, h, s1, s2, t, r1, r2])
    return data, step


def draw_locker(data, h, s1, s2, t, r1, r2):
    """
    "Locker", "Cuboid"
    :param (h, s1, s2): position
    :param (r1, r2, t): shape
    """
    data[cut(16 + h):cut(16 + h + t), cut(16 + s1):cut(16 + s1 + r1), cut(16 + s2):cut(16 + s2 + r2)] = 1
    step = np.asarray([14, h, s1, s2, t, r1, r2])
    return data, step


def draw_tilt_back(data, h, s1, s2, t, r1, r2, tilt_fact):
    """
    "Back", "Cuboid"
    :param (h, s1, s2): position
    :param (t, r1, r2, tilt_fact): shape
    """
    if t == 0:
        tilt_amount = tilt_fact
    else:
        tilt_amount = tilt_fact/t
    for h_i in range(16+h, 16+h+t):
        if h_i == 32:
            break
        data[cut(h_i), cut(int(16+np.rint(tilt_amount*(h_i-16-h)+s1))):cut(int(16+np.rint(tilt_amount*(h_i-16-h)+s1))+r1), cut(16+s2):cut(16+s2+r2)] = 1
    step = np.asarray([15, h, s1, s2, t, r1, r2, tilt_fact])
    return data, step


def draw_chair_beam(data, h, s1, s2, t, r1, r2):
    """
    "Chair_Beam", "Cuboid"
    :param (h, s1, s2): position
    :param (t, r1, r2): shape
    """
    data[cut(16 + h):cut(16 + h + t), cut(16 + s1):cut(16 + s1 + r1), cut(16 + s2):cut(16 + s2 + r2)] = 1
    step = np.asarray([16, h, s1, s2, t, r1, r2])
    return data, step


def draw_line(data, x1, y1, z1, x2, y2, z2, r=1):
    """
    "Line", "line"
    draw a line from (x1, y1, z1) to (x2, y2, z2) with a radius of r; the sampling version
    """
    eps = 1e-5
    full_l = 32
    half_l = full_l // 2
    p1, p2 = np.array([x1, y1, z1]) + half_l, np.array([x2, y2, z2]) + half_l
    if z1 > z2:
        p1, p2 = p2, p1

    line_len = norm(p2 - p1)
    if line_len == 0:
        line_a = (p2 - p1) * 0
    else:
        line_a = (p2 - p1) / line_len

    sample_n = int(np.ceil(np.pi * r * r * line_len * 10))
    # print(np.dot(np.random.rand(sample_n, 1), np.reshape(p2 - p1, (1, 3))))
    p3 = np.around(p1 + np.dot(np.random.rand(sample_n, 1), np.reshape(p2 - p1, (1, 3))) + (
            np.random.rand(sample_n, 3) - 0.5) * 2 * r).astype(int)
    p3 = np.clip(p3, 0, full_l - 1)
    # print(p3)
    # quit()

    line_b = p3 - p1
    proj = np.dot(line_b, line_a)

    mask_1 = np.logical_and(proj >= 0 - eps, proj <= line_len + eps)
    proj[np.logical_not(mask_1)] = 0
    dis = np.sqrt(norm(line_b, axis=1) ** 2 - proj ** 2 + eps * (eps + 2 * line_len))

    mask_2 = dis <= r + eps
    selected = np.array(p3[np.logical_and(mask_1, mask_2)])
    data[selected[:, 0], selected[:, 1], selected[:, 2]] = 1

    return data, [17, x1, y1, z1, x2, y2, z2, r]


def draw_back_support(data, h, s1, s2, t, r1, r2):
    """
    "Back_support", "Cuboid"
    :param (h, s1, s2): position
    :param (t, r1, r2): shape
    """
    data[cut(16+h):cut(16+h+t), cut(16+s1):cut(16+s1+r1), cut(16+s2):cut(16+s2+r2)] = 1
    step = np.asarray([18, h, s1, s2, t, r1, r2])
    return data, step


def draw_cylinder(data, h, c1, c2, t, r):
    """
    :param h: position
    :param (r, t): shape
    """
    d = dictance2center(16 + c1, 16 + c2)
    mask = np.where(d <= r * 1.01)
    for i in range(cut(16 + h), cut(16 + h + t)):
        data[i, ...][mask] = 1
    return data


def cut(x):
    """
    :param x: position of a voxel unit
    :return: cutoff x that go outside of voxel boundaries
    """
    if x < 0:
        return 0
    elif x > 32:
        return 32
    else:
        return x


def dictance2center(c1, c2):
    """
    :param x:
    :param y:
    :return:
    """
    x = np.arange(32)
    y = np.arange(32)
    xx, yy = np.meshgrid(x, y)
    xx = xx + 0.5
    yy = yy + 0.5
    d = np.sqrt(np.square(xx - c2) + np.square(yy - c1))

    return d

