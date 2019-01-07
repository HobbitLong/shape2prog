from __future__ import print_function

import numpy as np


class Interpreter(object):
    """interpreting program vectors into understandable program strings"""
    def __init__(self, translate, rotate, end):
        self.translate = translate
        self.rotate = rotate
        self.end = end

    def interpret(self, pgm, param):

        n_block = pgm.shape[0]
        param = np.round(param).astype(np.int32)

        result = ""
        for i in range(n_block):
            res = self.interpret_block(pgm[i], param[i])
            if res is None:
                continue
            else:
                result += res
                result += "\n"

        return result

    def interpret_block(self, pgm, param):
        """
        interpret each block
        """
        flag = 1
        block_res = []
        if pgm[0] == self.translate:
            if pgm[1] == self.translate:
                if 1 <= pgm[2] < self.translate:

                    sentence = "for(i<{}, 'Trans', u1=({},{},{}))"\
                        .format(param[0, 0], param[0, 1], param[0, 2], param[0, 3])
                    block_res.append(sentence)

                    sentence = "for(i<{}, 'Trans', u2=({},{},{}))"\
                        .format(param[1, 0], param[1, 1], param[1, 2], param[1, 3])
                    block_res.append("    "+sentence)

                    sentence = self.interpret_sentence(pgm[2], param[2], num_trans=2, num_rot=0)
                    block_res.append("        "+sentence)

                else:
                    pass
            elif 1 <= pgm[1] < self.translate:

                sentence = "for(i<{}, 'Trans', u=({},{},{}))" \
                    .format(param[0, 0], param[0, 1], param[0, 2], param[0, 3])
                block_res.append(sentence)

                sentence = self.interpret_sentence(pgm[1], param[1], num_trans=1, num_rot=0)
                block_res.append("    " + sentence)

            else:
                pass
        elif pgm[0] == self.rotate:
            if pgm[1] == 10 or pgm[1] == 17:

                sentence = "for(i<{}, 'Rot', theta={}\N{DEGREE SIGN}, axis=({},{},{})"\
                    .format(param[0, 0], int(360/param[0,0]),
                            param[1, 0], param[1, 1], param[1, 2])
                block_res.append(sentence)

                sentence = self.interpret_sentence(pgm[1], param[1], num_trans=0, num_rot=1)
                block_res.append("    " + sentence)

            else:
                pass
        elif 1 <= pgm[0] < self.translate:

            sentence = self.interpret_sentence(pgm[0], param[0], num_trans=0, num_rot=0)
            block_res.append(sentence)

        else:
            pass

        if len(block_res) == 0:
            return None
        else:
            res = ''
            for i in range(len(block_res)):
                res += block_res[i] + '\n'
            return res

    def interpret_sentence(self, pgm, param, num_trans=0, num_rot=0):
        """
        interpret each sentence
        """
        if num_trans == 0 and num_rot == 0:
            if pgm == 1:
                sentence = "draw('Leg', 'Cub', P=({},{},{}), G=({},{},{}))"\
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 2:
                sentence = "draw('Top', 'Rec', P=({},{},{}), G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 3:
                sentence = "draw('Top', 'Square', P=({},{},{}), G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 4:
                sentence = "draw('Top', 'Circle', P=({},{},{}), G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 5:
                sentence = "draw('Layer', 'Rec', P=({},{},{}), G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 6:
                sentence = "draw('Sup', 'Cylinder', P=({},{},{}), G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 7:
                sentence = "draw('Sup', 'Cub', P=({},{},{}), G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 8:
                sentence = "draw('Base', 'Circle', P=({},{},{}), G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 9:
                sentence = "draw('Base', 'Square', P=({},{},{}), G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 10:
                angle = round(param[5]) % 4
                if angle == 0:
                    p1, p2, p3 = param[0], param[1], param[2] - param[4]
                elif angle == 1:
                    p1, p2, p3 = param[0], param[1] + param[4], param[2]
                elif angle == 2:
                    p1, p2, p3 = param[0], param[1], param[2] + param[4]
                elif angle == 3:
                    p1, p2, p3 = param[0], param[1] - param[4], param[2]
                else:
                    raise ValueError("The angle type of the cross is wrong")
                sentence = "draw('Base', 'Line', P1=({},{},{}), P2=({},{},{}))" \
                    .format(param[0], param[1], param[2], p1, p2, p3)
            elif pgm == 11:
                sentence = "draw('Sideboard', 'Cub', P=({},{},{}), G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 12:
                sentence = "draw('Hori_Bar', 'Cub', P=({},{},{}), G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 13:
                sentence = "draw('Vert_Board', 'Cub', P=({},{},{}), G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 14:
                sentence = "draw('Locker', 'Cub', P=({},{},{}), G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 15:
                theta = np.arctan(float(param[6])/param[3]) / np.pi * 180
                sentence = "draw('Back', 'Cub', P=({},{},{}), G=({},{},{}), theta={}\N{DEGREE SIGN})" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5], int(theta))
            elif pgm == 16:
                sentence = "draw('Chair_Beam', 'Cub', P=({},{},{}), G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 17:
                sentence = "draw('Connect', 'Line', P1=({},{},{}), P2=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 18:
                sentence = "draw('Back_sup', 'Cub', P=({},{},{}), G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif self.translate <= pgm <= self.end:
                sentence = None
            else:
                sentence = None

        elif num_trans == 1 and num_rot == 0:
            if pgm == 1:
                sentence = "draw('Leg', 'Cub', P=({},{},{})+i*u, G=({},{},{}))"\
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 2:
                sentence = "draw('Top', 'Rec', P=({},{},{})+i*u, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 3:
                sentence = "draw('Top', 'Square', P=({},{},{})+i*u, G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 4:
                sentence = "draw('Top', 'Circle', P=({},{},{})+i*u, G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 5:
                sentence = "draw('Layer', 'Rec', P=({},{},{})+i*u, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 6:
                sentence = "draw('Sup', 'Cylinder', P=({},{},{})+i*u, G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 7:
                sentence = "draw('Sup', 'Cub', P=({},{},{})+i*u, G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 8:
                sentence = "draw('Base', 'Circle', P=({},{},{})+i*u, G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 9:
                sentence = "draw('Base', 'Square', P=({},{},{})+i*u, G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 10:
                angle = round(param[5]) % 4
                if angle == 0:
                    p1, p2, p3 = param[0], param[1], param[2] - param[4]
                elif angle == 1:
                    p1, p2, p3 = param[0], param[1] + param[4], param[2]
                elif angle == 2:
                    p1, p2, p3 = param[0], param[1], param[2] + param[4]
                elif angle == 3:
                    p1, p2, p3 = param[0], param[1] - param[4], param[2]
                else:
                    raise ValueError("The angle type of the cross is wrong")
                sentence = "draw('Base', 'Line', P1=({},{},{})+i*u, P2=({},{},{}))+i*u" \
                    .format(param[0], param[1], param[2], p1, p2, p3)
            elif pgm == 11:
                sentence = "draw('Sideboard', 'Cub', P=({},{},{})+i*u, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 12:
                sentence = "draw('Hori_Bar', 'Cub', P=({},{},{})+i*u, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 13:
                sentence = "draw('Vert_Board', 'Cub', P=({},{},{})+i*u, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 14:
                sentence = "draw('Locker', 'Cub', P=({},{},{})+i*u, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 15:
                theta = np.arctan(float(param[6])/param[3]) / np.pi * 180
                sentence = "draw('Back', 'Cub', P=({},{},{})+i*u, G=({},{},{}), theta={}\N{DEGREE SIGN})" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5], int(theta))
            elif pgm == 16:
                sentence = "draw('Chair_Beam', 'Cub', P=({},{},{})+i*u, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 17:
                sentence = "draw('Connect', 'Line', P1=({},{},{})+i*u, P2=({},{},{}))+i*u" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 18:
                sentence = "draw('Back_sup', 'Cub', P=({},{},{})+i*u, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif self.translate <= pgm <= self.end:
                sentence = None
            else:
                sentence = None

        elif num_trans == 2 and num_rot == 0:
            if pgm == 1:
                sentence = "draw('Leg', 'Cub', P=({},{},{})+i*u1+j*u2, G=({},{},{}))"\
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 2:
                sentence = "draw('Top', 'Rec', P=({},{},{})+i*u1+j*u2, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 3:
                sentence = "draw('Top', 'Square', P=({},{},{})+i*u1+j*u2, G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 4:
                sentence = "draw('Top', 'Circle', P=({},{},{})+i*u1+j*u2, G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 5:
                sentence = "draw('Layer', 'Rec', P=({},{},{})+i*u1+j*u2, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 6:
                sentence = "draw('Sup', 'Cylinder', P=({},{},{})+i*u1+j*u2, G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 7:
                sentence = "draw('Sup', 'Cub', P=({},{},{})+i*u1+j*u2, G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 8:
                sentence = "draw('Base', 'Circle', P=({},{},{})+i*u1+j*u2, G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 9:
                sentence = "draw('Base', 'Square', P=({},{},{})+i*u1+j*u2, G=({},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4])
            elif pgm == 10:
                angle = round(param[5]) % 4
                if angle == 0:
                    p1, p2, p3 = param[0], param[1], param[2] - param[4]
                elif angle == 1:
                    p1, p2, p3 = param[0], param[1] + param[4], param[2]
                elif angle == 2:
                    p1, p2, p3 = param[0], param[1], param[2] + param[4]
                elif angle == 3:
                    p1, p2, p3 = param[0], param[1] - param[4], param[2]
                else:
                    raise ValueError("The angle type of the cross is wrong")
                sentence = "draw('Base', 'Line', P1=({},{},{})+i*u1+j*u2, P2=({},{},{}))+i*u1+j*u2" \
                    .format(param[0], param[1], param[2], p1, p2, p3)
            elif pgm == 11:
                sentence = "draw('Sideboard', 'Cub', P=({},{},{})+i*u1+j*u2, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 12:
                sentence = "draw('Hori_Bar', 'Cub', P=({},{},{})+i*u1+j*u2, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 13:
                sentence = "draw('Vert_Board', 'Cub', P=({},{},{})+i*u1+j*u2, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 14:
                sentence = "draw('Locker', 'Cub', P=({},{},{})+i*u1+j*u2, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 15:
                theta = np.arctan(float(param[6])/param[3]) / np.pi * 180
                sentence = "draw('Back', 'Cub', P=({},{},{})+i*u1+j*u2, G=({},{},{}), theta={}\N{DEGREE SIGN})" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5], int(theta))
            elif pgm == 16:
                sentence = "draw('Chair_Beam', 'Cub', P=({},{},{})+i*u1+j*u2, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 17:
                sentence = "draw('Connect', 'Line', P1=({},{},{})+i*u1+j*u2, P2=({},{},{}))+i*u" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif pgm == 18:
                sentence = "draw('Back_sup', 'Cub', P=({},{},{})+i*u1+j*u2, G=({},{},{}))" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            elif self.translate <= pgm <= self.end:
                sentence = None
            else:
                sentence = None

        elif num_trans == 0 and num_rot == 1:
            if pgm == 10:
                angle = round(param[5]) % 4
                if angle == 0:
                    p1, p2, p3 = param[0], param[1], param[2] - param[4]
                elif angle == 1:
                    p1, p2, p3 = param[0], param[1] + param[4], param[2]
                elif angle == 2:
                    p1, p2, p3 = param[0], param[1], param[2] + param[4]
                elif angle == 3:
                    p1, p2, p3 = param[0], param[1] - param[4], param[2]
                else:
                    raise ValueError("The angle type of the cross is wrong")
                sentence = "draw('Base', 'Line', P1=({},{},{}), P2=({},{},{}), theta*i, axis)" \
                    .format(param[0], param[1], param[2], p1, p2, p3)
            elif pgm == 17:
                sentence = "draw('Base', 'Line', P1=({},{},{}), P2=({},{},{}), theta*i, axis)" \
                    .format(param[0], param[1], param[2],
                            param[3], param[4], param[5])
            else:
                sentence = None

        else:
            sentence = None

        return sentence
