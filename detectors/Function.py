import numpy as np


def point_distance_line(point, line_point1, line_point2):
    """
    :param point:
    :param line_point1:
    :param line_point2:
    :return:
    计算点到直线的距离
    """
    #计算向量
    point = np.array(point, dtype=np.float64)
    line_point1 = np.array(line_point1, dtype=np.float64)
    line_point2 = np.array(line_point2, dtype=np.float64)

    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance


def isinline(point: tuple, line, precision=1e-3):
    """
    判断点是否在线上
    :param point: (x,y)
    :param line: [(x0,y0),(x1,y1)]
    :param precision:
    :return:
    """
    # 如果是垂直线
    minNum = 10 ** -5
    if abs(line[1][0] - line[0][0]) <= minNum:
        if abs(point[0] - line[0][0]) <= minNum:
            return True
        else:
            return False
    elif abs(point[0] - line[0][0]) <= minNum:
        if abs(line[1][0] - line[0][0]) <= minNum:
            return True
        else:
            return False
    else:
        line_slope = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])

        point_slope = (point[1] - line[0][1]) / (point[0] - line[0][0])

        # 当斜率与斜率的百分比误差在1e-3以内，就认为在线上
        if abs(line_slope) <= minNum:
            if abs(point_slope) <= minNum:
                return True
        elif np.abs((point_slope - line_slope) / line_slope) <= precision:
            return True
        else:
            return False

        return False


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def isinsegment(point: tuple, line, precision=1e-3, external=True):
    """
    判断点是否在线段上
    :param point:
    :param line:
    :param precision:
    :param external:
    :return:
    """
    # 先判断是不是在线上，在判断是不是在线段内
    if isinline(point, line, precision=precision):
        # 如果允许查找延长线上的点，只判断在线上就行了。
        if external:
            return True
        else:
            x_sign = np.sign((point[0] - line[1][0]) * (point[0] - line[0][0]))
            y_sign = np.sign((point[1] - line[1][1]) * (point[1] - line[0][1]))
            # 均异号，是在线段内
            if x_sign <= 0 and y_sign <= 0:
                return True
            else:
                return False
    else:
        return False


def line_intersection(line1, line2, external=True, precision=1e-3):
    """
     求两个线段是否相交，并返回交点
    :param line1:[(x0,y0),(x1,y1)]
    :param line2:[(x0,y0),(x1,y1)]
    :param external:是否判断其延长线相交
    :param precision: 精度，判断交点是否在线上时允许有一定的误差
    :return:
    """

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = det(xdiff, ydiff)
    if div == 0:
        # raise Exception('lines do not intersect')
        return None, None
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    # 允许交点在延长线上
    if external:
        return x, y
    # 交点是否在线段内
    elif isinsegment((x, y), line1, precision=precision, external=external) and isinsegment((x, y), line2,
                                                                                            precision=precision,
                                                                                            external=external):
        return x, y
    # 如果交点不再任何线上，返回None
    else:
        return None, None


def line_distance(line1, line2, external=True, precision=1e-3):
    """
    计算两条线段的距离，这个距离定义是如果两条线段相交则为0，平行则为实际距离，没有在线段内相交则返回None
    :param line1:[(x0,y0),(x1,y1)]
    :param line2:[(x0,y0),(x1,y1)]
    :param external:是否判断其延长线相交
    :param precision: 精度，判断交点是否在线上时允许有一定的误差
    :return:
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = det(xdiff, ydiff)
    if div == 0:
        # raise Exception('lines do not intersect')
        distance = point_distance_line((line1[0][0], line1[0][1]), (line2[0][0], line2[0][1]), (line2[1][0], line2[1][1]))
        return distance
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    # 允许交点在延长线上
    if external:
        return 0
    # 交点是否在线段内
    elif isinsegment((x, y), line1, precision=precision, external=external) and isinsegment((x, y), line2,
                                                                                            precision=precision,
                                                                                            external=external):
        return 0
    # 如果交点不再任何线上，返回None
    else:
        return None
