#!/usr/bin/env python3

import numpy

def CalTwoLineAngle(line1, line2):
    """
    :param line1: [x1, y1, x2, y2]
    :param line2: [x1, y1, x2, y2]
    :return: 是否成功，角度
    此函数会检测线的长度，如果过小会返回错误
    """
    l1p1 = numpy.array([line1[0], line1[1]], dtype=numpy.float64)
    l1p2 = numpy.array([line1[2], line1[3]], dtype=numpy.float64)
    l1Vector = l1p1 - l1p2
    l1VLen = numpy.linalg.norm(l1Vector)
    if l1VLen <= 0.00001:
        return False, 0

    l2p1 = numpy.array([line2[0], line2[1]], dtype=numpy.float64)
    l2p2 = numpy.array([line2[2], line2[3]], dtype=numpy.float64)
    l2Vector = l2p1 - l2p2
    l2VLen = numpy.linalg.norm(l2Vector)
    if l2VLen <= 0.00001:
        return False, 0

    cosAngle = numpy.dot(l1Vector, l2Vector) / (l1VLen * l2VLen)
    cosAngle = numpy.clip(cosAngle, -1.0, 1.0)
    angle = numpy.arccos(cosAngle)
    # 弧度制转为角度
    angle = numpy.degrees(angle)
    return True, angle


def GetAnnotation():
    """
    line: [x1, y1, x2, y2]
    path: [line1(up),line2(bottom)]
    annotation: [path1, path2]
    annotations = [annotation1, annotation2, ...]
    [[x1, y1, x2, y2],
    """
    annotations = {
        "1": [[[0, 269, 799, 260], [0, 337, 799, 328]]],
        "2": [[[0, 239, 799, 300], [0, 306, 799, 368]]],
        "3": [[[0, 307, 799, 229], [0, 376, 799, 297]]],
        "4": [[[289, 0, 295, 599], [357, 0, 364, 599]], [[0, 276, 799, 258], [0, 335, 799, 326]]],
    }

    annotationsOutput = dict()
    for key, value in annotations.items():
        pathList = []
        for path in value:
            l1, l2 = path
            centerP1X = (l1[0] + l2[0]) / 2
            centerP1Y = (l1[1] + l2[1]) / 2
            centerP2X = (l1[2] + l2[2]) / 2
            centerP2Y = (l1[3] + l2[3]) / 2

            _, angle = CalTwoLineAngle([0, 0, 1, 0], [centerP1X, centerP1Y, centerP2X, centerP2Y])

            pathList.append({
                "cutting_line": [centerP1X, centerP1Y, centerP2X, centerP2Y],
                "cutting_line_up": l1,
                "cutting_line_bottom": l2,
                "cutting_line_angle": angle
            })

        annotationsOutput[key] = pathList

    return annotationsOutput


def CalcPrecision(sawing_lines, annotation, golden):
    reses = []
    for sawing_line_id, sawing_line in enumerate(sawing_lines):
        res, angle = CalTwoLineAngle([0, 0, 1, 0], [sawing_line[0], sawing_line[1], sawing_line[2], sawing_line[3]])
        rule = annotation[golden][sawing_line_id]
        golden_sawing_line = rule["cutting_line"]
        center_of_golden = [(golden_sawing_line[0] + golden_sawing_line[2]) / 2, (golden_sawing_line[1] + golden_sawing_line[3]) / 2]
        center_of_sawing_line = [(sawing_line[0] + sawing_line[2]) / 2, (sawing_line[1] + sawing_line[3]) / 2]
        import math
        res = math.sqrt((angle - rule["cutting_line_angle"]) ** 2 + (center_of_golden[0] - center_of_sawing_line[0]) ** 2 + (center_of_golden[1] - center_of_sawing_line[1]) ** 2)
        reses.append(res)
    return reses

annotation = GetAnnotation()

res_by_ND = [
    ([[0, 309, 799, 295]], "1"),
    ([[0, 272, 799, 341]], "2"),
    ([[0, 346, 799, 262]], "3"),
    ([[321, 0, 332, 598], [0, 305, 799, 291]], "4"),
]

print("=ND=")
for res in res_by_ND:
    precisions = CalcPrecision(res[0], annotation, res[1])
    print("===" + res[1] + "===")
    for precision in precisions:
        print(precision)

res_by_LTD = [
    ([[0, 302.75, 799, 294.75]], "1"),
    ([[0, 272.875, 799, 334.375]], "2"),
    ([[0, 341.375, 799, 262.875]], "3"),
    ([[323.0, 0, 329.0, 599.0], [0, 301, 799, 292]], "4"),
]

print("=LTD=")
for res in res_by_LTD:
    precisions = CalcPrecision(res[0], annotation, res[1])
    print("===" + res[1] + "===")
    for precision in precisions:
        print(precision)
