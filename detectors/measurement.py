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
