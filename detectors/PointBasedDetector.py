import cv2
import numpy
import math

import Function

from skimage import measure


class WaferCuttingLineDetector:
    def __init__(self):
        """
        算法依赖边缘提取效果，边缘提取效果越好，算法效果越好，相关函数_DetectEdge
        """
        self.FixedSuperParam = {
            "edgeThreshold": 100,

            "CuttingRegionOffset": 3,
            "CuttingRegionMerge": 10,

            "intervalThreshold": 3,
            "lineThreshold": 5,

            "FOLHLPThreshold": 100,
            "FOLHLPMinLineLength": 10,
            "FOLHLPMaxLineGap": 2,
            "FOLAngleThreshold": 1,
            "FOLInterPThreshold": 0.7,
            "FOLLineMaxDistance": 3,

            "GroupLineAngleStep": 3,
            "GroupLineStepScale": 100,

            "CCLLineDistanceMin": 40,
            "CCLLineDistanceMax": 130,
            "CCLLinePairInterAngleMax": 0.5,
            "CCLLinePairOuterAngleMax": 30,
            "CCLLinePairDistanceMin": 100,
            "CCLCuttingRegionCheckMaxSuccession": 15,
            "CCLCuttingRegionOffset": 1,
            "CCLCuttingRegionWidth": 1,
            "CCLEdgeOffset": 1,
        }

        """
        此处指明了各个步骤是否进行可视化
        相关可视化参数在此处调整即可
        """
        self.visualization = {
            "_DetectEdge": False,
            "_DetectCuttingRegion": False,
            "_DetectPoint": False,
            "_DetectLineThrough": False,
            "_FilterOutLineHLP": False,
            "_FilterOutLineResult": False,
            "_GroupLine": False,
            "_CheckCuttingLineDrawLinePair": False,
            "_CheckCuttingLineDrawCuttingRegion": False,
            "_GetMinimumAreaCombination": False,
        }

        """
        中间结果存储
        """
        self.makeMidOutput = False
        self.midOutput = dict()

    def SetMakeMidOutput(self, makeMidOutput):
        self.makeMidOutput = makeMidOutput

    def GetMidOutput(self):
        return self.midOutput

    def _SetMidOutput(self, name, image):
        self.midOutput[name] = image

    def ShowImage(self, image):
        cv2.imshow("result", image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def GetVisualization(self, progress):
        return self.visualization.get(progress, False)

    def GetSuperParam(self, param):
        """
        :param param:
        :return:
        考虑到有些数值的大小需要结合镜头和晶圆的距离、晶圆切割道的固有宽度标准等，所以可以调整此函数以结合各种外部数据动态调整各个超参数
        """
        return self.FixedSuperParam[param]

    def _CalTwoLineAngle(self, line1, line2):
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

    def _ClipAngle180(self, angle):
        while angle >= 180:
            angle -= 180
        while angle < 0:
            angle += 180
        return angle

    def _ClipAnglePN90(self, angle):
        while angle >= 90:
            angle -= 180
        while angle < -90:
            angle += 180
        return angle

    def _DetectEdge(self, image):
        """
        :param image: 需要提取边缘的原始图像
        :param threshold: 多高的数值被认为是边缘，即边缘阈值
        :return: 和image一样的图像，其中255表示边缘，0表示非边缘，但是结果是单通道（为了减少运算量和存储开销）
        """
        # 第二个参数基本上是数据类型，就是计算时候用的数据类型，防止截断误差
        x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        x = cv2.convertScaleAbs(x)
        y = cv2.convertScaleAbs(y)
        edgeImage = cv2.addWeighted(x, 0.5, y, 0.5, 0)
        edgeImage = numpy.where(edgeImage > self.GetSuperParam("edgeThreshold"), 255, 0).astype(numpy.uint8)

        if self.makeMidOutput:
            self._SetMidOutput("DetectEdge", edgeImage)
        if self.GetVisualization("_DetectEdge"):
            self.ShowImage(edgeImage)

        if len(edgeImage.shape) == 3:
            edgeImage = edgeImage[:, :, 0]
        return edgeImage

    def _DCRGetDegree(self, image, bbox, offset):
        inOutDegree = 0
        if bbox[0] <= offset:
            inOutDegree += 1
        if bbox[1] <= offset:
            inOutDegree += 1
        if bbox[2] >= image.shape[0] - offset:
            inOutDegree += 1
        if bbox[3] >= image.shape[1] - offset:
            inOutDegree += 1
        return inOutDegree

    def _DetectCuttingRegion(self, image):
        """
        :param: 边缘图
        :return:
        先看看连通域的BBox和边界有没有交集，没有的话这个连通域就不会是切割线的连通域
        因为如果是切割线旁边的那个区域的话，基本是一个连通域，而且都是至少从一边进入，从另一边出去，度至少大于2
        返回image, 如果是255，说明这个区域可以被切割，因为此区域有大于2的度。
        """
        image = 255 - image

        imageLabeled = measure.label(image, 0, connectivity=2)
        props = measure.regionprops(imageLabeled)

        cuttingRegionIndexList = list()
        noCuttingRegionIndexList = list()
        for index, prop in enumerate(props, start=1):
            """
            bbox: [tuple] Bounding box (min_row, min_col, max_row, max_col).
            Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col).
            """
            bbox = prop.bbox
            # 如果刚好在边缘出现了噪点，那么可能导致这个切割面无法有2个以上的度，所以可以用这个offset修改BBox的测试边界，把测试边界往内部收缩
            # 此处进行两次检测，即offset=0和offset=CuttingRegionOffset，
            # 所以offset设置为近似大于等于噪声点的宽度或长度。
            offset = self.GetSuperParam("CuttingRegionOffset")
            inOutDegree = self._DCRGetDegree(image, bbox, 0)
            inOutDegree |= self._DCRGetDegree(image, bbox, offset)

            if inOutDegree > 1:
                cuttingRegionIndexList.append(index)
            else:
                noCuttingRegionIndexList.append(index)

        image[:, :] = 0

        # 对单个区域进行膨胀收缩，填充掉内部的空缺部分，这样可以减少后续端点提取过程提取到的端点数，减少获取的线的数量
        # 但是过多的膨胀操作后，会导致方向不是标准的矩形不再呈现矩形形状，其边缘会微弱向外凸出，所以数值不应该过大，可以结合边线清晰程度确定
        mergeSize = self.GetSuperParam("CuttingRegionMerge")
        mergeSize = max(0, mergeSize // 2)

        dKernel = numpy.ones((3, 3), numpy.uint8)
        eKernel = numpy.ones((3, 3), numpy.uint8)

        oneRegion = numpy.zeros(image.shape, dtype=image.dtype)
        for index in cuttingRegionIndexList:
            oneRegion[:] = 0
            oneRegion[imageLabeled == index] = 255
            for i in range(mergeSize):
                oneRegion = cv2.dilate(oneRegion, dKernel)
            for i in range(mergeSize):
                oneRegion = cv2.erode(oneRegion, eKernel)

            image |= oneRegion

        if self.makeMidOutput:
            self._SetMidOutput("PossibleCleanArea", image)
        if self.GetVisualization("_DetectCuttingRegion"):
            self.ShowImage(image)
            self.ShowImage(255 - image)

        return image

    def _DetectPoint(self, image):
        """
        :param image: 输入图像
        :return: 掩码后的图像，image中只要是大于0的就被人为是有效点
        如果有个边缘区域在环状掩码区域中有过长的边界，则只会提取出这个边界的两个端点
        比如有一条长边缘刚好就在环状掩码下面，则只会提取出这个长边缘的端点。
        """
        offset = 1

        """
        intervalThreshold: 多小的间隔会被认为是同一个区域，单位：像素
        lineThreshold: 多大的连续区域会被认为是边线而不是端点，单位：像素
        """
        intervalThreshold = self.GetSuperParam("intervalThreshold")
        lineThreshold = self.GetSuperParam("lineThreshold")

        intervalThreshold = max(0, intervalThreshold // 2)
        lineThreshold = max(0, lineThreshold // 2)

        # 环状掩码
        maskHLeft = offset
        maskHRight = image.shape[0] - offset
        maskVTop = offset
        maskVBottom = image.shape[1] - offset

        image = numpy.copy(image)
        # 环状掩码
        image[maskHLeft:maskHRight, maskVTop:maskVBottom] = 0

        dKernel = numpy.ones((3, 3), numpy.uint8)
        eKernel = numpy.ones((3, 3), numpy.uint8)
        # 膨胀侵蚀融合
        for i in range(intervalThreshold):
            image = cv2.dilate(image, dKernel)
        for i in range(intervalThreshold):
            image = cv2.erode(image, eKernel)

        # 这一步不确定是否需要，不加好像效果也没有变化，而且可以加快速度，目前没有遇到离谱的案例，但是还是打开了
        image[maskHLeft:maskHRight, maskVTop:maskVBottom] = 0

        # 反转操作，方便提取端点
        imageNotLine = 255 - image
        imageNotLine[maskHLeft:maskHRight, maskVTop:maskVBottom] = 0
        # 边线端点提取，多次3x3小核处理
        # 如果要用大核一次性处理，那可能不能用矩形核，不太方便实现
        for i in range(lineThreshold):
            imageNotLine = cv2.dilate(imageNotLine, dKernel)
        for i in range(lineThreshold):
            imageNotLine = cv2.erode(imageNotLine, eKernel)

        # 获取端点
        dKernel = numpy.ones((3, 3), numpy.uint8)
        imageNotLineDilated = cv2.dilate(imageNotLine, dKernel)
        imagePoint = imageNotLineDilated - imageNotLine
        image = numpy.where(imageNotLine == 255, image, 0)
        image = image + imagePoint

        # 环状掩码
        image[maskHLeft:maskHRight, maskVTop:maskVBottom] = 0

        if self.GetVisualization("_DetectPoint"):
            self.ShowImage(image)

        return image

    def _DVLGetPointType(self, image, point):
        """
        :param point: (row, col)
        :return:
        """
        vt, ht = 0, 0
        if point[0] < 1:
            vt = -1
        elif point[0] > image.shape[0] - 2:
            vt = 1

        if point[1] < 1:
            ht = -1
        elif point[1] > image.shape[1] - 2:
            ht = 1

        return vt, ht

    def _DVLCheckPointSameType(self, image, point1, point2):
        """
        :param image:
        :param point1:
        :param point2:
        :return:
        检查两个点是否是同一个类型的点，如果是返回True，否则返回False
        """
        p1t = self._DVLGetPointType(image, point1)
        p2t = self._DVLGetPointType(image, point2)

        if (p1t[0] == 0 and p1t[1] == 0) or (p2t[0] == 0 and p2t[1] == 0):
            # 有些连通域的中心点不是在环状掩码上
            return False

        if p1t[0] == p2t[0] and p1t[1] == p2t[1]:
            return True

        if (p1t[0] != 0 and p1t[1] != 0) or (p2t[0] != 0 and p2t[1] != 0):
            # 如果其中一个是4个角落的角点
            if p1t[0] == p2t[0] or p1t[1] == p2t[1]:
                return True

        return False

    def _DVLShowPointAndLine(self, image, props, lineList, showWindow, midOutputName, showOriginImage=False):
        imageDraw = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=numpy.uint8)
        if showOriginImage:
            imageDraw[:, :, 0] = image
            imageDraw[:, :, 1] = image
            imageDraw[:, :, 2] = image
        for line in lineList:
            cv2.line(imageDraw,
                     (int(props[line[0]].centroid[1]), int(props[line[0]].centroid[0])),
                     (int(props[line[1]].centroid[1]), int(props[line[1]].centroid[0])),
                     (255, 255, 255), lineType=cv2.LINE_AA)
        for prop in props:
            cv2.circle(imageDraw, (int(prop.centroid[1]), int(prop.centroid[0])), 2, (0, 0, 255), 5)

        if self.makeMidOutput:
            self._SetMidOutput(midOutputName, imageDraw)

        if showWindow:
            self.ShowImage(imageDraw)

    def _CalTwoPointDistance(self, point1, point2):
        div1 = point1[0] - point2[0]
        div2 = point1[1] - point2[1]

        return math.sqrt(div1 * div1 + div2 * div2)

    def _ClipLineInImage(self, image, lineList):
        newLineList = []
        for line in lineList:
            borderLeft = [0, 0, 0, image.shape[0] - 1]
            borderTop = [0, 0, image.shape[1] - 1, 0]
            borderRight = [image.shape[1] - 1, 0, image.shape[1] - 1, image.shape[0] - 1]
            borderBottom = [0, image.shape[0] - 1, image.shape[1] - 1, image.shape[0] - 1]

            startPoint = (line[0], line[1])
            endPoint = (line[2], line[3])

            x1, y1 = Function.line_intersection(self._LineInClassToFunction(borderLeft), [startPoint, endPoint], False)
            x2, y2 = Function.line_intersection(self._LineInClassToFunction(borderTop), [startPoint, endPoint], False)
            x3, y3 = Function.line_intersection(self._LineInClassToFunction(borderRight), [startPoint, endPoint], False)
            x4, y4 = Function.line_intersection(self._LineInClassToFunction(borderBottom), [startPoint, endPoint], False)

            pointPairList = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            notNonePointPairList = list()
            for pointPair in pointPairList:
                if pointPair[0] is not None and pointPair[1] is not None:
                    notNonePointPairList.append(pointPair)

            pointPairList = notNonePointPairList
            uniqueIndexList = []
            for pSelf in range(len(pointPairList)):
                for pTo in range(len(pointPairList)):
                    if self._CalTwoPointDistance(pointPairList[pSelf], pointPairList[pTo]) < 0.1:
                        if pTo not in uniqueIndexList:
                            uniqueIndexList.append(pTo)
                        break

            point1 = pointPairList[uniqueIndexList[0]]
            point2 = pointPairList[uniqueIndexList[1]]
            newLineList.append([point1[0], point1[1], point2[0], point2[1]])
        return newLineList

    def _DetectLineThrough(self, image):
        """
        :param image: 需要处理的图像
        :return:
        算法将每个连通域认为是一个端点，然后测试端点两两之间是否应该存在连线
        """
        # 提取连通域
        imageLabeled = measure.label(image, 0, connectivity=2)
        props = measure.regionprops(imageLabeled)

        numOfRegions = len(props)
        # 生成连线，此处保证生成的连线是无向的
        lineList = []
        for i in range(numOfRegions):
            for j in range(i + 1, numOfRegions):
                # 可以将环状掩码的区域分为8个特殊分区，如果两个点位于同一个分区中，则不能将它们连线
                if self._DVLCheckPointSameType(image, props[i].centroid, props[j].centroid) is False:
                    lineList.append((i, j))

        # =========================================
        # 绘制连线和端点
        if self.GetVisualization("_DetectLineThrough") or self.makeMidOutput:
            self._DVLShowPointAndLine(image, props, lineList,
                                      self.GetVisualization("_DetectLineThrough"),
                                      "ThroughLinesCandidate")
        # =========================================
        return props, lineList

    def _FOLShowHLP(self, image, lineSegmentList, showWindow):
        imageDraw = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=numpy.uint8)
        imageDraw[:, :, 0] = image
        imageDraw[:, :, 1] = image
        imageDraw[:, :, 2] = image
        for lineSegment in lineSegmentList:
            cv2.line(imageDraw,
                     (int(lineSegment[0][0]), int(lineSegment[0][1])),
                     (int(lineSegment[0][2]), int(lineSegment[0][3])),
                     (0, 0, 255), lineType=cv2.LINE_AA)

        if self.makeMidOutput:
            self._SetMidOutput("HoughLineP", imageDraw)

        if showWindow:
            self.ShowImage(imageDraw)

    def _PropsLineToSE(self, props, line):
        """
        :param props:
        :param line:
        :return:
        返回坐标定义以cv2画线函数为准
        """
        return props[line[0]].centroid[1], props[line[0]].centroid[0], props[line[1]].centroid[1], props[line[1]].centroid[0]

    def _LineToBBox(self, line):
        left = min(line[0], line[2])
        right = max(line[0], line[2])

        top = min(line[1], line[3])
        bottom = max(line[1], line[3])

        return left, top, right, bottom

    def _CalIInterPInBBox2(self, bbox1, bbox2):
        sBBox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        leftLine = max(bbox1[1], bbox2[1])
        rightLine = min(bbox1[3], bbox2[3])
        topLine = max(bbox1[0], bbox2[0])
        bottomLine = min(bbox1[2], bbox2[2])

        if leftLine >= rightLine or topLine >= bottomLine:
            return 0
        else:
            intersect = (rightLine - leftLine) * (bottomLine - topLine)
            return (intersect / sBBox2) * 1.0

    def _PointInBBox(self, point, bbox):
        if bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]:
            return True
        return False

    def _GetSegmentDistance(self, lc, ls, compare=min):
        """
        :param lc: [x1, y1, x2, y2]
        :param ls:
        :param compare:
        :return:
        """
        lcInterCheck = self._LineInClassToFunction(lc)
        lsInterCheck = self._LineInClassToFunction(ls)
        distance = Function.line_distance(lcInterCheck, lsInterCheck, False)
        if distance is None:
            distance1 = Function.point_distance_line((ls[0], ls[1]), (lc[0], lc[1]), (lc[2], lc[3]))
            distance2 = Function.point_distance_line((ls[2], ls[3]), (lc[0], lc[1]), (lc[2], lc[3]))
            distance = compare(distance1, distance2)
        return distance

    def _FOLLineInter(self, lc, ls, image, method="minDistance"):
        """
        :param lc:
        :param ls:
        :param image:
        :param method: 可选：bbox, interPoint, midDistance, minDistance
        :return:
        """
        # 角度近似后，用线段的BBox判断是否重叠度高
        # 但是不合理，存在一定的问题
        if method == "bbox":
            bbox1 = self._LineToBBox(lc)
            bbox2 = self._LineToBBox(ls)
            percent = self._CalIInterPInBBox2(bbox1, bbox2)
            if percent >= self.GetSuperParam("FOLInterPThreshold"):
                return True

        # 以交点是否在图内判断是否叠加
        elif method == "interPoint":
            lcInterCheck = [(lc[0], lc[1]), (lc[2], lc[3])]
            lsInterCheck = [(ls[0], ls[1]), (ls[2], ls[3])]
            point1, point2 = Function.line_intersection(lcInterCheck, lsInterCheck, True)
            if point1 is not None and point2 is not None and \
                    self._PointInBBox((point1, point2), (0, 0, image.shape[0] - 1, image.shape[1] - 1)):
                return True

        # 用s线的中点和c线计算距离
        elif method == "midDistance":
            lsMidPoint = ((ls[0] + ls[2]) / 2, (ls[1] + ls[3]) / 2)
            distance = Function.point_distance_line(lsMidPoint, (lc[0], lc[1]), (lc[2], lc[3]))
            if distance <= self.GetSuperParam("FOLInterPointMaxDistance"):
                return True

        # 用s线的两端点和c线计算距离，用最小值作为距离
        # 如果存在交点则直接判定为重叠
        # 或者平行线距离小于2像素
        elif method == "minDistance":
            distance = self._GetSegmentDistance(lc, ls)
            if distance <= self.GetSuperParam("FOLLineMaxDistance"):
                return True
        return False

    def _ClipAnglePN90MF(self, angle):
        """
        :param angle:
        :return:
        注意这个函数依赖的是numpy的arccos，所以默认angle是在0到180
        而且会修改原始数据
        """
        angle[angle >= 90] -= 180
        return angle

    def _CalLineInterAngleMF(self, lineMatrix1, lineMatrix2):
        """
        :param lineMatrix1: (num of line, 4) numpy.array
        :param lineMatrix2: (num of line, 4) numpy.array
        :return:
        返回矩阵第i行第j列即 M1中第1条线和M2中第2条线的夹角
        """
        vectorM1 = lineMatrix1[:, 0:2] - lineMatrix1[:, 2:4]
        vectorM1Length = numpy.linalg.norm(vectorM1, axis=1)
        vectorM1Length = vectorM1Length[numpy.newaxis, :]

        vectorM2 = lineMatrix2[:, 0:2] - lineMatrix2[:, 2:4]
        vectorM2Length = numpy.linalg.norm(vectorM2, axis=1)
        vectorM2Length = vectorM2Length[numpy.newaxis, :]

        vectorMulMatrix = numpy.matmul(vectorM1, vectorM2.T)
        lengthMulMatrix = numpy.matmul(vectorM1Length.T, vectorM2Length)

        cosAngle = vectorMulMatrix / lengthMulMatrix
        cosAngle = numpy.clip(cosAngle, -1.0, 1.0)
        angle = numpy.arccos(cosAngle)
        angle = numpy.degrees(angle)

        return angle

    def _FilterOutLine(self, image, props, lineList):
        """
        :param image: 可以用于提取线段的图，二值图像
        :param props:
        :param lineList:
        :return:
        """
        edgeImage = self._DetectEdge(image)

        lineSegmentList = cv2.HoughLinesP(edgeImage, 1.0, numpy.pi/180,
                                          self.GetSuperParam("FOLHLPThreshold"),
                                          None,
                                          self.GetSuperParam("FOLHLPMinLineLength"),
                                          self.GetSuperParam("FOLHLPMaxLineGap"))

        if self.GetVisualization("_FilterOutLineHLP") or self.makeMidOutput:
            self._FOLShowHLP(image, lineSegmentList, self.GetVisualization("_FilterOutLineHLP"))

        # 这里设置为如果图中没有任何线段，那么就任何贯穿线都没有意义
        if len(lineSegmentList) == 0:
            return []

        # 过滤，就是过滤到完全不经过任何线段的贯穿线
        # ===========================================================
        # 矩阵方案，仅角度测试用了矩阵方案
        npLineList = []
        for line in lineList:
            lc = self._PropsLineToSE(props, line)
            npLineList.append(lc)
        npLineMatrix = numpy.array(npLineList)

        npSegmentList = []
        for lineSegment in lineSegmentList:
            ls = lineSegment[0]
            npSegmentList.append(ls)
        npSegmentMatrix = numpy.array(npSegmentList)

        angle = self._CalLineInterAngleMF(npLineMatrix, npSegmentMatrix)
        angle = self._ClipAnglePN90MF(angle)
        angle = numpy.abs(angle)
        angleMatched = angle <= self.GetSuperParam("FOLAngleThreshold")

        angleMatchedSum = numpy.sum(angleMatched, axis=1)

        newLineList = []
        for lcIndex in range(angleMatched.shape[0]):
            if angleMatchedSum[lcIndex]:
                for lsIndex in range(angleMatched.shape[1]):
                    if angleMatched[lcIndex, lsIndex]:
                        lc = npLineList[lcIndex]
                        ls = npSegmentList[lsIndex]
                        if self._FOLLineInter(lc, ls, image):
                            newLineList.append(lineList[lcIndex])
                            break
        # ===========================================================

        if self.GetVisualization("_FilterOutLineResult") or self.makeMidOutput:
            self._DVLShowPointAndLine(image, props, newLineList,
                                      self.GetVisualization("_FilterOutLineResult"),
                                      "FilterOutLineResult")

        return newLineList

    def _GroupLine(self, image, props, lineList):
        """
        :param image: 用于提取长宽，以及可视化，如果不可视化的话就是无用参数
        :param props:
        :param lineList:
        :return:
        倾角近似的线被分到同一组，因为如果是切割面，那么至少有两条相互平行的线
        效果取决于前面步骤对端点的提取以及step的设置
        超参中的步长用于确定多少度一个组
        """
        lineGroup = dict()

        hLine = numpy.array([0, 0, 1, 0], dtype=numpy.float64)
        for line in lineList:
            check, angle = self._CalTwoLineAngle(self._PropsLineToSE(props, line), hLine)
            # 无向，所以只需要限制在180内
            angle = self._ClipAngle180(angle)

            if check is False:
                # 直接丢弃这条线，因为上一个步骤已经剔除这种情况，但是还是防着
                continue

            step = self.GetSuperParam("GroupLineAngleStep")
            stepNum = int(angle / step)
            if (angle - stepNum * step) > (step / 2):
                stepNum += 1

            # 因为dict一般不用float作为key，所以如果step是float的话，乘上一个大的倍数scale转为整数，作为key
            scale = self.GetSuperParam("GroupLineStepScale")
            groupID = int(stepNum * step * scale)
            if groupID not in lineGroup.keys():
                lineGroup[groupID] = list()
            lineGroup[groupID].append(line)

        lineGroupNoSingle = dict()
        for key, value in lineGroup.items():
            # 只有一条线则不考虑
            if len(value) > 1:
                lineGroupNoSingle[key] = value
        lineGroup = lineGroupNoSingle

        # =========================================
        # 分组绘制线
        if self.GetVisualization("_GroupLine"):
            for key, value in lineGroup.items():
                self._DVLShowPointAndLine(image, props, value, self.GetVisualization("_GroupLine"), "None")
        # =========================================
        return lineGroup

    def _LineInClassToFunction(self, line):
        return [(line[0], line[1]), (line[2], line[3])]

    def _DrawLinePairList(self, image, props, linePairList, showOriginImage=False):
        imageDraw = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=numpy.uint8)
        if showOriginImage:
            imageDraw[:, :, 0] = image
            imageDraw[:, :, 1] = image
            imageDraw[:, :, 2] = image
        for linePair in linePairList:
            l1 = self._PropsLineToSE(props, linePair.l1)
            l2 = self._PropsLineToSE(props, linePair.l2)

            cv2.line(imageDraw,
                     (int(l1[0]), int(l1[1])),
                     (int(l1[2]), int(l1[3])),
                     (255, 255, 255), lineType=cv2.LINE_AA)

            cv2.line(imageDraw,
                     (int(l2[0]), int(l2[1])),
                     (int(l2[2]), int(l2[3])),
                     (255, 255, 255), lineType=cv2.LINE_AA)
        for prop in props:
            cv2.circle(imageDraw, (int(prop.centroid[1]), int(prop.centroid[0])), 2, (0, 0, 255), 5)
        self.ShowImage(imageDraw)

    class CCLLinePair:
        def __init__(self, props, l1, l2):
            """
            :param props:
            :param l1: [propsIndex1, propsIndex2]
            :param l2: [propsIndex1, propsIndex2]
            """
            self.props = props
            self.l1 = l1
            self.l2 = l2

        def __eq__(self, other):
            if (self.l1[0] == other.l1[0] and
                self.l1[1] == other.l1[1] and
                self.l2[0] == other.l2[0] and
                self.l2[1] == other.l2[1]) or (self.l1[0] == other.l2[0] and
                                               self.l1[1] == other.l2[1] and
                                               self.l2[0] == other.l1[0] and
                                               self.l2[1] == other.l1[1]):
                return True
            return False

    def _GetLinePairCenter(self, props, linePair: CCLLinePair):
        l1 = self._PropsLineToSE(props, linePair.l1)
        l2 = self._PropsLineToSE(props, linePair.l2)
        x = l1[0] + l1[2] + l2[0] + l2[2]
        y = l1[1] + l1[3] + l2[1] + l2[3]

        return x / 4, y / 4

    def _GetLinePairsAngle(self, props, linePair1: CCLLinePair, linePair2: CCLLinePair):
        """
        :param props:
        :param linePair1:
        :param linePair2:
        :return:
        不检测直线长度
        """
        lp1l1 = self._PropsLineToSE(props, linePair1.l1)
        lp1l2 = self._PropsLineToSE(props, linePair1.l2)
        lp2l1 = self._PropsLineToSE(props, linePair2.l1)
        lp2l2 = self._PropsLineToSE(props, linePair2.l2)

        lp1l = numpy.array([lp1l1, lp1l2])
        lp2l = numpy.array([lp2l1, lp2l2])

        angle = self._CalLineInterAngleMF(lp1l, lp2l)
        angle = self._ClipAnglePN90MF(angle)
        angle = numpy.abs(angle)
        return numpy.min(angle)

    def _GetLinePairsDistance(self, props, linePair1: CCLLinePair, linePair2: CCLLinePair):
        lp1l1 = self._PropsLineToSE(props, linePair1.l1)
        lp1l2 = self._PropsLineToSE(props, linePair1.l2)
        lp2l1 = self._PropsLineToSE(props, linePair2.l1)
        lp2l2 = self._PropsLineToSE(props, linePair2.l2)

        distance11 = self._GetSegmentDistance(lp1l1, lp2l1)
        distance12 = self._GetSegmentDistance(lp1l1, lp2l2)
        distance21 = self._GetSegmentDistance(lp1l2, lp2l1)
        distance22 = self._GetSegmentDistance(lp1l2, lp2l2)

        distance = min(distance11, distance12, distance21, distance22)
        return distance

    def _LinePairCheckCompatible(self, props, linePair1: CCLLinePair, linePair2: CCLLinePair):
        deltaAngle = self._GetLinePairsAngle(props, linePair1, linePair2)
        # 如果两个直线对的夹角大于阈值，则这两个直线对是相容的
        if deltaAngle >= self.GetSuperParam("CCLLinePairOuterAngleMax"):
            return True

        distance = self._GetLinePairsDistance(props, linePair1, linePair2)
        # 如果两个直线对之间的最小距离大于阈值，则直线对相容
        if distance >= self.GetSuperParam("CCLLinePairDistanceMin"):
            return True

        return False

    class CCLCombination:
        def __init__(self, linePairList):
            self.area = 0
            self.cuttingRegionError = 0
            self.edgeError = 0
            self.linePairList = linePairList

    def _CCLMakeCuttingRegion(self, image, imageCuttingRegionOne, props, linePair):
        """
        :param image:
        :param imageCuttingRegionOne:
        :param props:
        :param linePair:
        :return:
        """
        l1 = self._PropsLineToSE(props, linePair.l1)
        l2 = self._PropsLineToSE(props, linePair.l2)

        # 这里的画线不是为了可视化，不能加抗锯齿
        cv2.line(imageCuttingRegionOne,
                 (int(l1[0]), int(l1[1])),
                 (int(l1[2]), int(l1[3])),
                 255, thickness=2)

        cv2.line(imageCuttingRegionOne,
                 (int(l2[0]), int(l2[1])),
                 (int(l2[2]), int(l2[3])),
                 255, thickness=2)

        imageLabeled = measure.label(255 - imageCuttingRegionOne, 0, connectivity=2)
        x, y = self._GetLinePairCenter(props, linePair)
        regionMaskId = imageLabeled[int(y), int(x)]
        image[imageLabeled == regionMaskId] = 255

        cv2.line(image,
                 (int(l1[0]), int(l1[1])),
                 (int(l1[2]), int(l1[3])),
                 255, thickness=2)

        cv2.line(image,
                 (int(l2[0]), int(l2[1])),
                 (int(l2[2]), int(l2[3])),
                 255, thickness=2)

        return image

    def _CCLCheckCombinationParam(self, checkImage, props, selectedLinePairList):
        """
        :param checkImage: 如果是255 ，表示切割区域，如果是0，表示切割线
        :param props:
        :param selectedLinePairList:
        :return:  返回Combination的各种参数
        """
        image = numpy.zeros((checkImage.shape[0], checkImage.shape[1]), dtype=numpy.uint8)
        imageCuttingRegionOne = numpy.zeros((checkImage.shape[0], checkImage.shape[1]), dtype=numpy.uint8)
        s33Kernel = numpy.ones((3, 3), numpy.uint8)

        for linePair in selectedLinePairList:
            imageCuttingRegionOne[:, :] = 0
            image = self._CCLMakeCuttingRegion(image, imageCuttingRegionOne, props, linePair)

        imageDilatedLast = image
        imageDilatedCurrent = image

        # 只考虑膨胀为一条线
        # for i in range(self.GetSuperParam("CCLCuttingRegionOffset")):
        #     tmp = imageDilatedCurrent
        #     imageDilatedCurrent = cv2.dilate(imageDilatedCurrent, s33Kernel)
        #     imageDilatedLast = tmp

        # 考虑膨胀为一个面积
        for i in range(self.GetSuperParam("CCLCuttingRegionOffset")):
            imageDilatedLast = cv2.dilate(imageDilatedLast, s33Kernel)
        imageDilatedCurrent = imageDilatedLast
        for i in range(self.GetSuperParam("CCLCuttingRegionWidth")):
            imageDilatedCurrent = cv2.dilate(imageDilatedCurrent, s33Kernel)

        imageDilatedDelta = imageDilatedCurrent - imageDilatedLast

        imageErodedLast = image
        imageErodedCurrent = image
        for i in range(self.GetSuperParam("CCLEdgeOffset")):
            tmp = imageErodedCurrent
            imageErodedCurrent = cv2.erode(imageErodedCurrent, s33Kernel)
            imageErodedLast = tmp
        imageErodedDelta = imageErodedLast - imageErodedCurrent

        # 如果以当前的LinePair来切割，那么其对应的安全区域是哪个
        checkCuttingRegionOffset = imageDilatedDelta
        # 如果以当前的LinePair来切割，那么其对应的切割线是哪个
        checkCuttingEdge = imageErodedDelta
        targetCuttingRegionOffset = 255 - checkImage
        imageCRCross = checkCuttingRegionOffset & targetCuttingRegionOffset

        # 如果ImageCRCross存在过大的连续区域，则这个combination直接判定无效
        # 因为是切割区域，安全区域，不可能有过大的连续区域
        # 由于已经是线形，所以基本上面积就是长度
        valid = True
        imageLabeled = measure.label(imageCRCross, 0, connectivity=2)
        props = measure.regionprops(imageLabeled)
        maxSuccession = 0
        for prop in props:
            if maxSuccession < prop.area:
                maxSuccession = prop.area
        if maxSuccession > self.GetSuperParam("CCLCuttingRegionCheckMaxSuccession"):
            return False, 9999999, 9999999, 9999999

        targetCuttingEdge = checkImage
        # 边线比较细，所以可以扩展处理
        targetCuttingEdge = cv2.erode(targetCuttingEdge, s33Kernel)
        imageERCross = checkCuttingEdge & targetCuttingEdge

        # self.ShowImage(targetCuttingRegionOffset)
        # self.ShowImage(imageCRCross)
        #
        # self.ShowImage(targetCuttingEdge)
        # self.ShowImage(imageERCross)

        # 总面积数
        cuttingRegionArea = numpy.sum(image) // 255
        # 总误差数
        cuttingRegionError = numpy.sum(imageCRCross) // 255
        edgeError = numpy.sum(imageERCross) // 255

        return valid, cuttingRegionArea, cuttingRegionError, edgeError

    def _CCLFindCombination(self, checkImage, props, linePairList, validLinePairList, selectedLinePairList, combinationList):
        while len(validLinePairList) > 0:
            linePair = validLinePairList.pop()
            newValidLinePairList = list()
            for linePairCheck in validLinePairList:
                if self._LinePairCheckCompatible(props, linePairCheck, linePair):
                    newValidLinePairList.append(linePairCheck)

            newSelectedLinePairList = selectedLinePairList.copy()
            newSelectedLinePairList.append(linePair)

            valid, area, cuttingRegionError, edgeError = self._CCLCheckCombinationParam(checkImage, props, newSelectedLinePairList)
            if valid:
                combination = WaferCuttingLineDetector.CCLCombination(newSelectedLinePairList)
                combination.area = area
                combination.cuttingRegionError = cuttingRegionError
                combination.edgeError = edgeError
                combinationList.append(combination)

            self._CCLFindCombination(checkImage, props, linePairList, newValidLinePairList.copy(), newSelectedLinePairList, combinationList)

    def _CCLDrawCuttingRegion(self, image, props, linePair, justLine=False):
        l1 = self._PropsLineToSE(props, linePair.l1)
        l2 = self._PropsLineToSE(props, linePair.l2)

        if justLine is False:
            imageRegion = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=numpy.uint8)
            cv2.line(imageRegion,
                     (int(l1[0]), int(l1[1])),
                     (int(l1[2]), int(l1[3])),
                     (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            cv2.line(imageRegion,
                     (int(l2[0]), int(l2[1])),
                     (int(l2[2]), int(l2[3])),
                     (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            imageLabeled = measure.label(255 - imageRegion, 0, connectivity=2)
            x, y = self._GetLinePairCenter(props, linePair)
            regionMaskId = imageLabeled[int(y), int(x)]

            image[imageLabeled == regionMaskId] = 255

        cv2.line(image,
                 (int(l1[0]), int(l1[1])),
                 (int(l1[2]), int(l1[3])),
                 (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        cv2.line(image,
                 (int(l2[0]), int(l2[1])),
                 (int(l2[2]), int(l2[3])),
                 (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        return image

    def _CCLDrawCombinationList(self, image, props, combinationList, justLine=False, showOriginImage=False):
        for combination in combinationList:
            imageDraw = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=numpy.uint8)
            if showOriginImage:
                if len(image.shape) == 3:
                    imageDraw[:, :, :] = image
                else:
                    imageDraw[:, :, 0] = image
                    imageDraw[:, :, 1] = image
                    imageDraw[:, :, 2] = image

            for linePair in combination.linePairList:
                imageDraw = self._CCLDrawCuttingRegion(imageDraw, props, linePair, justLine)
            self.ShowImage(imageDraw)

    def _CCLSortCombination(self, combination):
        if combination.area == 0:
            return 9999999999
        return (combination.cuttingRegionError + combination.edgeError) // len(combination.linePairList)

    def _CheckCuttingLine(self, checkImage, props, lineGroup):
        """
        :param checkImage: 255为可切割的区域
        :param props:
        :param lineGroup:
        :return:
        """
        # 第一步：形成直线对，保证没有对称重复
        linePairList = list()

        for key, value in lineGroup.items():
            npLineList = []
            for index in range(len(value)):
                lc = self._PropsLineToSE(props, value[index])
                npLineList.append(lc)
            npLineMatrix = numpy.array(npLineList)
            angle = self._CalLineInterAngleMF(npLineMatrix, npLineMatrix)
            angle = self._ClipAnglePN90MF(angle)
            angle = numpy.abs(angle)
            angleMatched = angle <= self.GetSuperParam("CCLLinePairInterAngleMax")
            angleMatchedSum = numpy.sum(angleMatched, axis=1) - 1

            for l1Index in range(angleMatched.shape[0]):
                if angleMatchedSum[l1Index]:
                    for l2Index in range(l1Index+1, angleMatched.shape[1]):
                        if angleMatched[l1Index][l2Index]:
                            l1 = npLineList[l1Index]
                            l2 = npLineList[l2Index]
                            lDistance = self._GetSegmentDistance(l1, l2)
                            # 线与线之间的距离也就是切割道的宽度限制
                            if self.GetSuperParam("CCLLineDistanceMin") <= lDistance <= self.GetSuperParam(
                                    "CCLLineDistanceMax"):
                                newLinePair = WaferCuttingLineDetector.CCLLinePair(props, value[l1Index],
                                                                                   value[l2Index])
                                linePairList.append(newLinePair)

        if self.GetVisualization("_CheckCuttingLineDrawLinePair"):
            self._DrawLinePairList(checkImage, props, linePairList)
        # =========================================

        if len(linePairList) == 0:
            return []

        # 第二步：检查
        validLinePairList = linePairList.copy()
        selectedLinePairList = list()
        combinationList = list()
        self._CCLFindCombination(checkImage, props, linePairList, validLinePairList, selectedLinePairList, combinationList)

        if len(combinationList) > 0:
            combinationList.sort(key=self._CCLSortCombination)

        # =========================================
        # 绘制组合切割区域
        if self.GetVisualization("_CheckCuttingLineDrawCuttingRegion"):
            self._CCLDrawCombinationList(checkImage, props, combinationList)
        # =========================================

        return combinationList

    def _GetMinimumAreaCombination(self, image, props, combinationList):
        """
        :param props:
        :param combinationList:
        :return:
        寻找的过程会依据最大评分的分组数进行限制，就是如果最大评分有2个LinePair，那么输出结果不会是只有1个LinePair的
        """
        if len(combinationList) == 0:
            return None
        bestCombination = combinationList[0]
        resultCombination = combinationList[0]
        for i in range(1, len(combinationList)):
            if len(combinationList[i].linePairList) == len(bestCombination.linePairList):
                if combinationList[i].area < resultCombination.area:
                    resultCombination = combinationList[i]

        if self.GetVisualization("_GetMinimumAreaCombination"):
            self._CCLDrawCombinationList(image, props, [resultCombination], True, True)
        return resultCombination

    def _PPNormLineDirection(self, line):
        """
        :param line: [x1, y1, x2, y2]
        :return:
        如果是以水平线向右为初始方向，且顺时针的话，那么基本是判定靠上的点为起点。
        """
        if line[1] > line[3]:
            line = [line[2], line[3], line[0], line[1]]
        elif line[1] == line[3]:
            # 如果高度一样则以靠左优先
            if line[0] > line[2]:
                line = [line[2], line[3], line[0], line[1]]
        return line

    def _Postprocess(self, image, props, combination):
        """
        :param image:
        :param props:
        :param combination:
        :return:
        点和图像轴的对应关系依据cv2.line，即得到的坐标点可以直接用到cv2.line上，不需要修改顺序
        若要用到其他地方，需要结合cv2.line的点和图像的关系调整点的顺序
        """
        if combination is None:
            return None

        borderLeft = [0, 0, 0, image.shape[0] - 1]
        borderTop = [0, 0, image.shape[1] - 1, 0]
        borderRight = [image.shape[1] - 1, 0, image.shape[1] - 1, image.shape[0] - 1]
        borderBottom = [0, image.shape[0] - 1, image.shape[1] - 1, image.shape[0] - 1]

        combinationInfo = dict()
        combinationInfo["cutting_path_num"] = len(combination.linePairList)
        combinationInfo["cutting_path_list"] = list()

        for linePair in combination.linePairList:
            centerX, centerY = self._GetLinePairCenter(props, linePair)
            # 以第一条线为基准
            l1 = self._PropsLineToSE(props, linePair.l1)
            vector = (l1[0] - l1[2], l1[1] - l1[3])

            cuttingPathInfo = dict()

            startPoint = (centerX - 100 * vector[0], centerY - 100 * vector[1])
            endPoint = (centerX + 100 * vector[0], centerY + 100 * vector[1])

            # 不考虑点在图像内的贯穿线
            cuttingLineThrough = [startPoint[0], startPoint[1], endPoint[0], endPoint[1]]
            cuttingLineThrough = self._PPNormLineDirection(cuttingLineThrough)

            x1, y1 = Function.line_intersection(self._LineInClassToFunction(borderLeft), [startPoint, endPoint], False)
            x2, y2 = Function.line_intersection(self._LineInClassToFunction(borderTop), [startPoint, endPoint], False)
            x3, y3 = Function.line_intersection(self._LineInClassToFunction(borderRight), [startPoint, endPoint], False)
            x4, y4 = Function.line_intersection(self._LineInClassToFunction(borderBottom), [startPoint, endPoint], False)

            pointPairList = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            notNonePointPairList = list()
            for pointPair in pointPairList:
                if pointPair[0] is not None and pointPair[1] is not None:
                    notNonePointPairList.append(pointPair)

            pointPairList = notNonePointPairList
            uniqueIndexList = []
            for pSelf in range(len(pointPairList)):
                for pTo in range(len(pointPairList)):
                    if self._CalTwoPointDistance(pointPairList[pSelf], pointPairList[pTo]) < 0.1:
                        if pTo not in uniqueIndexList:
                            uniqueIndexList.append(pTo)
                        break

            # 考虑点在图像内的贯穿线
            point1 = pointPairList[uniqueIndexList[0]]
            point2 = pointPairList[uniqueIndexList[1]]
            cuttingLine = [point1[0], point1[1], point2[0], point2[1]]
            cuttingLine = self._PPNormLineDirection(cuttingLine)

            hLine = [0, 0, 1, 0]
            _, angle = self._CalTwoLineAngle(cuttingLine, hLine)

            cuttingPathLine1 = self._PPNormLineDirection(self._PropsLineToSE(props, linePair.l1))
            cuttingPathLine2 = self._PPNormLineDirection(self._PropsLineToSE(props, linePair.l2))

            x1, y1 = Function.line_intersection(self._LineInClassToFunction(borderLeft),
                                                self._LineInClassToFunction(cuttingPathLine1), True)
            x2, y2 = Function.line_intersection(self._LineInClassToFunction(borderLeft),
                                                self._LineInClassToFunction(cuttingPathLine2), True)
            if y1 <= y2:
                cuttingPathUp = cuttingPathLine1
                cuttingPathBottom = cuttingPathLine2
            else:
                cuttingPathUp = cuttingPathLine2
                cuttingPathBottom = cuttingPathLine1

            cuttingPathInfo["cutting_line_through"] = cuttingLineThrough
            cuttingPathInfo["cutting_line"] = cuttingLine
            cuttingPathInfo["angle_of_cutting_line"] = angle
            cuttingPathInfo["cutting_path_up"] = cuttingPathUp
            cuttingPathInfo["cutting_path_bottom"] = cuttingPathBottom

            combinationInfo["cutting_path_list"].append(cuttingPathInfo)
        return combinationInfo

    def DetectWaferCuttingLine(self, image):
        self.midOutput = dict()

        oriEdgeImage = self._DetectEdge(image)

        cuttingImage = self._DetectCuttingRegion(oriEdgeImage)
        """
        255是可以被切掉的区域，那么取反之后就包括切线
        后续步骤中有端点提取，所以不需要再次提取边界
        """
        edgeImage = 255 - cuttingImage

        pointImage = self._DetectPoint(edgeImage)

        props, lineList = self._DetectLineThrough(pointImage)
        if len(lineList) == 0:
            return None

        lineList = self._FilterOutLine(edgeImage, props, lineList)
        if len(lineList) == 0:
            return None

        # 分组
        # 可能会导致一些差值很小的线被分到不同组
        # lineGroup = self._GroupLine(pointImage, props, lineList)

        # 不分组
        lineGroup = {"0": lineList}

        combinationList = self._CheckCuttingLine(255 - oriEdgeImage, props, lineGroup)

        # 找相对面积最小的作为输出
        combination = self._GetMinimumAreaCombination(image, props, combinationList)

        CombinationInfo = self._Postprocess(image, props, combination)

        return CombinationInfo
