import cv2
import numpy as np
from src.utils.disjoint_set import DisjointSets

class ContourFinder:
    def __init__(self, history=1000, nMixtures=3, contourSizeThreshold=0.1, medianFilterSize=9, contourMergeThreshold=0.01):
        self.foreground = None
        self.contourSizeThreshold = contourSizeThreshold
        self.medianFilterSize = int(medianFilterSize)
        self.contourMergeThreshold = contourMergeThreshold
        self.diagonal = 0.0
        self.suppressRectangles = []
        self.history = history
        self.nMixtures = nMixtures
        self.massCenters = []

    def translate(self, rect, widthHeight):
        # 将矩形平移并返回平移后的点
        return (rect[0] + widthHeight[0] * rect[2], rect[1] + widthHeight[1] * rect[3], rect[2], rect[3])

    def distanceBetweenRects(self, a, b):
        # 计算两个矩形之间的最小距离
        pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

        minDist = cv2.norm(np.array(a[:2]) - np.array(b[:2]))
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                minDist = min(minDist,
                              cv2.norm(np.array(self.translate(a, pairs[i]))
                                       - np.array(self.translate(b, pairs[j]))
                                       ))

        return minDist

    def filterOutBadContours(self, contours):
        # 移除面积太小的轮廓
        areas = [cv2.contourArea(contour) for contour in contours]
        maxArea = max(areas, default=0)

        threshold = self.contourSizeThreshold * maxArea

        return [contour for contour in contours if cv2.contourArea(contour) > threshold]

    def getCentersAndBoundingBoxes(self, contours, massCenters, boundingBoxes):
        # 获取轮廓的质心和边界框
        massCenters.clear()
        boundingBoxes.clear()

        # 使用多边形拟合轮廓
        contourPolygons = [cv2.approxPolyDP(np.array(contour), 3, True) for contour in contours]

        for i in range(len(contours)):
            # 计算质心
            contourMoments = cv2.moments(contours[i], False)
            # 添加条件判断避免除以零错误
            if contourMoments['m00'] != 0:
                massCenters.append(
                    (contourMoments['m10'] / contourMoments['m00'], contourMoments['m01'] / contourMoments['m00']))
            else:
                massCenters.append((0, 0))

            # 计算由轮廓表示的多边形的边界框
            boundingBoxes.append(cv2.boundingRect(contourPolygons[i]))

        return massCenters, boundingBoxes

    def suppressRectangle(self, rect):
        # 抑制给定矩形
        self.suppressRectangles.append(rect)

    def suppressMassCenters(self, contours, massCenters, boundingBoxes):
        # 抑制质心位于被抑制矩形内的轮廓
        for i in range(len(contours) - 1, -1, -1):
            for j in range(len(self.suppressRectangles) - 1, -1, -1):
                if self.suppressRectangles[j].contains(massCenters[i]):
                    contours.pop(i)
                    massCenters.pop(i)
                    boundingBoxes.pop(i)
                    break

    def mergeContours(self, contours, massCenters, boundingBoxes):
        # 合并附近的轮廓
        sets = DisjointSets(len(contours))

        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                if self.distanceBetweenRects(boundingBoxes[i], boundingBoxes[j]) < self.contourMergeThreshold * self.diagonal:
                    sets.union(i, j)

        itemsForSet = {}
        for i in range(len(contours)):
            if sets.findSet(i) not in itemsForSet:
                itemsForSet[sets.findSet(i)] = []
            itemsForSet[sets.findSet(i)].append(i)

        newContours = []
        for indices in itemsForSet.values():
            if len(indices) == 1:
                newContours.append(contours[indices[0]])
            else:
                aggregate = contours[indices[0]].copy()
                for i in range(1, len(indices)):
                    aggregate = np.concatenate((aggregate, contours[indices[i]]))
                newContours.append(aggregate)
        contours = list(contours)
        contours.clear()
        contours.extend(newContours)
        return contours

    def findContours(self, frame, fgFrame, hierarchy, contours, massCenters, boundingBoxes):
        # 设置对角线长度
        self.diagonal = np.sqrt(frame.shape[0] ** 2 + frame.shape[1] ** 2)

        # 清空轮廓和层次结构对象
        contours.clear()
        hierarchy.clear()

        # 查找前景
        self.foreground = fgFrame
        _, self.foreground = cv2.threshold(self.foreground, 130, 255, cv2.THRESH_BINARY)

        # 通过中值模糊去除噪点
        #print("Median filter size:", self.medianFilterSize)
        cv2.medianBlur(self.foreground, int(self.medianFilterSize), self.foreground)

        # 查找轮廓
        contours, hierarchy = cv2.findContours(self.foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 仅保留足够大的轮廓
        self.filterOutBadContours(contours)

        # 获取质心和边界框
        massCenters, boundingBoxes = self.getCentersAndBoundingBoxes(contours, massCenters, boundingBoxes)

        # 移除出现在被抑制矩形中的质心
        self.suppressMassCenters(contours, massCenters, boundingBoxes)

        # 合并附近的轮廓
        contours = self.mergeContours(contours, massCenters, boundingBoxes)

        # 再次查找质心和边界框
        massCenters, boundingBoxes = self.getCentersAndBoundingBoxes(contours, massCenters, boundingBoxes)

        # 对于车辆和人，最好计算相应轮廓的凸包。
        hull = [cv2.convexHull(contour, False) for contour in contours]
        contours = list(contours)
        contours.clear()
        contours.extend(hull)

        # 近似轮廓以减少点数
        epsilon = 3.0
        reducedContours = [cv2.approxPolyDP(contour, epsilon, False) for contour in contours]
        contours.clear()
        contours.extend(reducedContours)
        #print(self.massCenters)
        if len(massCenters) > 0:
            #print(self.massCenters)
            self.massCenters.append(massCenters)
        return contours, massCenters, boundingBoxes

    def shift(self, shift_amount):
        # 将质心在时间轴上左右移动
        for i in range(len(self.massCenters)):
            # 移动质心对应的时间
            self.massCenters[i] = (self.massCenters[i][0] + shift_amount, self.massCenters[i][1])

    def reset_contours(self):
        # 重置质心位置
        self.massCenters = [(0, 0) for _ in self.massCenters]