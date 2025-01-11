import cv2
import numpy as np
import re

from src.tube.common.tube_base import Tube, Slice


class TubeGenerator:
    def __init__(self):
        # 经典的管道容器
        self.tubeBuffer = []

        # 两个常量，用于管道合并和相似性判断
        self.MAX_NON_UPDATE_TIME = 20
        self.SIMILARITY_TH = 0.1

    def assignObjects(self, frame, boundingRects, contours, frameIndex):
        """
        给帧中的运动目标分配管道
        :param frame: 当前帧
        :param boundingRects: 运动目标的矩形框列表
        :param contours: 运动目标的轮廓列表
        :param frameIndex: 当前帧索引
        """
        #print(f"shape(frame): {frame.shape}")
        print(f"len(boundingRects): {len(boundingRects)}")
        print(f"len(contours): {len(contours)}")
        #print(f"frameIndex: {frameIndex}")
        # 管道合并
        #print(f"1.len of self.tubeBuffer: {len(self.tubeBuffer)}")
        for rid in range(len(contours)):
            contour = contours[rid]

            mergeList = []
            for tid in range(len(self.tubeBuffer)):
                tube = self.tubeBuffer[tid]
                if tube.startFrame + tube.getLength() + self.MAX_NON_UPDATE_TIME > frameIndex:
                    if self.canBeAssigned(tube.frames, contour):
                        mergeList.append(tid)

            # 合并相关的管道
            if len(mergeList) > 1:
                self.mergeTubes(mergeList)
        #print(f"2.len of self.tubeBuffer: {len(self.tubeBuffer)}")
        # 新管道生成和目标分配
        suitableTubes = [[] for _ in range(len(self.tubeBuffer))]

        for rid in range(len(contours)):
            contour = contours[rid]

            suitableCnt = 0
            for tid in range(len(self.tubeBuffer)):
                tube = self.tubeBuffer[tid]
                if tube.startFrame + tube.getLength() + self.MAX_NON_UPDATE_TIME > frameIndex:
                    if self.canBeAssigned(tube.frames, contour):
                        suitableCnt += 1
                        suitableTubes[tid].append(rid)

            assert suitableCnt <= 1

            if suitableCnt == 0:
                suitableTubes.append([rid])
        #print(f"3.len of self.tubeBuffer: {len(self.tubeBuffer)}")
        for tid in range(len(suitableTubes)):
            if tid >= len(self.tubeBuffer):
                self.tubeBuffer.append(Tube(frameIndex, len(self.tubeBuffer)))

            newSlice = Slice()
            for rid in suitableTubes[tid]:
                newRect = boundingRects[rid]
                newContour = contours[rid]
                newSlice.boundingRects.append(newRect)
                newSlice.contours.append(newContour)

                x, y, width, height = newRect
                objectMap = frame[y:y + height, x:x + width].copy()

                newSlice.objects.append(objectMap)

            if newSlice.getObjNumber() > 0:
                self.tubeBuffer[tid].frames.append(newSlice)
        #print(f"4.len of self.tubeBuffer: {len(self.tubeBuffer)}")
        #print()

    def mergeTubes(self, mergeList):
        """
        合并多个管道
        :param mergeList: 待合并管道的索引列表
        """
        newTube = Tube(float('inf'), 0)
        upper = 0

        for mid in range(len(mergeList)):
            tid = mergeList[mid]
            newTube.startFrame = min(newTube.startFrame, self.tubeBuffer[tid].startFrame)
            upper = max(upper, self.tubeBuffer[tid].startFrame + self.tubeBuffer[tid].getLength())

        for sid in range(newTube.startFrame, upper):
            newSlice = Slice()
            for mid in range(len(mergeList)):
                tid = mergeList[mid]
                tube = self.tubeBuffer[tid]
                if tube.startFrame <= sid < tube.startFrame + tube.getLength():
                    slice = tube.frames[sid - tube.startFrame]
                    assert slice.getObjNumber() > 0
                    for rid in range(slice.getObjNumber()):
                        newSlice.boundingRects.append(slice.boundingRects[rid])
                        newSlice.contours.append(slice.contours[rid])
                        objectMap = slice.objects[rid].copy()
                        newSlice.objects.append(objectMap)

            if newSlice.getObjNumber() > 0:
                newTube.frames.append(newSlice)

        # 移除合并的管道并重新分配管道ID
        for i in range(len(mergeList) - 1, -1, -1):
            del self.tubeBuffer[mergeList[i]]

        self.tubeBuffer.append(newTube)
        for i in range(len(self.tubeBuffer)):
            self.tubeBuffer[i].id = i

    def canBeAssigned(self, slices, contour):
        """
        判断一个运动目标是否可以被分配给一个管道
        :param slices: 管道的一系列帧
        :param contour: 运动目标的轮廓
        :return: 是否可以分配
        """
        for revi in range(1, self.MAX_NON_UPDATE_TIME + 1):
            sid = len(slices) - revi
            if sid < 0:
                break

            slice = slices[sid]
            for otherContour in slice.contours:
                if self.contourIsOverlap(contour, otherContour) or self.contourIsOverlap(otherContour, contour):
                    return True

        return False

    def calcSimilarity(self, ra, rb):
        """
        计算两个矩形框的相似性
        :param ra: 矩形框A
        :param rb: 矩形框B
        :return: 相似性
        """
        unionRect = ra & rb
        combineRect = ra | rb
        return 1.0 * unionRect.area / combineRect.area

    def contourIsOverlap(self, ca, cb):
        """
        检测两个轮廓是否重叠
        :param ca: 轮廓A
        :param cb: 轮廓B
        :return: 是否重叠
        """
        for point in ca:
            # 将point[0]转换为字符串
            point_str = str(point[0])
            # 使用正则表达式提取数字
            coordinates = re.findall(r'\b\d+\b', point_str)
            if len(coordinates) == 2:
                x, y = map(int, coordinates)
                dis = cv2.pointPolygonTest(cb, (x, y), False)
                if dis > 0:
                    return True
        return False

    def showTubes(self):
        """
        显示所有管道
        """
        for tube in self.tubeBuffer:
            tube.showTube(tube)

    def saveTubes(self, fileName="tubes.txt"):
        """
        保存管道信息到文件
        :param fileName: 文件名，默认为"tubes.txt"
        """
        with open(fileName, "w") as f:
            f.write(f"{len(self.tubeBuffer)}\n")
            for tube in self.tubeBuffer:
                f.write(f"{tube.startFrame} {tube.getLength()}\n")

                for slice in tube.frames:
                    f.write(f"{slice.getObjNumber()}\n")

                    for rect in slice.boundingRects:
                        f.write(f"{rect[0]} {rect[1]} {rect[2]} {rect[3]}\n")

    def loadTubes(self, fileName):
        """
        从文件加载管道信息
        """
        with open(fileName, "r") as f:
            bufferSize = int(f.readline().strip())
            self.tubeBuffer = [Tube() for _ in range(bufferSize)]

            for tid in range(bufferSize):
                tube = self.tubeBuffer[tid]
                tube.startFrame, tubeLength = map(int, f.readline().split())
                tube.frames = [Slice() for _ in range(tubeLength)]

                for sid in range(tubeLength):
                    slice = tube.frames[sid]
                    objectNumber = int(f.readline().strip())
                    slice.boundingRects = []

                    for oid in range(objectNumber):
                        rect_values = list(map(int, f.readline().split()))
                        rect = cv2.Rect(rect_values[0], rect_values[1], rect_values[2], rect_values[3])
                        slice.boundingRects.append(rect)
                        print(type(rect))

    def sortTubes(self):
        """
        对管道进行排序
        """
        self.tubeBuffer.sort(key=lambda x: (x.startFrame, x.id))


