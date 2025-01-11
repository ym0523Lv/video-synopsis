import cv2
import numpy as np

class Slice:
    def __init__(self):
        """
        Slice类用于表示视频帧中的切片。

        Attributes:
        - boundingRects (List[cv2.Rect]): 切片中对象的边界矩形列表。
        - contours (List[np.ndarray]): 切片中对象的轮廓列表。
        - objects (List[cv2.Mat]): 切片中对象的图像列表。
        """
        self.boundingRects = []
        self.contours = []
        self.objects = []

    def getObjNumber(self):
        """
        获取切片中对象的数量。

        Returns:
        - int: 切片中对象的数量。
        """
        return len(self.boundingRects)

    def printRects(self):
        """输出切片中每个对象的边界矩形。"""
        for rect in self.boundingRects:
            print(rect)

    def getArea(self):
        """
        计算切片中所有对象边界矩形的总面积。

        Returns:
        - float: 切片中所有对象边界矩形的总面积。
        """
        area = 0.0
        for rect in self.boundingRects:
            area += rect.area()
        return area

    def isOverlap(self, other):
        """
        检查当前切片与另一个切片是否有重叠的对象。

        Parameters:
        - other (Slice): 另一个切片对象。

        Returns:
        - bool: 如果有重叠的对象，则返回True；否则返回False。
        """
        for i in range(self.getObjNumber()):
            for j in range(other.getObjNumber()):
                rect1 = self.boundingRects[i]
                rect2 = other.boundingRects[j]
                x1, y1, w1, h1 = rect1
                x2, y2, w2, h2 = rect2
                if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
                    return True
        return False


class Tube:
    def __init__(self, startFrame, tubeId):
        """
        Tube类用于表示视频管道。

        Parameters:
        - startFrame (int): 管道起始帧在原始视频中的帧号。
        - tubeId (int): 管道ID。
        """
        self.startFrame = startFrame
        self.id = tubeId
        self.finalPlace = 0
        self.saturDeg = 0.0
        self.rearrangeable = True
        self.frames = []

    def getLength(self):
        """
        获取管道包含的帧数。

        Returns:
        - int: 管道的长度（帧数）。
        """
        return len(self.frames)

    def showTube(self, tube):
        """
        显示管道的帧信息。

        Parameters:
        - tube (Tube): 管道对象。
        """
        rng = np.random.default_rng(12345)
        color = (rng.integers(0, 256), rng.integers(0, 256), rng.integers(0, 256))
        name = f"tube {self.id}"
        print(f"{name}: {self.getLength()}")

        for j in range(self.getLength()):
            sliceObj = self.frames[j]
            contourCanvas = np.zeros((660, 518, 3), dtype=np.uint8)

            for rid, rect in enumerate(sliceObj.boundingRects):
                sliceObj.objects[rid].copyTo(contourCanvas[rect.y:rect.y + rect.height, rect.x:rect.x + rect.width])

            cv2.imshow(name, contourCanvas)
            cv2.waitKey(0)

        cv2.destroyWindow(name)
