import cv2
import numpy as np

class DrawUtils:
    @staticmethod
    def drawCross(img, center, color, diameter):
        """
        在图像上绘制一个十字。

        Parameters:
        - img: 图像
        - center: 十字的中心坐标
        - color: RGB 颜色
        - diameter: 十字的直径
        """
        p1 = (center[0] - diameter, center[1] - diameter)
        p2 = (center[0] + diameter, center[1] + diameter)
        cv2.line(img, p1, p2, color, 2, cv2.LINE_AA, 0)

        p3 = (center[0] + diameter, center[1] - diameter)
        p4 = (center[0] - diameter, center[1] + diameter)
        cv2.line(img, p3, p4, color, 2, cv2.LINE_AA, 0)

    @staticmethod
    def drawBoundingRect(img, boundingRect):
        """
        在图像上绘制边界矩形，并返回矩形的中心坐标。

        Parameters:
        - img: 图像
        - boundingRect: 边界矩形
        Returns: 矩形的中心坐标
        """
        cv2.rectangle(img, boundingRect.tl(), boundingRect.br(), (0, 255, 0), 2, 8, 0)
        center = (boundingRect.x + (boundingRect.width // 2), boundingRect.y + (boundingRect.height // 2))
        cv2.circle(img, center, 3, (0, 0, 255), -1, 1, 0)
        return center

    @staticmethod
    def drawTrajectory(img, trajectory, color):
        """
        在图像上绘制轨迹。

        Parameters:
        - img: 图像
        - trajectory: 轨迹的点集
        - color: 颜色
        """
        if len(trajectory) < 2:
            return
        for i in range(len(trajectory) - 1):
            cv2.line(img, trajectory[i], trajectory[i + 1], color, 1, cv2.LINE_AA, 0)

    @staticmethod
    def contourShow(drawingName, contours, boundingRect, imgSize):
        """
        在新图像中绘制轮廓并显示。

        Parameters:
        - drawingName: 显示窗口的名称
        - contours: 轮廓点集的集合
        - boundingRect: 边界矩形的集合
        - imgSize: 图像的大小
        """
        drawing = np.zeros(imgSize, dtype=np.uint8)
        for i in range(len(contours)):
            cv2.drawContours(drawing, contours, i, (127, 127, 127), cv2.FILLED, 8, [], 0, (0, 0))
            DrawUtils.drawBoundingRect(drawing, boundingRect[i])
        cv2.imshow(drawingName, drawing)
