import traceback

import cv2
import sys
import numpy as np
from src.utils.video import VideoBasic

class TubeStitcher:
    def __init__(self, resultVideoName, fpsOfOriginal=30):
        """
        初始化 TubeStitcher 对象。

        参数:
        - resultVideoName: 输出视频的文件名
        - fpsOfOriginal: 原始视频的帧率，默认为30帧/秒
        """
        self.resultVideoName = resultVideoName
        self.fpsOfOriginal = fpsOfOriginal
        self.resultWriter = None

    def __del__(self):
        """
        析构函数，用于释放视频写入器资源。
        """
        if self.resultWriter is not None and self.resultWriter.isOpened():
            self.resultWriter.release()

    def poissonImageEditing(self, obj, contour, background):
        """
        使用 Poisson Image Editing 算法将目标对象粘贴到背景中。

        参数:
        - obj: 目标对象的图像
        - contour: 目标对象的轮廓
        - background: 背景图像
        """
        print("obj shape:", obj.shape)
        print("background shape:", background.shape)

        # 创建与目标对象相同大小的单通道灰度图像作为掩码
        foreMask = np.zeros(obj.shape[:2], dtype=np.uint8)

        print("foreMask shape:", foreMask.shape)

        # 计算轮廓相对于边界框的偏移量
        bbox = cv2.boundingRect(contour)
        ori = (bbox[0], bbox[1])
        relativeContour = [c - ori for c in contour]

        polygons = np.array([relativeContour], dtype=np.int32)
        cv2.fillPoly(foreMask, polygons, 255)

        mu = cv2.moments(contour, False)
        output = background.copy()

        center_x = int(bbox[0] + bbox[2] / 2.0 + 0.5)
        center_y = int(bbox[1] + bbox[3] / 2.0 + 0.5)
        print(f"(center_x, center_y){(center_x, center_y)}")

        vis = background.copy()
        print("Contour shape:", contour.shape)
        '''# Check if contours and object are not empty before drawing
        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)  # Draw contour
        # Display the result
        cv2.imshow("Poisson Image Editing Visualization", vis)
        cv2.waitKey(0)
        # 显示 obj 图像
        cv2.imshow("Object Image", obj)
        cv2.waitKey(0)'''

        try:
            cv2.seamlessClone(obj, background, foreMask, (center_x, center_y), cv2.MIXED_CLONE , output)
            print()
        except Exception as e:
            print("TubeStitcher.poissonImageEditing.seamlessClone Error:", str(e))
            print("Exception traceback:", traceback.format_exc())

        return output

    def tubeStitching(self, tubeBuffer, background, upperLimit=sys.maxsize):
        """
        管道拼接，将各个运动目标按照管道信息拼接到背景中。

        参数:
        - tubeBuffer: Tube 对象的列表，包含运动目标的管道信息
        - background: 背景图像
        - upperLimit: 输出视频的最大帧数，默认为系统最大整数值
        """
        frameNum = 0
        for tid in range(len(tubeBuffer)):
            frameNum = max(frameNum, tubeBuffer[tid].finalPlace + tubeBuffer[tid].getLength())
        frameNum = min(frameNum, upperLimit)

        outputFrames = [background.copy() for _ in range(frameNum)]

        for tid in range(len(tubeBuffer)):
            lower = tubeBuffer[tid].finalPlace
            tube = tubeBuffer[tid]
            upper = min(lower + tube.getLength(), frameNum)

            for fid in range(lower, upper):
                curSlice = tube.frames[fid - lower]
                for sid in range(curSlice.getObjNumber()):
                    rect = curSlice.boundingRects[sid]
                    obj = curSlice.objects[sid]
                    self.poissonImageEditing(obj, curSlice.contours[sid], outputFrames[fid])
                    cv2.putText(
                        outputFrames[fid],
                        VideoBasic.frameToTime(tube.startFrame + fid - lower, self.fpsOfOriginal),
                        (rect[0], rect[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, 8
                    )

        if self.resultWriter is None or not self.resultWriter.isOpened():
            self.resultWriter = cv2.VideoWriter(
                self.resultVideoName,
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30.0,
                (background.shape[1], background.shape[0])
            )

        for i in range(len(outputFrames)):
            self.resultWriter.write(outputFrames[i])

    def tubeStitchingLowMemoryCost(self, tubeBuffer, background, upperLimit=sys.maxsize):
        """
        低内存消耗的管道拼接，通过排序确保早期的管道叠加在晚期的管道之上。

        参数:
        - tubeBuffer: Tube 对象的列表，包含运动目标的管道信息
        - background: 背景图像
        - upperLimit: 输出视频的最大帧数，默认为系统最大整数值
        """
        frameNum = 0
        for tid in range(len(tubeBuffer)):
            frameNum = max(frameNum, tubeBuffer[tid].finalPlace + tubeBuffer[tid].getLength())
        frameNum = min(frameNum, upperLimit)

        # 按照finalPlace从晚到早排序，确保早期的管道叠加在晚期的管道之上
        tubeBuffer.sort(key=lambda x: x.finalPlace, reverse=True)

        if self.resultWriter is None or not self.resultWriter.isOpened():
            self.resultWriter = cv2.VideoWriter(
                self.resultVideoName,
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30.0,
                (background.shape[1], background.shape[0])
            )

        for curOutIdx in range(frameNum):
            outputFrame = background.copy()

            for tid in range(len(tubeBuffer)):
                tube = tubeBuffer[tid]
                if curOutIdx >= tube.finalPlace and curOutIdx < tube.finalPlace + tube.getLength():
                    relativeIdx = curOutIdx - tube.finalPlace
                    curSlice = tube.frames[relativeIdx]
                    for sid in range(curSlice.getObjNumber()):
                        rect = curSlice.boundingRects[sid]
                        obj = curSlice.objects[sid]
                        outputFrame = self.poissonImageEditing(obj, curSlice.contours[sid], outputFrame)
                        cv2.putText(
                            outputFrame,
                            VideoBasic.frameToTime(tube.startFrame + relativeIdx, self.fpsOfOriginal),
                            (rect[0], rect[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, 8
                        )
                        '''cv2.imshow("outputFrame Image", outputFrame)
                        cv2.waitKey(0)'''

            self.resultWriter.write(outputFrame)

        self.resultWriter.release()

    def setFPS(self, fps):
        """
        设置原始视频的帧率。

        参数:
        - fps: 原始视频的帧率
        """
        self.fpsOfOriginal = fps
