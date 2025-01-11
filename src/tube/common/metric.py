import cv2
import numpy as np

class Metric:
    def __init__(self):
        self.frameWidth = 0
        self.frameHeight = 0
        self.oriTotFrame = 0
        self.resTotFrame = 0
        self.totTubes = 0
        self.unionAreaOfAllTubes = 0
        self.intersectingAreaOfAllTubes = 0
        self.totTime = 0
        self.cd_fenzi = 0
        self.cd_fenmu = 0

    def reset(self):
        self.resTotFrame = 0
        self.toTtubes = 0
        self.unionAreaOfAllTubes = 0
        self.intersectingAreaOfAllTubes = 0
        self.totTime = 0
        self.cd_fenmu = 0
        self.cd_fenzi = 0

    def print(self):
        print(f"原先视频总帧：{self.oriTotFrame}\t结果视频总帧：{self.resTotFrame}\t总管道数：{self.totTubes}")
        print(f"新旧总帧比值：{1.0 * self.resTotFrame / self.oriTotFrame}")

        print(f"self.resTotFrame: {self.resTotFrame}")
        print(f"self.frameWidth: {self.frameWidth}")
        print(f"self.frameHeight: {self.frameHeight}")
        print(f"所有视频相交面积/所有帧的面积：{1.0 * self.intersectingAreaOfAllTubes / (1.0 * self.resTotFrame * self.frameWidth * self.frameHeight)}")

        print(f"所有视频相交面积 比 所有视频的并集面积：{1.0 * self.intersectingAreaOfAllTubes / self.unionAreaOfAllTubes}")
        print(f"CD：{1.0 * self.cd_fenzi / self.cd_fenmu}")
        tot_time_seconds = 1.0 * self.totTime / cv2.getTickFrequency()
        print(f"程序运行时间：{tot_time_seconds} 秒")
    def update(self, tube_buffer, background, upper_limit=np.inf):
        frame_num = 0
        for tid in range(len(tube_buffer)):
            frame_num = max(frame_num, tube_buffer[tid].finalPlace + tube_buffer[tid].getLength())
        frame_num = min(frame_num, upper_limit)

        masks = [np.zeros((background.shape[0], background.shape[1]), dtype=np.uint8) for _ in range(frame_num)]

        for tid in range(len(tube_buffer)):
            lower = tube_buffer[tid].finalPlace
            upper = min(lower + tube_buffer[tid].getLength(), frame_num)

            for fid in range(lower, upper):
                cur_slice = tube_buffer[tid].frames[fid - lower]
                for sid in range(cur_slice.getObjNumber()):
                    rect = cur_slice.boundingRects[sid]
                    width, height = rect[2], rect[3]
                    self.unionAreaOfAllTubes += width * height
                    masks[fid][rect[1]:rect[1] + height, rect[0]:rect[0] + width] = 1

        for mask in masks:
            self.intersectingAreaOfAllTubes += np.sum(mask)

        self.resTotFrame += len(masks)

        for i in range(len(tube_buffer)):
            max_slice_number = max([frame.getObjNumber() for frame in tube_buffer[i].frames], default=0)
            self.cd_fenmu += max_slice_number

        for i in range(len(tube_buffer)):
            for j in range(len(tube_buffer)):
                if (tube_buffer[i].startFrame - tube_buffer[j].startFrame) * (
                        tube_buffer[i].finalPlace - tube_buffer[j].finalPlace) < 0:
                    self.cd_fenzi += 1


