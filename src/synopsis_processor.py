import cv2
import sys
import numpy as np
import torch
from src.tracking.contour_finder import ContourFinder
from src.tube.common.metric import Metric
from src.tube.common.tube_generator import TubeGenerator
from src.tube.common.tube_stitcher import TubeStitcher

#from src.tube.dgcotr_new.tube_arranger import TubeArranger
#from src.tube.GNN_Lv.tube_arranger import TubeArranger
from src.tube.rl_env.tube_arranger import PipeRearrangeEnv
from stable_baselines3 import PPO

from src.utils.draw_utils import DrawUtils
import time

'''视频处理的大类'''
class SynopsisProcessor:
    def __init__(self, synopsisVideoName):
        # 创建TubeGenerator、Metric和TubeStitcher实例
        self.tubeGenerator = TubeGenerator()
        self.metric = Metric()
        self.tubeStitcher = TubeStitcher(synopsisVideoName)
        # 创建ContourFinder实例
        self.contourFinder = ContourFinder()
        # 背景矩阵
        self.background = None

    def __del__(self):
        if self.tubeGenerator is not None:
            del self.tubeGenerator
        if self.metric is not None:
            del self.metric
        if self.tubeStitcher is not None:
            del self.tubeStitcher

    def getTubes(self, videoPath, forePath, tubeSavePath):
        decoder = cv2.VideoCapture(videoPath)
        foreVideo = cv2.VideoCapture(forePath)

        if decoder.isOpened() and foreVideo.isOpened():
            self.metric.oriTotFrame = decoder.get(cv2.CAP_PROP_FRAME_COUNT)
            self.metric.frameWidth = decoder.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.metric.frameHeight = decoder.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.tubeStitcher.setFPS(decoder.get(cv2.CAP_PROP_FPS))

            self.background = np.zeros(
                (int(self.metric.frameHeight), int(self.metric.frameWidth), 3), dtype=np.float32
            )
        else:
            print("Video loading failed.")
            sys.exit(1)

        print("Start getting tubes ...")
        contours = []
        hierarchy = []
        frameNumber = 0

        ccnt = 0
        while True:
            ret_frame, frame = decoder.read()
            ret_foreground, foreground = foreVideo.read()

            if not ret_frame or not ret_foreground:
                print("Video reading failed.")
                break

            ccnt += 1
            #if ccnt<78:
            #    continue

            print(f"{ccnt}/{self.metric.oriTotFrame}")

            segmentationMap = cv2.cvtColor(foreground, cv2.COLOR_RGB2GRAY)
            mc = []
            boundRect = []
            contours, mc, boundRect = self.contourFinder.findContours(frame, segmentationMap, hierarchy, contours, mc, boundRect)

            #cv2.drawContours(frame,contours,-1,(0,255,0),3)
            #cv2.imshow("",foreground)
            #cv2.waitKey(0)
            self.tubeGenerator.assignObjects(frame, boundRect, contours, frameNumber)
            # Update background.
            fframe = frame.astype(np.float32)
            self.background = (self.background * frameNumber + fframe) / (frameNumber + 1)
            frameNumber += 1
        print("Sorting tubes ...")
        self.tubeGenerator.sortTubes()
        print("Saving tubes ...")
        self.tubeGenerator.saveTubes(f"./tube_record/{tubeSavePath}")
    '''
    图着色
    '''
    '''
    def startRearranging(self, curBack, maxNodeNumber, tolerateTh):
        print("================ Rearranging Stage =================")
        self.metric.reset()

        tubeArranger = TubeArranger(maxNodeNumber, tolerateTh)

        st = time.perf_counter()

        for i in range(len(self.tubeGenerator.tubeBuffer)):
            print(f"Processing tube {i + 1}/{len(self.tubeGenerator.tubeBuffer)}")
            tube = self.tubeGenerator.tubeBuffer[i]
            if tube.getLength() <= 10:
                #print(f"Tube {i + 1} skipped: length <= 10")
                continue
            #print(f"Processing tube {i + 1}")
            self.metric.totTubes += 1
            tubeArranger.tubeRearranging(tube)

        ed = time.perf_counter()

        self.tubeStitcher.tubeStitchingLowMemoryCost(tubeArranger.tubeBuffer, curBack)
        self.metric.update(tubeArranger.tubeBuffer, curBack)

        print(f"最大节点数 {maxNodeNumber}\t阈值 {tolerateTh}\t")
        self.metric.totTime += ed - st

        print(f"self.metric: {self.metric}")
        print(f"self.metric.resTotFrame: {self.metric.resTotFrame}")
        print(f"self.metric.frameWidth: {self.metric.frameWidth}")
        print(f"self.metric.frameHeight: {self.metric.frameHeight}")
        self.metric.print()

        del tubeArranger
    '''

    '''
    def startRearranging(self, curBack, maxNodeNumber, tolerateTh):
        """
        使用 GAT 模型对管道进行重排，并在最终阶段进行拼接。
        Parameters:
        - curBack: 当前背景
        - maxNodeNumber: GNN 的最大节点数
        - tolerateTh: 容忍度阈值，用于空间冲突判断
        """
        print("================ Rearranging Stage =================")
        self.metric.reset()

        # 初始化 TubeArranger，传入输入维度、输出维度等参数
        input_dim = 128  # 假设输入特征维度为 128
        output_dim = 128  # 假设输出特征维度为 128
        tubeArranger = TubeArranger(input_dim=input_dim, output_dim=output_dim)

        st = time.perf_counter()

        # 遍历所有管道，执行重排
        for i in range(len(self.tubeGenerator.tubeBuffer)):
            print(f"Processing tube {i + 1}/{len(self.tubeGenerator.tubeBuffer)}")
            tube = self.tubeGenerator.tubeBuffer[i]
            
            # 更新计量器
            self.metric.totTubes += 1

            # 构建特征矩阵和边列表
            node_features = torch.randn(tube.getLength(), input_dim)  # 随机生成特征向量
            edge_index = tubeArranger.build_edge_index(node_features, threshold=tolerateTh)  # 构建边

            # 检查 edge_index 是否为空
            if edge_index.numel() == 0:
                print(f"Warning: No edges were created for tube {i + 1}. Skipping...")
                continue

            # 使用 GAT 模型进行管道重排
            tubeArranger.tubeRearranging(tube, edge_index, node_features)
            print("rearranging")

        ed = time.perf_counter()

        # 在重排完成后，进行拼接操作
        self.tubeStitcher.tubeStitchingLowMemoryCost(tubeArranger.tubeBuffer, curBack)

        # 更新度量器
        self.metric.update(tubeArranger.tubeBuffer, curBack)

        print(f"最大节点数 {maxNodeNumber}\t阈值 {tolerateTh}")
        self.metric.totTime += ed - st

        # 输出度量器信息
        print(f"self.metric: {self.metric}")
        print(f"self.metric.resTotFrame: {self.metric.resTotFrame}")
        print(f"self.metric.frameWidth: {self.metric.frameWidth}")
        print(f"self.metric.frameHeight: {self.metric.frameHeight}")
        self.metric.print()

    '''

    def startRearranging(self, curBack):
        """
        使用强化学习对管道进行重排。
        Parameters:
        - curBack: 当前背景
        """
        print("================ Rearranging Stage =================")
        self.metric.reset()

        # 创建强化学习环境实例
        # print(len(self.contourFinder.massCenters))
        env = PipeRearrangeEnv(self.contourFinder)  # 使用 ContourFinder 实例

        # 创建 PPO 代理
        model = PPO("MlpPolicy", env, verbose=1)

        # 训练代理
        model.learn(total_timesteps=10000)  # 根据需要调整时间步长

        st = time.perf_counter()

        # 遍历所有管道，执行重排
        for i in range(len(self.tubeGenerator.tubeBuffer)):
            print(f"Processing tube {i + 1}/{len(self.tubeGenerator.tubeBuffer)}")
            tube = self.tubeGenerator.tubeBuffer[i]

            # 更新计量器
            self.metric.totTubes += 1

            # 获取管道的初始状态
            state = env.reset()  # 重置环境

            done = False
            while not done:
                # 代理根据当前状态选择动作
                action, _ = model.predict(state)

                # 执行动作，获得新状态和奖励
                state, reward, done, _ = env.step(action)

        ed = time.perf_counter()

        # 在重排完成后，进行拼接操作
        self.tubeStitcher.tubeStitchingLowMemoryCost(self.tubeGenerator.tubeBuffer, curBack)

        # 更新度量器
        self.metric.update(self.tubeGenerator.tubeBuffer, curBack)

        print(f"完成重排，最大节点数 {1000}，阈值 {10}")
        self.metric.totTime += ed - st

        # 输出度量器信息
        print(f"self.metric: {self.metric}")
        print(f"self.metric.resTotFrame: {self.metric.resTotFrame}")
        print(f"self.metric.frameWidth: {self.metric.frameWidth}")
        print(f"self.metric.frameHeight: {self.metric.frameHeight}")
        self.metric.print()

    def rearranging(self):
        curBack = self.background.astype(np.uint8)
        self.startRearranging(curBack)

