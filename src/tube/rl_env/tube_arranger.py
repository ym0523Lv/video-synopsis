import numpy as np
import gym
from gym import spaces

class PipeRearrangeEnv(gym.Env):
    def __init__(self, contour_finder):
        super(PipeRearrangeEnv, self).__init__()

        # 关联 ContourFinder 类实例，用于获取管道的质心信息
        self.contour_finder = contour_finder

        # 定义状态空间：管道的质心 (mass centers)，假设每个质心有2D坐标
        if len(self.contour_finder.massCenters) == 0:
            #print(self.contour_finder.massCenters)
            raise ValueError("ContourFinder has no massCenters. Please check the contour detection process.")

        # 状态空间根据质心的数量定义，shape=(n, 2)，n是管道的数量
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(contour_finder.massCenters), 2), dtype=np.float32)

        # 动作空间：+1 或 -1 表示左右移动时间轴位置
        self.action_space = spaces.Discrete(2)  # 0: -1 (左移), 1: +1 (右移)

        # 环境的一些参数
        self.max_time_step = 360
        self.current_time_step = 0

        # 设置碰撞阈值（两个质心之间的最小距离）
        self.collision_threshold = min(640, 360) * 0.05

    def reset(self):
        """重置环境，返回初始状态"""
        # 重置 ContourFinder 中的质心位置
        self.contour_finder.reset_contours()

        # 重置时间步
        self.current_time_step = 0

        # 返回初始状态（质心位置）
        return np.array(self.contour_finder.massCenters)

    def step(self, action):
        """根据动作调整时间轴，返回新的状态、奖励和是否完成"""
        # 根据动作调整时间轴，左移或右移
        if action == 0:
            self.contour_finder.shift(-1)  # 左移
        else:
            self.contour_finder.shift(1)   # 右移

        # 获取新的质心位置（状态）
        new_state = np.array(self.contour_finder.massCenters)

        # 计算奖励
        reward = self.calculate_reward(new_state)

        # 增加时间步
        self.current_time_step += 1

        # 判断是否完成（条件：没有碰撞或达到最大时间步）
        done = self.check_done(new_state) or (self.current_time_step >= self.max_time_step)

        return new_state, reward, done, {}

    def calculate_reward(self, state):
        """根据质心之间的距离和碰撞情况计算奖励"""
        reward = 0

        # 根据质心间的距离计算奖励，避免碰撞
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                dist = np.linalg.norm(state[i] - state[j])

                if dist < self.collision_threshold:
                    reward -= 5  # 碰撞惩罚
                else:
                    reward += dist  # 距离越远，奖励越高

        return reward

    def check_done(self, state):
        """判断是否完成，条件：如果有碰撞则为 False"""
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                dist = np.linalg.norm(state[i] - state[j])
                if dist < self.collision_threshold:
                    return False  # 有碰撞，未完成
        return True  # 无碰撞，任务完成
