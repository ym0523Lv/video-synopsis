class VideoBasic:
    @staticmethod
    def frameToTime(frameIndex, fps):
        # 计算总秒数
        totSeconds = frameIndex // fps
        # 计算秒数
        seconds = totSeconds % 60
        # 计算分钟数
        totSeconds //= 60
        minutes = totSeconds % 60
        # 计算小时数
        totSeconds //= 60
        hours = totSeconds

        # 将时间格式化为字符串
        timeString = "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))

        # 返回时间字符串
        return timeString
