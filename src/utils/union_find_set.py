class UnionFindSet:
    def __init__(self, size):
        """
        构造函数，初始化并查集，指定元素数量。

        Parameters:
        - size (int): 元素数量。
        """
        self.size = size
        self.f = []
        self.init()

    def __del__(self):
        """
        析构函数，释放动态分配的内存。
        """
        del self.f

    def find(self, x):
        """
        查找元素所属的集合。

        Parameters:
        - x (int): 要查找的元素。

        Returns:
        - int: 元素所属的集合。
        """
        return x if x == self.f[x] else self.find(self.f[x])

    def merge(self, x, y):
        """
        合并两个集合。

        Parameters:
        - x (int): 第一个要合并的元素。
        - y (int): 第二个要合并的元素。
        """
        x, y = self.find(x), self.find(y)
        if x != y:
            self.f[x] = y

    def init(self):
        """
        初始化并查集，创建各自独立的集合。
        """
        self.f = [i for i in range(self.size)]

    def reset(self):
        """
        重置并查集，使每个元素独立为一个集合。
        """
        self.init()

    def get_size(self):
        """
        获取元素数量。

        Returns:
        - int: 元素数量。
        """
        return self.size
