class DisjointSets:
    class Node:
        def __init__(self):
            self.rank = 0  # 大致表示节点在其子树中的最大高度
            self.index = 0  # 节点代表的元素的索引
            self.parent = None  # 节点的父节点

    def __init__(self, count=0):
        # 创建空的不相交集数据结构
        self.m_numElements = 0
        self.m_numSets = 0
        self.m_nodes = []  # 代表元素的节点列表
        self.addElements(count)

    def __del__(self):
        # 析构函数，释放节点内存
        for node in self.m_nodes:
            del node
        self.m_nodes.clear()
        self.m_numElements = 0
        self.m_numSets = 0

    def findSet(self, elementId):
        # 查找元素当前所属的集合标识符。
        # 注意：尽管这个方法是常量的，但为了优化，一些内部数据被修改。
        assert elementId < self.m_numElements

        # 查找代表元素所在集合的根节点
        curNode = self.m_nodes[elementId]
        while curNode.parent is not None:
            curNode = curNode.parent
        root = curNode

        # 路径压缩：将 elementId 到根节点的路径上的所有节点的父节点直接设为根节点，
        # 优化了后续 findSet 调用的效率
        curNode = self.m_nodes[elementId]
        while curNode != root:
            nextNode = curNode.parent
            curNode.parent = root
            curNode = nextNode

        return root.index

    def union(self, setId1, setId2):
        # 合并两个集合
        assert setId1 < self.m_numElements
        assert setId2 < self.m_numElements

        if setId1 == setId2:
            return  # 已经合并

        node1 = self.m_nodes[setId1]
        node2 = self.m_nodes[setId2]

        # 确定具有更高 rank 的表示集合的节点。
        # 具有更高 rank 的节点可能有一个更大的子树，
        # 为了更好地平衡表示合并的树，具有更高 rank 的节点将是具有较低 rank 的节点的父节点。
        if node1.rank > node2.rank:
            node2.parent = node1
        elif node1.rank < node2.rank:
            node1.parent = node2
        else:  # node1.rank == node2.rank
            node2.parent = node1
            node1.rank += 1  # 更新 rank

        # 由于两个集合已经合并为一个，因此集合数量减少一个
        self.m_numSets -= 1

    def addElements(self, numToAdd):
        # 向不相交集数据结构中添加指定数量的元素
        assert numToAdd >= 0

        # 将指定数量的元素节点插入并初始化到 m_nodes 数组的末尾
        self.m_nodes.extend([None] * numToAdd)
        for i in range(self.m_numElements, self.m_numElements + numToAdd):
            self.m_nodes[i] = self.Node()
            self.m_nodes[i].parent = None
            self.m_nodes[i].index = i
            self.m_nodes[i].rank = 0

        # 更新元素和集合计数
        self.m_numElements += numToAdd
        self.m_numSets += numToAdd

    def numElements(self):
        # 返回当前不相交集数据结构中的元素数量
        return self.m_numElements

    def numSets(self):
        # 返回当前不相交集数据结构中的集合数量
        return self.m_numSets
