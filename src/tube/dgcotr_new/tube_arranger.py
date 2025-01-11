import copy
import cv2
from src.tube.common.tube_base import Tube, Slice
from src.tube.common.metric import Metric

class SubNode:
    def __init__(self, gId, belongedTube, sliceId):
        """
        SubNode 类表示某个管道的帧。

        Parameters:
        - gId (int): 动态图中的节点大小。
        - belongedTube (int): 帧属于哪个管道。
        - sliceId (int): 帧在管道的位置。
        """
        self.gId = gId
        self.belongedTube = belongedTube
        self.sliceId = sliceId
        self.color = -1

class MainNode:
    def __init__(self, gId):
        """
        MainNode 类表示一个管道。

        Parameters:
        - gId (int): 索引。
        """
        self.isDeleted = False
        self.gId = gId
        self.color = 0
        self.subNodes = []
        self.neighborList = []

    def __getitem__(self, index):
        """
        重载 [] 运算符，获取管道中的帧。

        Parameters:
        - index (int): 索引。

        Returns:
        - SubNode: 管道中的帧。
        """
        assert index < len(self.subNodes)
        return self.subNodes[index]

    def addSubNode(self, subNode):
        """
        往管道中添加帧。

        Parameters:
        - subNode (SubNode): 要添加的帧。
        """
        self.subNodes.append(subNode)

    def getSubNodeNumber(self):
        """
        获取管道中帧的数量。

        Returns:
        - int: 管道中帧的数量。
        """
        return len(self.subNodes)

class DynamicGraph:
    def __init__(self, tolerate=0):
        """
        DynamicGraph 类表示动态图结构。

        Parameters:
        - tolerate (int): 容忍度阈值。
        """
        self.TOLERATE_TH = tolerate
        self.curMinColor = 0
        self.timeChecker = [0, 0]
        self.mainNodes = []
        self.subNodes = []
        self.edges = {}

    def __getitem__(self, key):
        """
        重载 [] 运算符，获取边缘信息。

        Parameters:
        - key: 边缘的键。

        Returns:
        - List: 边缘信息列表。
        """
        assert key in self.edges, "key 不在 edges 里"
        return self.edges[key]

    def getSize(self):
        """
        获取图的管道数量。

        Returns:
        - int: 图的管道数量。
        """
        return len(self.mainNodes)

    def addNodes(self, newTube, placeTubes):
        """
        向图中添加节点。

        Parameters:
        - newTube (Tube): 要添加的管道。
        - placeTubes (List[Tube]): 已有的管道列表。
        """
        newTube.id = len(placeTubes)
        self.mainNodes.append(MainNode(newTube.id))

        newMainNode = self.mainNodes[newTube.id]
        for i in range(newTube.getLength()):
            self.edges[len(self.subNodes)] = []
            newMainNode.addSubNode(len(self.subNodes))
            self.subNodes.append(SubNode(len(self.subNodes), newTube.id, i))

        for mnid in range(len(self.mainNodes) - 1):
            if not self.mainNodes[mnid].isDeleted:
                placeTube = placeTubes[mnid]
                if self.buildEdge(newTube, placeTube) == True:
                    self.mainNodes[mnid].neighborList.append(newTube.id)
                    newMainNode.neighborList.append(mnid)

    def chooseAbandonedNode(self):
        """
        选择可抛弃的节点。

        Returns:
        - int: 可抛弃的节点 ID。
        """
        return self.subNodes[list(self.edges.keys())[0]].belongedTube

    def deleteNode(self, mainNodeId):
        """
        删除节点。

        Parameters:
        - mainNodeId (int): 要删除的节点 ID。
        """
        mainNode = self.mainNodes[mainNodeId]
        mainNode.isDeleted = True
        for i in range(mainNode.getSubNodeNumber()):
            del self.edges[mainNode.subNodes[i]]

        for _, neighborList in self.edges.items():
            for i in reversed(range(len(neighborList))):
                if self.subNodes[neighborList[i]].belongedTube == mainNodeId:
                    del neighborList[i]

    def getMinUsableColor(self, mainNodeId):
        """
        返回最小可用颜色。

        Parameters:
        - mainNodeId (int): 主节点 ID。

        Returns:
        - int: 最小可用颜色。
        """
        usableColor = [0] * 50000
        mainNode = self.mainNodes[mainNodeId]
        for snid in range(mainNode.getSubNodeNumber()):
            subNodeId = mainNode[snid]
            neighbors = self.edges[subNodeId]
            for neiId in range(len(neighbors)):
                cantUseColor = self.subNodes[neighbors[neiId]].color
                if cantUseColor < 0:
                    continue
                relativeCanUseColor = cantUseColor - snid
                if relativeCanUseColor >= 0:
                    usableColor[relativeCanUseColor] += 1
        minUsableColor = self.curMinColor
        while minUsableColor < 50000:
            if usableColor[minUsableColor] <= self.TOLERATE_TH:
                break
            minUsableColor += 1
        return minUsableColor

    def coloring(self, mainNodeId, color):
        """
        给这个节点着色。

        Parameters:
        - mainNodeId (int): 主节点 ID。
        - color (int): 要着的颜色。
        """
        mainNode = self.mainNodes[mainNodeId]
        for i in range(mainNode.getSubNodeNumber()):
            self.subNodes[mainNode[i]].color = color + i

    def buildEdge(self, newTube, otherTube):
        """
        建立边。

        Parameters:
        - newTube (Tube): 新的管道。
        - otherTube (Tube): 已有的管道。

        Returns:
        - bool: 是否成功建立边。
        """
        nodeNew = self.mainNodes[newTube.id]
        nodeOther = self.mainNodes[otherTube.id]

        ok = False
        for i in range(nodeNew.getSubNodeNumber()):
            for j in range(nodeOther.getSubNodeNumber()):
                isOverlap = newTube.frames[i].isOverlap(otherTube.frames[j])
                if isOverlap == True:
                    subId1 = nodeNew[i]
                    subId2 = nodeOther[j]
                    self.edges[subId1].append(subId2)
                    self.edges[subId2].append(subId1)
                    ok = True
        return ok

    def recoloring(self, mainNodeId, tubeBuffer):
        """
        重新着色。

        Parameters:
        - mainNodeId (int): 主节点 ID。
        - tubeBuffer (List[Tube]): 管道列表。
        """
        newMainNode = self.mainNodes[mainNodeId]
        minStart = tubeBuffer[newMainNode.gId].finalPlace
        maxColor = tubeBuffer[newMainNode.gId].finalPlace + tubeBuffer[newMainNode.gId].getLength()
        for i in range(len(newMainNode.neighborList)):
            neiTube = tubeBuffer[newMainNode.neighborList[i]]
            minStart = min(minStart, neiTube.finalPlace)
            maxColor = max(maxColor, neiTube.finalPlace + neiTube.getLength())

        neighborColors = [tubeBuffer[newMainNode.gId].finalPlace] + \
                         [tubeBuffer[newMainNode.neighborList[i]].finalPlace for i in range(len(newMainNode.neighborList))]
        tubeBuffer[mainNodeId].finalPlace = minStart
        self.coloring(mainNodeId, tubeBuffer[mainNodeId].finalPlace)

        for i in range(len(newMainNode.neighborList)):
            tubeBuffer[newMainNode.neighborList[i]].finalPlace = -1
            mainNode = self.mainNodes[newMainNode.neighborList[i]]
            mainNode.color = -1
            for j in range(mainNode.getSubNodeNumber()):
                self.subNodes[mainNode.subNodes[j]].color = -1

        order = self.makeOrder(newMainNode.neighborList)
        tmpMaxColor = self.curMinColor + tubeBuffer[mainNodeId].getLength()

        for i in range(len(order)):
            neighborId = newMainNode.neighborList[order[i]]
            tubeBuffer[neighborId].finalPlace = self.getMinUsableColor(neighborId)
            self.coloring(neighborId, tubeBuffer[neighborId].finalPlace)
            tmpMaxColor = max(tmpMaxColor, tubeBuffer[neighborId].finalPlace + tubeBuffer[neighborId].getLength())

        if tmpMaxColor > maxColor:
            tubeBuffer[newMainNode.gId].finalPlace = neighborColors[0]
            self.coloring(newMainNode.gId, neighborColors[0])
            for i in range(len(newMainNode.neighborList)):
                tubeBuffer[newMainNode.neighborList[i]].finalPlace = neighborColors[i + 1]
                self.coloring(newMainNode.neighborList[i], neighborColors[i + 1])

    def makeOrder(self, neighborList):
        """
        生成排序。

        Parameters:
        - neighborList (List[int]): 邻居节点列表。

        Returns:
        - List[int]: 排序后的节点列表。
        """
        return list(range(len(neighborList)))

    def updateTubeBuffer(self, tubeBuffer, MAX_NODE_NUMBER):
        """
        更新管道缓存。

        Parameters:
        - tubeBuffer (List[Tube]): 管道列表。
        - MAX_NODE_NUMBER (int): 最大管道数量。
        """
        if len(self.mainNodes) >= MAX_NODE_NUMBER:
            mainNodeId = self.chooseAbandonedNode()
            self.curMinColor = max(self.curMinColor, tubeBuffer[mainNodeId].finalPlace)
            self.deleteNode(mainNodeId)
'''动态图着色'''

class TubeArranger:
    def __init__(self, maxNodeNumber=700, tolerateTh=0):
        """
        TubeArranger 类用于管道重排。

        Parameters:
        - maxNodeNumber (int): 最大管道数量。
        - tolerateTh (int): 容忍度阈值。
        """
        self.MAX_NODE_NUMBER = maxNodeNumber
        self.dynamicGraph = DynamicGraph(tolerateTh)
        self.tubeBuffer = []

    def tubeRearranging(self, tube):
        """
        管道重排。

        Parameters:
        - tube (Tube): 要进行重排的管道。
        """
        self.dynamicGraph.addNodes(tube, self.tubeBuffer)
        self.tubeBuffer.append(copy.deepcopy(tube))
        minUsableColor = self.dynamicGraph.getMinUsableColor(tube.id)
        self.tubeBuffer[-1].finalPlace = minUsableColor
        self.dynamicGraph.coloring(tube.id, minUsableColor)
        self.dynamicGraph.updateTubeBuffer(self.tubeBuffer, self.MAX_NODE_NUMBER)

    def saveTubes(self, fileName):
        with open(fileName, "w") as f:
            f.write(f"{len(self.tubeBuffer)}\n")
            for tid in range(len(self.tubeBuffer)):
                tube = self.tubeBuffer[tid]
                f.write(f"{tube.startFrame} {tube.finalPlace} {tube.getLength()}\n")

                for sid in range(tube.getLength()):
                    slice = tube.frames[sid]
                    f.write(f"{slice.getObjNumber()}\n")

                    for oid in range(slice.getObjNumber()):
                        rect = slice.boundingRects[oid]
                        f.write(f"{rect.x} {rect.y} {rect.width} {rect.height}\n")

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
'''

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

'''
'''GNN'''
'''
class TubeArranger:
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        self.tubeBuffer = []
        self.gnn_model = GNNModel(input_dim, output_dim, hidden_dim)

    def tubeRearranging(self, tube, edge_index, node_features):
        new_main_node = MainNode(len(self.tubeBuffer))
        self.tubeBuffer.append(new_main_node)

        # 使用 GNN 模型对管道进行排序
        node_features = self.gnn_model(node_features, edge_index)

        # 更新管道中的子节点信息
        for i in range(tube.getLength()):
            new_sub_node = SubNode(len(self.tubeBuffer), new_main_node.gId, i)
            new_main_node.addSubNode(new_sub_node)

    def saveTubes(self, fileName):
        with open(fileName, "w") as f:
            f.write(f"{len(self.tubeBuffer)}\n")
            for main_node in self.tubeBuffer:
                f.write(f"{main_node.gId}\n")
                for sub_node in main_node.subNodes:
                    f.write(f"{sub_node.gId} {sub_node.belongedTube} {sub_node.sliceId}\n")

    def loadTubes(self, fileName):
        with open(fileName, "r") as f:
            bufferSize = int(f.readline().strip())
            self.tubeBuffer = [MainNode(i) for i in range(bufferSize)]
            for main_node in self.tubeBuffer:
                main_node_id = int(f.readline().strip())
                for _ in range(main_node.getSubNodeNumber()):
                    gId, belongedTube, sliceId = map(int, f.readline().split())
                    sub_node = SubNode(gId, belongedTube, sliceId)
                    main_node.addSubNode(sub_node)

'''


