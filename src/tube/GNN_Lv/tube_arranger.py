import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SubNode:
    def __init__(self, gId, belongedTube, sliceId):
        self.gId = gId  # 全局唯一 ID
        self.belongedTube = belongedTube  # 所属管道 ID
        self.sliceId = sliceId  # 管道中的位置 ID
        self.color = -1  # 初始颜色

class MainNode:
    def __init__(self, gId):
        self.isDeleted = False  # 标志是否已被删除
        self.gId = gId  # 全局唯一 ID
        self.color = 0  # 初始颜色
        self.subNodes = []  # 管道中的所有帧
        self.neighborList = []  # 相邻管道

    def addSubNode(self, subNode):
        self.subNodes.append(subNode)  # 添加帧

    def getSubNodeNumber(self):
        return len(self.subNodes)  # 返回帧数量


# 定义一个基于图注意力网络（GAT）的模型
class GATModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, heads=4):
        """
        初始化 GAT 模型。

        参数：
        - input_dim: 输入节点特征的维度
        - output_dim: 输出节点特征的维度
        - hidden_dim: GNN 隐藏层的维度，默认值为 64
        - heads: 图注意力网络（GAT）的头数，默认值为 4（多头注意力机制）
        """
        super(GATModel, self).__init__()

        # 第一层图注意力卷积层，输入维度为input_dim，输出维度为hidden_dim，使用多个heads
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)

        # 第二层图注意力卷积层，输入维度为hidden_dim * heads（多头的结果拼接），输出维度为output_dim
        # 第二层使用一个head输出最终的结果
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)

    def forward(self, x, edge_index):
        """
        前向传播函数，用于计算节点特征。

        参数：
        - x: 节点特征矩阵，形状为 (num_nodes, input_dim)
        - edge_index: 边索引矩阵，形状为 (2, num_edges)，表示节点之间的边连接关系

        返回：
        - 经过 GAT 处理后的节点特征矩阵
        """
        # 使用第一层 GAT 进行特征传播并使用 ELU 激活函数
        x = F.elu(self.conv1(x, edge_index))

        # 通过第二层 GAT 进一步处理特征，并得到最终输出特征
        x = self.conv2(x, edge_index)
        return x


# 管道重排器，使用 GAT 模型对视频管道中的子节点进行重排
class TubeArranger:
    def __init__(self, input_dim, output_dim, hidden_dim=64, heads=4):
        """
        初始化 TubeArranger 类。

        参数：
        - input_dim: 输入特征的维度
        - output_dim: 输出特征的维度
        - hidden_dim: GNN 隐藏层的维度，默认值为 64
        - heads: 图注意力网络（GAT）的头数，默认值为 4
        """
        self.tubeBuffer = []  # 用于存储生成的 MainNode，代表视频管道的缓存

        # 创建 GAT 模型，用于处理节点特征
        self.gnn_model = GATModel(input_dim, output_dim, hidden_dim, heads)

    def tubeRearranging(self, tube, edge_index, node_features):
        """
        使用 GNN 模型对管道进行重排。

        参数：
        - tube: 代表当前要处理的 Tube 对象，包含多个子节点
        - edge_index: 边索引矩阵，表示 Tube 中的帧之间的连接关系
        - node_features: 节点特征矩阵，表示每个节点的初始特征

        此方法主要用于重排 Tube 中的子节点，调整每个节点的顺序。
        """
        # print("rearranging")
        # 创建一个新的 MainNode 对象，作为新管道的主节点
        new_main_node = MainNode(len(self.tubeBuffer))
        # 使用当前 tubeBuffer 长度作为 MainNode 的全局 ID，方便后续的处理
        # 新的主节点代表当前管道中的主帧

        # 将这个新的主节点添加到管道缓存中，等待后续使用
        self.tubeBuffer.append(new_main_node)

        # 使用 GNN 模型，根据节点特征和边索引对节点特征进行更新
        # 通过图神经网络 (GNN) 来捕获节点间的关联信息，并产生新的节点特征
        node_features = self.gnn_model(node_features, edge_index)
        print("节点特征已更新:", node_features)  # 输出更新后的节点特征，方便调试

        # 遍历当前 Tube 中的所有帧（每一帧会作为一个子节点）
        for i in range(tube.getSubNodeNumber()):
            # 创建每一帧对应的 SubNode 对象
            # 参数分别为：子节点的全局 ID、主节点的全局 ID、子节点在该管道中的帧位置
            new_sub_node = SubNode(len(self.tubeBuffer), new_main_node.gId, i)

            # 将新创建的子节点添加到主节点中，形成主节点与子节点之间的关联
            new_main_node.addSubNode(new_sub_node)
            print(f"添加子节点: 子节点ID={len(self.tubeBuffer)}, 主节点ID={new_main_node.gId}, 位置={i}")

        # 至此，当前管道的重排操作结束，新的主节点与其子节点已建立关联

    def build_edge_index(self, node_features, threshold=0.5):
        """
        构建边列表（edge_index），包括时间顺序边和空间相关边。

        Parameters:
        - node_features (Tensor): 节点特征矩阵，形状为 [num_nodes, input_dim]。
        - threshold (float): 空间相关边的特征相似度阈值。

        Returns:
        - edge_index (Tensor): 形状为 [2, num_edges] 的边列表，表示节点之间的连接关系。
        """
        edge_index = []

        # 时间顺序边：同一管道中的相邻帧
        for tube in self.tubeBuffer:
            sub_nodes = tube.subNodes
            for i in range(len(sub_nodes) - 1):
                edge_index.append([sub_nodes[i].gId, sub_nodes[i + 1].gId])  # i -> i+1
                edge_index.append([sub_nodes[i + 1].gId, sub_nodes[i].gId])  # i+1 -> i（双向边）

        # 空间相关边：基于特征相似度的帧间连接
        for i in range(len(node_features)):
            for j in range(i + 1, len(node_features)):
                if self.compute_overlap(node_features[i], node_features[j], threshold) > 0:
                    edge_index.append([i, j])  # i -> j
                    edge_index.append([j, i])  # j -> i（双向边）

        # 转换为 Tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return edge_index

    def saveTubes(self, fileName):
        with open(fileName, "w") as f:
            f.write(f"{len(self.tubeBuffer)}\n")
            for main_node in self.tubeBuffer:
                f.write(f"{main_node.gId}\n")
                for sub_node in main_node.subNodes:
                    f.write(f"{sub_node.gId} {sub_node.belongedTube} {sub_node.sliceId}\n")

    def loadTubes(self, fileName):
        try:
            with open(fileName, "r") as f:
                bufferSize = int(f.readline().strip())
                self.tubeBuffer = [MainNode(i) for i in range(bufferSize)]
                for main_node in self.tubeBuffer:
                    main_node_id = int(f.readline().strip())
                    for _ in range(main_node.getSubNodeNumber()):
                        gId, belongedTube, sliceId = map(int, f.readline().split())
                        sub_node = SubNode(gId, belongedTube, sliceId)
                        main_node.addSubNode(sub_node)
        except FileNotFoundError:
            print(f"File {fileName} not found.")
        except Exception as e:
            print(f"Error loading tubes: {e}")

    def compute_overlap(self, feature_i, feature_j, threshold=0.5):
        """
        计算两个节点特征之间的重叠程度。
        """
        distance = torch.norm(feature_i - feature_j, p=2)
        overlap = 1.0 if distance < threshold else 0.0
        return overlap

    def compute_total_loss(self, node_features, edge_index):
        """
        计算整体损失，包括时间顺序一致性和空间冲突。
        """
        predicted_features = self.gnn_model(node_features, edge_index)

        temporal_loss = 0.0
        spatial_loss = 0.0

        # 时间顺序一致性损失
        for tube in self.tubeBuffer:
            sub_nodes = tube.subNodes
            for i in range(len(sub_nodes) - 1):
                temporal_loss += torch.abs(
                    predicted_features[sub_nodes[i].gId] - predicted_features[sub_nodes[i + 1].gId])

        # 空间冲突损失
        for i in range(len(predicted_features)):
            for j in range(i + 1, len(predicted_features)):
                spatial_loss += self.compute_overlap(predicted_features[i], predicted_features[j])

        # 加权总损失
        lambda_temporal = 1.0
        lambda_spatial = 1.0
        total_loss = lambda_temporal * temporal_loss + lambda_spatial * spatial_loss

        return total_loss
#强化学习
#