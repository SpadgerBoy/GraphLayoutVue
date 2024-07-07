'''
根据N、all_edges获取数据所需要数据node_emb, node_level, pos_init, edge_index, edge_type
'''

import torch.nn as nn
from process.transforms import *
from process.get_layer import layer_graph
import torch
from torch_geometric.data import Data, Dataset, Batch

def get_graph_data(all_edges, node_level):
     
    r'''
    根据网络拓扑结构进行初步的预处理，使用torch_geometric.data 将图数据打包
    '''

    edge_list = np.array(all_edges).tolist()

    #torch_pos = torch.tensor(pos_list, dtype=torch.float32)
    torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
    torch_edge_index = torch.transpose(torch_edge_index, 0, 1)
    node_level = torch.tensor(node_level, dtype=torch.long)

    data = [Data(num_nodes=len(node_level), edge_index=torch_edge_index, graph_name='graph1', level=node_level)]
    return data

class GraphLayoutDataset(Dataset):
    r'''
    调用transforms对数据打包的图数据进行预处理，以获得预期的图拓扑信息
    '''
    def __init__(self, data=None, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform
        for item in self.data:
            self.transform(item)
        #print('GraphLayoutDataset:', self.data)

    def __getitem__(self, idx):
        data = self.data[idx].clone()
        return data

    def __len__(self):
        return len(self.data)


class data_process(nn.Module):
    def __init__(self):
        super(data_process, self).__init__()

    def forward(self, node_num, all_edges):

        transforms = Compose([
            CountNodesPerGraph(),
            AddUndiectedEdge(),
            AddNodeType(),
            AddNodeMask(node_mask=0.0),
            AddNodeDegree(),
            AddLaplacianEigenvectorPE(k=3),  # Offline edge augmentation
            #AddRandomWalkPE(walk_length=int(config.model.laplacian_eigenvector)),
            AddEdgeType(),
            AddHigherOrderEdges(order=3),  # Offline edge augmentation
            #AddFragmentEdge(fragment_edge_type=config.model.fragment_edge_type),
        ])

        # 获得分层信息
        node_level = layer_graph(all_edges)
        # 对图数据进行进行初步预处理，并打包
        data = get_graph_data(all_edges, node_level)
        # 根据初步预处理的数据，使用transforms对其进行再次预处理
        new_data = GraphLayoutDataset(data, transform=transforms)
        
        # 得到我们预期的预处理信息
        node_emb = new_data[0].node_emb.tolist()
        edge_index = new_data[0].edge_index.tolist()
        edge_type = new_data[0].edge_type.tolist()
        # 随机生成一个服从正态分布的正态分布坐标（初始噪音）
        pos_init = torch.randn(node_num, 2).tolist()

        print(
            f'Response data:'
            f'\n    node_emb: Array[{len(node_emb)}, {len(node_emb[0])}]',
            f'\n    node_level: Array[{len(node_level)}]',
            f'\n    pos_init: Array[{len(pos_init)}, {len(pos_init[0])}]',
            f'\n    edge_index: Array[{len(edge_index)}, {len(edge_index[0])}]',
            f'\n    edge_type: Array[{len(edge_type)}]',
        )

        node_emb = [i for row in node_emb for i in row]
        pos_init = [i for row in pos_init for i in row]
        edge_index = [i for row in edge_index for i in row]


        return node_emb, node_level, pos_init, edge_index, edge_type


if __name__ == '__main__':

    edges = [[0, 21], [1, 13], [2, 21], [2, 16], [3, 16], [4, 10], [5, 21], [5, 20], [6, 21], [6, 20], [7, 16],
                 [7, 12], [8, 17], [8, 15], [9, 15], [10, 27], [11, 27], [12, 26], [13, 23], [14, 27], [14, 24],
                 [15, 27], [16, 22], [17, 27], [18, 28], [19, 30], [20, 27], [21, 24], [22, 32], [23, 33], [23, 32],
                 [24, 39], [24, 36], [25, 32], [25, 39], [26, 34], [27, 38], [28, 34], [29, 35], [30, 37], [30, 39],
                 [31, 47], [31, 44], [32, 45], [32, 48], [33, 49], [33, 46], [34, 42], [35, 48], [35, 42], [36, 43],
                 [36, 45], [37, 43], [38, 47], [38, 46], [39, 44], [40, 47], [41, 47], [41, 48], [42, 58], [42, 50],
                 [43, 56], [44, 60], [44, 54], [45, 54], [46, 53], [47, 54], [47, 55], [48, 52], [48, 58], [49, 51],
                 [49, 58], [50, 63], [50, 65], [51, 68], [52, 65], [53, 67], [54, 62], [55, 67], [55, 66], [56, 63],
                 [57, 65], [58, 68], [59, 62], [59, 66], [60, 68], [61, 63], [61, 68], [62, 75], [63, 79], [63, 80],
                 [64, 74], [65, 71], [65, 79], [66, 73], [66, 72], [67, 70], [68, 71], [68, 78], [69, 88], [70, 84],
                 [71, 86], [72, 87], [73, 81], [74, 87], [75, 88], [75, 84], [76, 82], [77, 87], [77, 89], [78, 86],
                 [79, 90], [79, 81], [80, 84], [81, 92], [81, 97], [82, 94], [82, 92], [83, 96], [83, 95], [84, 93],
                 [85, 96], [86, 92], [86, 97], [87, 96], [88, 92], [89, 94], [89, 99], [90, 91], [90, 98]]
    node_level0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
                  4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7,
                  7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10,
                  10]
    node_num = 100

    # all_edges = torch.tensor(edges)

    model = data_process()

    node_emb, node_level, pos_init, edge_index, edge_type = model(node_num, all_edges)
    
    print(node_emb, '\n', node_level, '\n', pos_init, '\n', edge_index, '\n', edge_type, )



    

