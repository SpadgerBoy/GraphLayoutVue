from process.get_data import data_process


def get_dataset(node_num, all_edges):

    data_model = data_process()
    node_emb, node_level, pos_init, edge_index, edge_type = data_model(node_num, all_edges)

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

    node_num = 100

    # all_edges = torch.tensor(edges)

    node_emb, node_level, pos_init, edge_index, edge_type = get_dataset(node_num, edges)

    print(node_emb, '\n', node_level, '\n', pos_init, '\n', edge_index, '\n', edge_type, )
