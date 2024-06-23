from processing.tabular_data_parse import parse_sheet
from processing.get_data import data_process
tabular_dataset = []

def load_tabular_dataset(node_num, all_edges):
    '''
        read tabular dataset and process
    '''
    global node_emb, node_level, pos_init, edge_index, edge_type
    data_model = data_process()
    node_emb, node_level, pos_init, edge_index, edge_type = data_model(node_num, all_edges)

    return node_emb, node_level, pos_init, edge_index, edge_type


def get_tabular_dataset(node_num, all_edges):

    data_model = data_process()
    node_emb, node_level, pos_init, edge_index, edge_type = data_model(node_num, all_edges)

    return node_emb, node_level, pos_init, edge_index, edge_type