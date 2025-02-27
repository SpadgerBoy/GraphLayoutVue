from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from process.run import get_dataset

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'


@app.route('/getdata', methods=['POST'])
@cross_origin()
def process_data():
    data = request.get_json()
    num_nodes = data.get('data1')
    all_edges = data.get('data2')

    print(
        f'Received data:'
        f'\n    num_nodes: {num_nodes}',
        f'\n    all_edges: Array[{len(all_edges)}, {len(all_edges[0])}]', 
        )


    node_emb, node_level, pos_init, edge_index, edge_type = get_dataset(num_nodes, all_edges)
    response = {
        'node_emb': node_emb,
        'node_level': node_level,
        'pos_init': pos_init,
        'edge_index': edge_index,
        'edge_type': edge_type,
    }

    return jsonify(response)


if __name__ == "__main__":
    print('run 101.36.73.174:14449')
    app.run(host='101.36.73.174', port=14449)

    

