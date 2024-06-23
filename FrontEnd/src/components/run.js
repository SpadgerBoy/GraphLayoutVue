
import DiffModel from "./DM/DM.js";
import {createNMArray, randn_like, generateNormalDistribution as torch_randn} from './DM/tools.js'
import convertGridToCart from './data_process/coordinate_homogenization.js'
import getData from './data_process/get_data.js'

var config = {
    beta_start: 0.001,
    beta_end: 0.02,
    steps: 200,
    hidden_dim: 128,
};

export default async function run_onnx(N, all_edges) {
  let DM = new DiffModel(config);

  
  var onnx_path = './onnx/train200_adam/GLM.onnx';

  var repoense = await getData(N, all_edges);

  var node_emb = repoense.node_emb;
  var node_level = repoense.node_level;
  var pos_init = repoense.pos_init;
  var edge_index = repoense.edge_index;
  var edge_type = repoense.edge_type;
  var num_graphs = 1;


  pos_init = createNMArray(pos_init, N, 2);

  var pos = await DM.test(onnx_path, node_emb, node_level, pos_init, edge_index, edge_type, num_graphs);  
  
  pos = convertGridToCart(pos, node_level);
  // console.log('pos_gen', pos);

  return pos;

};
// export default run_onnx;

