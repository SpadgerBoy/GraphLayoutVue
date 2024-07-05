
import DiffModel from "./DM/DM.js";
import {createNMArray, } from './DM/tools.js'
import convertGridToCart from './data_process/coordinate_homogenization.js'
import getData from './data_process/get_data.js'

const config = {
    beta_start: 0.001,
    beta_end: 0.02,
    steps: 200,
    model: "./onnx/GLM200.onnx",
};

// const config = {
//   beta_start: 0.001,
//   beta_end: 0.02,
//   steps: 100,
//   model: "./onnx/GLM100.onnx",
// };
export default async function get_new_graph(N, all_edges) {
  const time1 = performance.now();
  
  const repoense = await getData(N, all_edges);

  const node_emb = repoense.node_emb;
  const node_level = repoense.node_level;
  const pos_init = repoense.pos_init;
  const edge_index = repoense.edge_index;
  const edge_type = repoense.edge_type;
  // const num_graphs = 1;

  const time2 = performance.now();


  pos_init = createNMArray(pos_init, N, 2);

  let DM = new DiffModel(config);

  const pos = await DM.run(node_emb, node_level, pos_init, edge_index, edge_type, num_graphs);  

  pos = convertGridToCart(pos, node_level);
  
  console.log('pos_gen', pos);

  const time3 = performance.now();

  console.log('onnx_time:',(time3 - time2)/1000, 'total_time:',(time3 - time1)/1000);


  return pos;

};


