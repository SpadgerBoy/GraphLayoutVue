/**
 * 根据N、all_edges由模型计算出新的坐标分配
 */
import DiffModel from "./DM/DM.js";
import {createNMArray, } from './DM/tools.js'
import convertGridToCart from './data_process/coordinate_homogenization.js'
import getData from './data_process/get_data.js'

// 配置模型参数
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

export default async function get_new_pos(N, all_edges) {
  
  //从后端获取数据
  const repoense = await getData(N, all_edges);
  const node_emb = repoense.node_emb;
  const node_level = repoense.node_level;
  var pos_init = repoense.pos_init;
  const edge_index = repoense.edge_index;
  const edge_type = repoense.edge_type;
  // const num_graphs = 1;


  //将一个一维数组转换为[N,2]二维数组
  pos_init = createNMArray(pos_init, N, 2);

  //初始化DiffModel
  let DM = new DiffModel(config);
  var pos = await DM.run(node_emb, node_level, pos_init, edge_index, edge_type);  

  //将坐标均匀化
  pos = convertGridToCart(pos, node_level);
  
  // console.log('pos_gen', pos);

  return pos;

};


