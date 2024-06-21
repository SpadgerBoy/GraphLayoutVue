import ort from 'onnxruntime-web';
import {createArray, createNMArray, opeNMArrays,  center_pos, randn_like, generateNormalDistribution as torch_randn} from './tools.js'

// Diffusion Model 
class DiffModel {

  constructor(config) {
    this.config = config;
    this.betas = get_beta_schedule(config.beta_start, config.beta_end, config.steps);
    this.alphas = get_alphas(this.betas);
    this.steps = config.steps;
    this.hiddem_dim = config.hiddem_dim;
  }

  async load_onnx(onnx_path) {
    try {
        const session = await ort.InferenceSession.create(onnx_path);
  
        return session;
  
    } catch (error) {
        document.write(`failed to load ONNX model: ${error}.`);
    }
  }

  async onnx_infer(onnx_session, node_emb, node_level, pos, edge_index, edge_type){

    const N = pos.length;
    const E = edge_type.length;

    //node_emb = createArray(node_emb);
    //node_level = createArray(node_level);
    pos = createArray(pos);
    //edge_index = createArray(edge_index);
    //edge_type = createArray(edge_type);

    var node_emb_tensor = new ort.Tensor('float32', node_emb, [N, 3]);
    var node_level_tensor = new ort.Tensor('int64', node_level, [N, 1]);
    var pos_tensor = new ort.Tensor('float32', pos, [N, 2]);
    var edge_index_tensor = new ort.Tensor('int64', edge_index, [2, E]);
    var edge_type_tensor = new ort.Tensor('int64', edge_type, [E]);

    var inputs = {  node_emb:node_emb_tensor, 
                    node_level: node_level_tensor, 
                    pos: pos_tensor, 
                    edge_index: edge_index_tensor, 
                    edge_type: edge_type_tensor,
                  };
    var output = await onnx_session.run(inputs);

    //console.log('pos_noise_predict:', output.pos_noise);
    var pos_noise = createNMArray(output.pos_noise.data, output.pos_noise.data.length/2, 2);

    return pos_noise;
  }

  async test(onnx_path, node_emb, node_type, node_level, pos_init, edge_index, edge_type, batch, num_graphs, extend_order=false, extend_radius=true){
    
    var alpha = get_sqrt(this.betas);
    var betas = this.betas;
    var sigmas = get_sqrt(this.alphas);
    var steps = this.steps;

    var pos = pos_init

    pos = center_pos(pos_init, batch);

    var seq = []
    for (var ii = steps-1; ii >= 0; ii--) {
      seq.push(ii);
    }
    //var seq_next = seq.slice(1, seq.length).concat([-1]);

    var onnx_session = await this.load_onnx(onnx_path)

    for(var i of seq){

      var j = i-1;
      var t = Array(num_graphs).fill(i)

      var pos_noise_predict = await this.onnx_infer(onnx_session, node_emb, node_level, pos, edge_index, edge_type);
      
      var pos_next = [];
      var temp = [];

      //去除噪音
      if(i > 0){
        //var eps_linker = torch_randn([pos.length, 2]);
        var eps_linker = randn_like([pos.length, 2]);
        //console.log('eps_linker:', eps_linker);
        temp = opeNMArrays(pos_noise_predict, betas[i]/sigmas[i], 'mul');
        var gamma = (sigmas[j]**2)/(sigmas[i]**2)*betas[i];
        
        const temp1 = opeNMArrays((opeNMArrays(pos, temp, 'sub')), 1/alpha[i], 'mul')
        var temp2 = opeNMArrays(eps_linker, gamma, 'mul')
        pos_next = opeNMArrays(temp1, temp2, 'add');
      }
      else{
        temp = opeNMArrays(pos_noise_predict, betas[i]/sigmas[i], 'mul');
        pos_next = opeNMArrays((opeNMArrays(pos, temp, 'sub')), 1/alpha[i], 'mul');
      }

      //console.log('pos_:', pos_next);
      pos = pos_next;
      pos = center_pos(pos, batch);

      if (pos.some(Number.isNaN)) {
        console.log('NaN detected. Please restart.');
        throw new Error('FloatingPointError');
      }

    }

    return pos;
    
  }
}
function get_sqrt(list_a){

    var list_b = list_a.map(function (a) {
      return Math.sqrt(1.0 - a);
    });
  
    return list_b;
}


function sigmoid(x) {
  return 1 / (Math.exp(-x) + 1);
}

function get_beta_schedule(beta_start, beta_end, steps){

  var betas = [];
  for (var i = 0; i < steps; i++) {
    var x = -6 + (i / (steps - 1)) * 12;
    var sigmoidValue = sigmoid(x);
    var beta = sigmoidValue * (beta_end - beta_start) + beta_start;
    betas.push(beta);
  }
  return betas;
}

function get_alphas(betas){
  var alphas = [];
  var acc = 1;
  
  for (var i = 0; i < betas.length; i++) {
    var alpha = 1 - betas[i];
    acc *= alpha;
    alphas.push(acc);
  }
  return alphas;
}


export default DiffModel;




