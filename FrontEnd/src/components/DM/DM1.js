import ort from 'onnxruntime-web/webgpu';
// import ort from 'onnxruntime-web';

import {createArray, createNMArray, opeNMArrays,  center_pos, randn_like, generateNormalDistribution as torch_randn} from './tools.js';


ort.env.wasm.wasmPaths = './js/';
ort.env.wasm.numThreads = 10;

// Diffusion Model 
export default class DiffModel {

  constructor(config) {
    this.config = config;
    if(config.steps === 100){
      this.betas0 = get_beta_schedule(config.beta_start, config.beta_end, 500);
      this.betas = this.betas0.slice(-config.steps)
    }
    else{
      this.betas = get_beta_schedule(config.beta_start, config.beta_end, config.steps);
    }
    
    this.alphas = get_alphas(this.betas);
    this.steps = config.steps;
    this.hiddem_dim = config.hiddem_dim;
    this.opt = {
      executionProviders: [],
      enableMemPattern: false,
      enableCpuMemArena: false,
      // extra: {
      //     session: {
      //         disable_prepacking: "1",
      //         use_device_allocator_for_initializers: "1",
      //         use_ort_model_bytes_directly: "1",
      //         use_ort_model_bytes_for_initializers: "1"
      //     }
      // },
    };

  }
  //产看webGPU是否可用
  async init() {
    // 是否支持 WebGPU
    if (!navigator.gpu) {
        alert("WebGPU not supported.");
        console.warn("WebGPU is not supported.");
    }

    // 适配器
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        console.warn("Couldn't request WebGPU adapter.");
    }

    // 获取逻辑设备
    const device = await adapter.requestDevice();
    ort.env.webgpu.device = device;
    console.log(device)
    console.log("WebGPU is supported.");

}
  async load_onnx(modelPath) {
    // const executionProviders = ['webgpu', 'webgl', 'webnn', 'wasm'];
    const executionProviders = ['webgpu', 'wasm'];

    for (const provider of executionProviders) {

      this.opt.executionProviders = [provider];
      const sess_opt = { ...this.opt };

      try {
        const session = await ort.InferenceSession.create(modelPath, sess_opt);
        console.log(`Session initialized with ${provider}.`);
        return session;
      } catch (error) {
        console.warn(`Your device or Browser doesn't support ${provider}:`, error);
      }
    }
    throw new Error('All execution providers failed to initialize.');
  }


  async onnx_infer(onnx_session, node_emb, node_level, pos, edge_index, edge_type){

    const N = pos.length;
    const E = edge_type.length;

    // node_emb = createArray(node_emb);
    // node_level = createArray(node_level);
    pos = createArray(pos);
    // edge_index = createArray(edge_index);
    // edge_type = createArray(edge_type);

    var time1 = performance.now();

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

    var time2 = performance.now();
    var output = await onnx_session.run(inputs);
    var time3 = performance.now();

    console.log('onnx_time0:', (time2 - time1)/1000, 'onnx_time1:', (time3 - time2)/1000);

    //console.log('pos_noise_predict:', output.pos_noise);
    var pos_noise = createNMArray(output.pos_noise.data, output.pos_noise.data.length/2, 2);

    return pos_noise;
  }

  async run(node_emb, node_level, pos_init, edge_index, edge_type, num_graphs){

    //查看WEBGPU是否可用
    this.init();

    var alpha = get_sqrt(this.betas);
    var betas = this.betas;
    var sigmas = get_sqrt(this.alphas);
    var steps = this.steps;

    var pos = pos_init;

    // pos = center_pos(pos_init, batch);
    pos = center_pos(pos_init);

    var seq = []
    for (var ii = steps-1; ii >= 0; ii--) {
      seq.push(ii);
    }
    //var seq_next = seq.slice(1, seq.length).concat([-1]);

    var onnx_session = await this.load_onnx(this.config.model);
    

    var onnx_time = 0;
    var matrix_time = 0;

    for(var i of seq){

      var j = i-1;
      // var t = Array(num_graphs).fill(i);

      var time1 = performance.now();
      var pos_noise_predict = await this.onnx_infer(onnx_session, node_emb, node_level, pos, edge_index, edge_type);
      var time2 = performance.now();

      console.log('onnx_time:', time2 - time1);
      onnx_time += time2 - time1;

      // var pos_next = [];
      // var temp = [];

      //去除噪音
      if(i > 0){
        //var eps_linker = torch_randn([pos.length, 2]);
        var eps_linker = randn_like([pos.length, 2]);

        var temp = opeNMArrays(pos_noise_predict, betas[i]/sigmas[i], 'mul');
        var gamma = (sigmas[j]**2)/(sigmas[i]**2)*betas[i];
        
        const temp1 = opeNMArrays((opeNMArrays(pos, temp, 'sub')), 1/alpha[i], 'mul')
        var temp2 = opeNMArrays(eps_linker, gamma, 'mul')
        var pos_next = opeNMArrays(temp1, temp2, 'add');
      }
      else{
        var temp = opeNMArrays(pos_noise_predict, betas[i]/sigmas[i], 'mul');
        var pos_next = opeNMArrays((opeNMArrays(pos, temp, 'sub')), 1/alpha[i], 'mul');
      }

      pos = pos_next;
      // pos = center_pos(pos, batch);
      pos = center_pos(pos);

      if (pos.some(Number.isNaN)) {
        console.log('NaN detected. Please restart.');
        throw new Error('FloatingPointError');
      }
      var time3 = performance.now();
      matrix_time += time3 - time2;

    }
    console.log('onnx_total_time:', onnx_time/1000);
    console.log('matrix_total_time:', matrix_time/1000);

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