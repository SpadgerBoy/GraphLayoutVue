/* 扩散模型 去噪 */

import ort from 'onnxruntime-web/webgpu';
// import ort from 'onnxruntime-web';

import {createNMArray, opeNMArrays,  center_pos, randn_like} from './tools.js';

ort.env.wasm.wasmPaths = './js/';
ort.env.wasm.numThreads = 6;  //设置线程数

/*
steps=100:
  10：2.64s
  8：2.0-2.2s
  7:1.9-2.0s
  6：1.9-2.0s
  5:2.0s
  4：2.0-2.1s
  2：2.9s
  1: 4.4s
*/ 

// Diffusion Model
export default class DiffModel {

  constructor(config) {

    this.config = config;

    //噪音因子betas
    if(config.steps === 100){
      this.betas0 = get_beta_schedule(config.beta_start, config.beta_end, 500);
      this.betas = this.betas0.slice(-config.steps)
    }
    else{
      this.betas = get_beta_schedule(config.beta_start, config.beta_end, config.steps);
    }
    this.alphas = get_alphas(this.betas);

    this.steps = config.steps;

    //配置onnxruntime的部分参数
    this.opt = {
      executionProviders: [],
      enableMemPattern: false,
      enableCpuMemArena: false,
    };

    // 查看WEBGPU是否可用，如果是由wasm初始化就不会调用GPU
    this.init();

  }
  //产看webGPU是否可用
  async init() {
    console.log('Number of CPU threads:', navigator.hardwareConcurrency/2);
    // 是否支持 WebGPU
    if (!navigator.gpu) {
        alert("WebGPU not supported.");
        console.warn("WebGPU is not supported.");
    }
    else{
      // 适配器
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
          console.warn("Couldn't request WebGPU adapter.");
      }

      // 获取逻辑设备
      const device = await adapter.requestDevice();
      // ort.env.webgpu.device = device;
      console.log(device)
      console.log("WebGPU is supported.");
    }


}
  //载入onnx模型
  async load_onnx(modelPath) {
    // const executionProviders = ['webgpu', 'webgl', 'webnn', 'wasm'];
    const executionProviders = ['wasm', 'webgpu', ];

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

  //使用onnx模型进行推理，
  async onnx_infer(onnx_session, node_emb_tensor, node_level_tensor, pos_tensor, edge_index_tensor, edge_type_tensor){

    var inputs = {  node_emb:node_emb_tensor, 
                    node_level: node_level_tensor, 
                    pos: pos_tensor, 
                    edge_index: edge_index_tensor, 
                    edge_type: edge_type_tensor,
                  };

    var output = await onnx_session.run(inputs);
    
    //将pos重整为2维数组
    var pos_noise = createNMArray(output.pos_noise.data, output.pos_noise.data.length/2, 2);

    return pos_noise;
  }

  async run(node_emb, node_level, pos_init, edge_index, edge_type){

    //噪音因子alpha，betas，sigmas
    var alpha = get_sqrt(this.betas);
    var betas = this.betas;
    var sigmas = get_sqrt(this.alphas);
    var steps = this.steps;

    var pos = pos_init;

    pos = center_pos(pos_init);

    var seq = []
    for (var ii = steps-1; ii >= 0; ii--) {
      seq.push(ii);
    }
    //var seq_next = seq.slice(1, seq.length).concat([-1]);

    //是否开启跨域隔离
    console.log('Is cross-origin isolated:', window.crossOriginIsolated);

    //载入onnx模型
    var onnx_session = await this.load_onnx(this.config.model);
    

    const N = pos.length;
    const E = edge_type.length;
    //将初始数据整理为onnx模型所需要的Tensor
    const node_emb_tensor = new ort.Tensor('float32', node_emb, [N, 3]);
    const node_level_tensor = new ort.Tensor('int64', node_level, [N, 1]);
    const edge_index_tensor = new ort.Tensor('int64', edge_index, [2, E]);
    const edge_type_tensor = new ort.Tensor('int64', edge_type, [E]);

    //
    for(var i of seq){

      var j = i-1;

      const pos0 = pos.flat();
      var pos_tensor = new ort.Tensor('float32', pos0, [N, 2]);

      //预测本步骤的噪音
      var pos_noise_predict = await this.onnx_infer(onnx_session, node_emb_tensor, node_level_tensor, pos_tensor, edge_index_tensor, edge_type_tensor);


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
      pos = center_pos(pos);

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
