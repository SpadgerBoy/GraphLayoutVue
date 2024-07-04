import ort from 'onnxruntime-web/webgpu';
import Tensor from 'onnxruntime-web';

import {createArray, createNMArray, opeNMArrays,  center_pos, randn_like, generateNormalDistribution as torch_randn} from './tools.js';
import tensorPreProcess from './dataPreProcess.js';

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

    this.TPP = new tensorPreProcess(ort);
    this.opt = {
      executionProviders: [],
      enableMemPattern: false,
      enableCpuMemArena: false,
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
    const executionProviders = ['webgpu'];

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

  async bindData2GPU(data, dataType, dims){

    const dataBuffer = ort.env.webgpu.device.createBuffer({
      // size: data.byteLength,
      size: Math.ceil(dims[0]*dims[1] * 4 / 16) * 16,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });

    ort.env.webgpu.device.queue.writeBuffer(dataBuffer, 0, data.buffer, data.byteOffset, data.byteLength);

    const data_tensor = ort.Tensor.fromGpuBuffer(dataBuffer, {
      dataType: dataType,
      dims: dims
    });

    console.log('node_emb_tensor:',data_tensor);

    return data_tensor;

  }
  // 预分配GPU张量
  // async preData2GPUTensor(dataType, dims){
  //   const myPreAllocatedBuffer = ort.env.webgpu.device.createBuffer({
  //     usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  //     size: Math.ceil(dims[0]*dims[1] * 4 / 16) * 16 /* align to 16 bytes */
  //   });
  //   const myPreAllocatedOutputTensor = ort.Tensor.fromGpuBuffer(myPreAllocatedBuffer, {
  //     dataType: dataType,
  //     dims: dims,
  //   });

  //   return myPreAllocatedOutputTensor;
  // }


  async onnx_infer(onnx_session, node_emb_tensor, node_level_tensor, pos_tensor, edge_index_tensor, edge_type_tensor, N, E, i){

    // node_emb = createArray(node_emb);
    // node_level = createArray(node_level);
    // pos = createArray(pos);
    // edge_index = createArray(edge_index);
    // edge_type = createArray(edge_type);

    var time1 = performance.now();
    // var node_emb_tensor = await this.bindData2GPU(node_emb, 'float32', [N, 3]);
    // console.log('success',node_emb_tensor);

    // var node_emb_tensor = new ort.Tensor('float32', node_emb, [N, 3]);
    // var node_level_tensor = new ort.Tensor('int64', node_level, [N, 1]);
    // var pos_tensor = new ort.Tensor('float32', pos, [N, 2]);
    // var edge_index_tensor = new ort.Tensor('int64', edge_index, [2, E]);
    // var edge_type_tensor = new ort.Tensor('int64', edge_type, [E]);


    if(i<199){
      const pos_tensor = ort.Tensor.fromGpuBuffer(pos_tensor, {
        dataType: 'float32',
        dims: [N, 2]
      });

    }

    const inputs = {  node_emb:node_emb_tensor, 
                    node_level: node_level_tensor, 
                    pos: pos_tensor, 
                    edge_index: edge_index_tensor, 
                    edge_type: edge_type_tensor,
                  };

    const preOutputTensor = await this.TPP.preData2GPUTensor('float32', [N, 2]);
    // const preOutputTensor = myPreAllocatedOutputTensor;
    const fetches = { 'pos_noise': preOutputTensor };

    var time2 = performance.now();
    var output = await onnx_session.run(inputs, fetches);
    var time3 = performance.now();

    console.log('onnx_time0:', (time2 - time1)/1000, 'onnx_time1:', (time3 - time2)/1000);
    console.log('output:', output);
    // var outputdata = output.pos_noise.getdata();

    return output.pos_noise;
    /*
    // 创建一个临时缓冲区以便从 GPU 中读取数据
    const readBuffer = ort.env.webgpu.device.createBuffer({
      size: outputDataBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // 创建一个命令编码器
    const commandEncoder = ort.env.webgpu.device.createCommandEncoder();

    // 将数据从 GPU 缓冲区复制到临时缓冲区
    commandEncoder.copyBufferToBuffer(
      outputDataBuffer, 0,
      readBuffer, 0,
      outputDataBuffer.size
    );

    // 提交命令队列
    const gpuCommands = commandEncoder.finish();
    ort.env.webgpu.device.queue.submit([gpuCommands]);

    // 等待缓冲区映射
    await readBuffer.mapAsync(GPUMapMode.READ);

    // 获取映射的数据
    const arrayBuffer = readBuffer.getMappedRange();
    const resultArray = new Float32Array(arrayBuffer);

    // 打印结果
    console.log('pos_noise:', resultArray);

    // 解除映射并清理缓冲区
    // readBuffer.unmap();*/

    // var pos_noise = createNMArray(output.pos_noise.data, output.pos_noise.data.length/2, 2);

    // return pos_noise;
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

    
    const N = pos.length;
    const E = edge_type.length;

    // node_emb = createArray(node_emb);
    // node_level = createArray(node_level);
    var pos_0 = createArray(pos);
    // edge_index = createArray(edge_index);
    // edge_type = createArray(edge_type);

    var time1 = performance.now();

    var node_emb_tensor = new ort.Tensor('float32', node_emb, [N, 3]);
    var node_level_tensor = new ort.Tensor('int64', node_level, [N, 1]);
    var pos_tensor = new ort.Tensor('float32', pos_0, [N, 2]);
    var edge_index_tensor = new ort.Tensor('int64', edge_index, [2, E]);
    var edge_type_tensor = new ort.Tensor('int64', edge_type, [E]);

    for(var i of seq){
      console.log('i:', i);

      var j = i-1;
      // var t = Array(num_graphs).fill(i);

      var time1 = performance.now();

      // var pos_noise_predict = await this.onnx_infer(onnx_session, node_emb, node_level, pos, edge_index, edge_type);
      var pos_noise_predict = await this.onnx_infer(onnx_session, node_emb_tensor, node_level_tensor, pos_tensor, edge_index_tensor, edge_type_tensor, N, E, i);
      var time2 = performance.now();
      pos_tensor = pos_noise_predict;
      console.log(pos_tensor);

      console.log('onnx_time:', (time2 - time1)/1000);
      onnx_time += time2 - time1;

      // var pos_next = [];
      // var temp = [];
      /*
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
      */
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