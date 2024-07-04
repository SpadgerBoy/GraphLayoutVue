import ort from 'onnxruntime-web/webgpu';
// import ort from 'onnxruntime-web';

import {createArray, createNMArray, opeNMArrays,  center_pos, randn_like, generateNormalDistribution as torch_randn} from './tools.js';


ort.env.wasm.wasmPaths = './js/';
ort.env.wasm.numThreads = 10;

// Diffusion Model 
export default class tensorPreProcess {
  constructor(ort) {
    this.ort = ort;
  }
    async bindData2GPU(data, dataType, dims){

    const dataBuffer = this.ort.env.webgpu.device.createBuffer({
      size: Math.ceil(dims[0]*dims[1] * 4 / 16) * 16,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });

    this.ort.env.webgpu.device.queue.writeBuffer(dataBuffer, 0, data.buffer, data.byteOffset, data.byteLength);

    const data_tensor = this.ort.Tensor.fromGpuBuffer(dataBuffer, {
      dataType: dataType,
      dims: dims
    });

    return data_tensor;

  }
  //预分配GPU张量
  async preData2GPUTensor(dataType, dims){
    const myPreAllocatedBuffer = this.ort.env.webgpu.device.createBuffer({
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      size: Math.ceil(dims[0]*dims[1] * 4 / 16) * 16 /* align to 16 bytes */
    });
    const myPreAllocatedOutputTensor = this.ort.Tensor.fromGpuBuffer(myPreAllocatedBuffer, {
      dataType: dataType,
      dims: dims,
    });

    return myPreAllocatedOutputTensor;
  }

}