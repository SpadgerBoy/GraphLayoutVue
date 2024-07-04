// import { GPU } from 'gpu.js';
const GPU = require('gpu.js');
function createOperationKernel(ope) {
    switch (ope) {
      case 'add':
        return function(a, b) {
          return a + b;
        };
      case 'sub':
        return function(a, b) {
          return a - b;
        };
      case 'mul':
        return function(a, b) {
          return a * b;
        };
      case 'div':
        return function(a, b) {
          return a / b;
        };
      default:
        throw new Error(`Unsupported operation: ${ope}`);
    }
  }
  
  //export function opeNMArraysGPU(array1, num2, ope) {
function opeNMArraysGPU(array1, num2, ope) {
    const gpu = new GPU();
  
    // 根据操作创建内核函数
    const kernelFunction = createOperationKernel(ope);
  
    let kernel;
    if (Array.isArray(num2)) {
      // 处理数组与数组之间的运算
      kernel = gpu.createKernel(kernelFunction)
        .setOutput([array1.length, array1[0].length]);
      const result = kernel(array1, num2);
      return result.toArray();
    } else {
      // 处理数组与标量的运算，将标量转换为与数组相同形状的数组
      const scalarArray = new Array(array1.length).fill(num2).map(() => new Array(array1[0].length).fill(num2));
      kernel = gpu.createKernel(kernelFunction)
        .setOutput([array1.length, array1[0].length]);
      const result = kernel(array1, scalarArray);
      return result.toArray();
    }
  }


const array1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
];
  
const array2 = [
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
];
  
const scalar = 2;
  
  // 加法测试
async function testAdd() {
    try {
      const result = await opeNMArraysGPU(array1, array2, 'add');
      console.log('加法结果:', result);
    } catch (error) {
      console.error('加法测试出错:', error);
    }
  }
  
  // 减法测试
async function testSub() {
    try {
      const result = await opeNMArraysGPU(array1, array2, 'sub');
      console.log('减法结果:', result);
    } catch (error) {
      console.error('减法测试出错:', error);
    }
  }
  
  // 乘法测试
async function testMul() {
    try {
      const result = await opeNMArraysGPU(array1, scalar, 'mul');
      console.log('乘法结果:', result);
    } catch (error) {
      console.error('乘法测试出错:', error);
    }
  }
  
  // 除法测试
async function testDiv() {
    try {
      // 注意：确保没有除以0的情况
      const result = await opeNMArraysGPU(array1, scalar, 'div');
      console.log('除法结果:', result);
    } catch (error) {
      console.error('除法测试出错:', error);
    }
  }
  
  // 执行所有测试
testAdd();
testSub();
testMul();
testDiv();



