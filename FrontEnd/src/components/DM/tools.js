export function arraycat(array1, array2){
  var concatenated = [];

  for (var i = 0; i < array1.length; i++) {
    var concatenatedRow = array1[i].concat(array2[i]);

    concatenated.push(concatenatedRow);
  }

  return concatenated;

}


//将一个长度为N*M的一维array转换成N*M维的array
export function createNMArray(array, rows, cols) {
    var doubleArray = [];
    
    for (var i = 0; i < rows; i++) {
      var innerArray = [];
      for (var j = 0; j < cols; j++) {
        innerArray.push(array[i * cols + j]); // 替换为你需要的初始值
      }
      doubleArray.push(innerArray);
    }
    
    return doubleArray;
};

//将一个N*M维的array转换成的一维array
export function createArray(array) {
  var Array = [];
  
  for (var i = 0; i < array.length; i++) {
    for (var j = 0; j < array[i].length; j++) {
      Array.push(array[i][j]); // 替换为你需要的初始值
    }
  }
  return Array;
};

export function opeNMArrays(array1, num2, ope) {

  //console.log(array1, num2)
  var result = [];

  if(Array.isArray(num2)){
    if(ope === 'add'){
      for (var i = 0; i < array1.length; i++) {
        var row = [];
        for (var j = 0; j < array1[0].length; j++){
          var sum = array1[i][j] + num2[i][j];
          row.push(sum);
        }
        result.push(row);
      }
    }
    else if(ope === 'sub'){
      for (var i = 0; i < array1.length; i++) {
        var row = [];
        for (var j = 0; j < array1[0].length; j++){
          var sum = array1[i][j] - num2[i][j];
          row.push(sum);
        }
        result.push(row);
      }
    }
  }
  else{
    if(ope === 'add'){
      for (var i = 0; i < array1.length; i++) {
        var row = [];
        for (var j = 0; j < array1[0].length; j++){
          var sum = array1[i][j] + num2;
          row.push(sum);
        }
        result.push(row);
      }
    }
    else if(ope === 'sub'){
      for (var i = 0; i < array1.length; i++) {
        var row = [];
        for (var j = 0; j < array1[0].length; j++){
          var sum = array1[i][j] - num2;
          row.push(sum);
        }
        result.push(row);
      }
    }
    else if(ope === 'mul'){
      for (var i = 0; i < array1.length; i++) {
        var row = [];
        for (var j = 0; j < array1[0].length; j++){
          var sum = array1[i][j] * num2;
          row.push(sum);
        }
        result.push(row);
      }
    }
    else if(ope === 'div'){
      for (var i = 0; i < array1.length; i++) {
        var row = [];
        for (var j = 0; j < array1[0].length; j++){
          var sum = array1[i][j] / num2;
          row.push(sum);
        }
        result.push(row);
      }
    }
  }

  return result;
};


export function opeArrays(array1, num2, ope) {

  var result = [];
  //num2如果是数组
  if(Array.isArray(num2)){
    if(ope === 'add'){
      for (var i = 0; i < array1.length; i++) {
        result.push(array1[i] + num2[i]);
      }
    }
    else if(ope === 'sub'){
      for (var i = 0; i < array1.length; i++) {
        result.push(array1[i] - num2[i]);
      }
    }
  }
  else{//num2不是数组
    if(ope === 'add'){
      for (var i = 0; i < array1.length; i++) {
        result.push(array1[i] + num2);
      }
    }
    else if(ope === 'sub'){
      for (var i = 0; i < array1.length; i++) {
        result.push(array1[i] - num2);
      }
    }
    else if(ope === 'mul'){
      for (var i = 0; i < array1.length; i++) {
        result.push(array1[i] * num2);
      }
    }
    else if(ope === 'div'){
      for (var i = 0; i < array1.length; i++) {
        result.push(array1[i] / num2);
      }
    }
  }

  return result;
};

/*
function scatter_mean(array1, array2) {
    var sums = {};
    var counts = {};
  
    for (var i = 0; i < array2.length; i++) {
      var index = array2[i];
      if (!(index in sums)) {
        sums[index] = [0, 0];
        counts[index] = 0;
      }
  
      sums[index][0] += array1[i][0];
      sums[index][1] += array1[i][1];
      counts[index]++;
    }
  
    var means = {};
    for (var index in sums) {
      means[index] = [sums[index][0] / counts[index], sums[index][1] / counts[index]];
    }
  
    return means;
}
  
export function center_pos(array1, array2) {
    var means = scatter_mean(array1, array2);
    var pos_center = [];
  
    for (var i = 0; i < array1.length; i++) {
      var index = array2[i];
      pos_center.push([array1[i][0] - means[index][0], array1[i][1] - means[index][1]]);
    }
  
    return pos_center;
}
*/
function scatter_mean(array1) {
  var sums = {};
  var counts = {};
  sums = [0, 0]
  for (var i = 0; i < array1.length; i++) {
    sums[0] += array1[i][0];
    sums[1] += array1[i][1];
  }

  var means = [0,0];
  means[0] = sums[0] / array1.length;
  means[1] = sums[1] / array1.length;

  return means;
}

export function center_pos(array1) {
  var means = scatter_mean(array1);
  var pos_center = [];

  for (var i = 0; i < array1.length; i++) {
    pos_center.push([array1[i][0] - means[0], array1[i][1] - means[1]]);
  }

  return pos_center;
}

function generateNoise(size) {
  var noise = new Array(size);
  
  for (var i = 0; i < size; i++) {
    noise[i] = Math.random(); // 生成一个介于 0 和 1 之间的随机数
  }
  
  return noise;
}
  
export function generateNormalDistribution(dims) {
    var size = dims[0]*dims[1];
    var noise = generateNoise(size);
  
    var sum = noise.reduce((a, b) => a + b, 0);
    var mean = sum / size;
  
    var squaredDiffSum = noise.reduce((a, b) => a + Math.pow(b - mean, 2), 0);
    var variance = squaredDiffSum / size;
  
    var stdDeviation = Math.sqrt(variance);
  
    var normalNoise = noise.map((value) => (value - mean) / stdDeviation);

    normalNoise = createNMArray(normalNoise, dims[0], dims[1])

    return normalNoise;
}


function generateNormalRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random(); // 防止 u 为 0
  while (v === 0) v = Math.random(); // 防止 v 为 0
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function randn_like(dims) {
  // 获取输入张量的形状
  const rows = dims[0];
  const cols = dims[1];


  // 创建与输入张量形状相同的张量
  let result = new Array(rows);
  for (let i = 0; i < rows; i++) {
    result[i] = new Array(cols);
    for (let j = 0; j < cols; j++) {
      result[i][j] = generateNormalRandom();
    }
  }

  return result;
}



  
  // 示例用法
  var pos = [[1, 2], [3, 4], [5, 6]];
  
  // var eps_linker = generateNormalDistribution([3, 2]);
  //console.log(eps_linker);


