/* 
 *矩阵运算所需的工具 
 */

//将两个矩阵拼接
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



//矩阵的运算
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

//将pos均值中心化
export function center_pos(array1) {
  var means = scatter_mean(array1);
  var pos_center = [];

  for (var i = 0; i < array1.length; i++) {
    pos_center.push([array1[i][0] - means[0], array1[i][1] - means[1]]);
  }

  return pos_center;
}





function generateNormalRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random(); // 防止 u 为 0
  while (v === 0) v = Math.random(); // 防止 v 为 0
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}


//生成正态分布的随机数
export function randn_like(dims) {

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


