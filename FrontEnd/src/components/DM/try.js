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
    console.log(sums);
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

function center_pos(array1, array2) {
  var means = scatter_mean(array1, array2);
  var pos_center = [];

  for (var i = 0; i < array1.length; i++) {
    var index = array2[i];
    pos_center.push([array1[i][0] - means[index][0], array1[i][1] - means[index][1]]);
  }

  return pos_center;
}*/


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

function center_pos(array1) {
  var means = scatter_mean(array1);
  var pos_center = [];

  for (var i = 0; i < array1.length; i++) {
    pos_center.push([array1[i][0] - means[0], array1[i][1] - means[1]]);
  }

  return pos_center;
}


  // 示例用法
  var pos = [[1, 2], [3, 4], [5, 6]];
  var batch = [0, 0, 0];
  var data1 = scatter_mean(pos);
  console.log(data1);
  var pos_center = center_pos(pos);
  console.log(pos_center); // 输出 [[-2, -2], [0, 0], [2, 2]]