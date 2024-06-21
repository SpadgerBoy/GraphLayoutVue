function scatter_mean(array1, array2, dim) {
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
  
  function center_pos(array1, array2) {
    var means = scatter_mean(array1, array2, 0);
    var pos_center = [];
  
    for (var i = 0; i < array1.length; i++) {
      var index = array2[i];
      pos_center.push([array1[i][0] - means[index][0], array1[i][1] - means[index][1]]);
    }
  
    return pos_center;
  }
  
  // 示例用法
  var pos = [[1, 2], [3, 4], [5, 6]];
  var batch = [0, 1, 0];
  
  var pos_center = center_pos(pos, batch);
  //console.log(pos_center); // 输出 [[-2, -2], [0, 0], [2, 2]]

  var seq = []
  for (var i = 499; i >= 0; i--) {
    seq.push(i);
  }
  var seq_next = seq.slice(1, seq.length).concat([-1]);
  console.log(seq)
  console.log(seq_next)
