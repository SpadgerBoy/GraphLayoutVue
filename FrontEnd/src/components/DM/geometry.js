//import common from './common.js';
import {createNMArray, createArray} from './tools.js';

//var x = [[0.2, 0.1], [0.6, 0.2], [0.3, 0.8]]
//var y = [[0, 0, 1], [1, 1, 2]]

//var z = get_distance(x, y)
//console.log(z)

export function get_distance(pos, edge_index) {
    var distances = [];

    for (var i = 0; i < edge_index[0].length; i++) {
      var aIndex = edge_index[0][i];
      var bIndex = edge_index[1][i];
  
      var dis = 0;
      for (var j = 0; j < pos[0].length; j++) {
        var temp = (pos[aIndex][j] - pos[bIndex][j]);
        dis += temp**2;
      }
      dis = Math.sqrt(dis);
      distances.push(dis);
    }
  
    return distances;
  };


export function eq_transform(score_d0, pos0, edge_index0, edge_length0, dims) {

    var score_d = createNMArray(score_d0, dims[0][0], dims[0][1]);
    var pos = createNMArray(pos0, dims[1][0], dims[1][1]);
    var edge_index = createNMArray(edge_index0, dims[2][0], dims[2][1]);
    var edge_length = createNMArray(edge_length0, dims[3][0], dims[3][1]);

    var N = pos.length;
    var dd_dr = [];

    for (var i = 0; i < edge_index[0].length; i++) {
      var aIndex = edge_index[0][i];
      var bIndex = edge_index[1][i];
  
      var diff = [];
      for (var j = 0; j < pos[0].length; j++) {
        var temp = (pos[aIndex][j] - pos[bIndex][j]) / edge_length[i];
        diff.push(temp);
      }
      dd_dr.push(diff);
    }

    var score_pos = new Array(N).fill(0).map(() => new Array(pos[0].length).fill(0));
  
    for (var i = 0; i < edge_index[0].length; i++) {
      var rowIndex = edge_index[0][i];
      var colIndex = edge_index[1][i];
  
      for (var j = 0; j < pos[0].length; j++) {
        score_pos[rowIndex][j] += dd_dr[i][j] * score_d[i];
        score_pos[colIndex][j] -= dd_dr[i][j] * score_d[i];
      }
    }
    var score_pos1 = createArray(score_pos)
    return score_pos1;
};
