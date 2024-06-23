import {createNMArray, createArray} from "./tools.js";

function assemble_node_pair_feature(node_attr, edge_index, edge_attr) {

  var h_row = [];
  var h_col = [];

  for (var i = 0; i < edge_index[0].length; i++) {
    var rowIndex = edge_index[0][i];
    var colIndex = edge_index[1][i];

    h_row.push(node_attr[rowIndex]);
    h_col.push(node_attr[colIndex]);
  }

  var h_pair = [];
  for (var i = 0; i < h_row.length; i++) {
    //console.log(h_row.length, h_col.length, edge_attr[i].length)
    var temp = h_row[i].concat(h_col[i], edge_attr[i]);
    h_pair.push(temp);
  }
  //console.log(h_pair[0])  
  var h_pair_local = createArray(h_pair);
  return h_pair_local;
};

function assemble_node_pair_feature_1(node_attr0, edge_index0, edge_attr0, dims) {

  const node_attr = createNMArray(node_attr0, dims[0][0], dims[0][1])
  const edge_index = createNMArray(edge_index0, dims[1][0], dims[1][1])
  const edge_attr = createNMArray(edge_attr0, dims[2][0], dims[2][1])

  var h_row = [];
  var h_col = [];

  for (var i = 0; i < edge_index[0].length; i++) {
    var rowIndex = edge_index[0][i];
    var colIndex = edge_index[1][i];

    h_row.push(node_attr[rowIndex]);
    h_col.push(node_attr[colIndex]);
  }

  var h_pair = [];
  for (var i = 0; i < h_row.length; i++) {
    var temp = h_row[i].concat(h_col[i], edge_attr[i]);
    h_pair.push(temp);
  }

  var h_pair_local = createArray(h_pair);

  return h_pair_local;
};





export default {
  assemble_node_pair_feature:assemble_node_pair_feature,
  assemble_node_pair_feature_1:assemble_node_pair_feature_1,
}


