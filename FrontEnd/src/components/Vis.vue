<template>
  <svg  class="canvas">
      <circle class="circle" 
        v-for="(position, index) in pos" 
          :key="index" 
          :cx="position[0] * 100 + '%'" 
          :cy="position[1] * 100 + '%'" 
          :r="cr"
      ></circle>
      <line class="line" 
        v-for="(edge, index) in all_edges" 
          :key="index" 
          :x1="pos[edge[0]][0] * 100+ '%'" 
          :y1="pos[edge[0]][1] * 100+ '%'" 
          :x2="pos[edge[1]][0] * 100+ '%'" 
          :y2="pos[edge[1]][1] * 100+ '%'"
      ></line>
  </svg>
</template>

<script>
/**
 * 根据pos和all_edges画出图
 */

export default {
  name: 'Vis',
  props: {
    pos: Array,
    all_edges: Array,
  },
  
  data() {
    return {
      cr:4,
      // all_edges: all_edges,
      // pos: pos,
    }
  },

}
</script>


<style scoped lang="less">
.canvas {
  width: 60%;
  height: 100%;
  .circle {
    fill: rgb(0, 0, 255);
  };
  .line {
    stroke: black;
    stroke-width: 1;
  };
}
</style>
