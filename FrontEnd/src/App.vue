<template>
    <div class="content-container">
      <Vis v-if="isDataReady"
        :pos="pos"
        :all_edges="all_edges"
      ></Vis>
      <div class="loading-container" v-else>
        Infering...
      </div>
    </div>
</template>

<script>

/**
 * 从pubilc/all_edges.txt读取图的拓扑结构，包括节点数量N和边all_edges
 * 然后将N和all_edges传入run.js运算得到层次布局下每个结点的坐标pos
 * 并将pos与all_edges传入Vis.vue绘图
 */

import axios from 'axios';
import get_new_pos from "./components/run.js";
import Vis from './components/Vis.vue'

export default {
  name: 'App',
  components: {
    Vis
  },

  data() {
    return {
      pos: null,
      all_edges:[],
      // all_edges:all_edges,
      isDataReady: false,
    }
  },

  async created() {
    try {
      const response = await axios.get('./all_edges.txt');
      console.log("Response Data:", response.data);

      const dataText = response.data.trim();
      const [firstPart, edgesPart] = dataText.split(':');

      const N = parseInt(firstPart.trim(), 10); //10进制转换
      this.all_edges = eval(edgesPart.trim());  //将字符串转换为数组

      console.log('N:',N);
      console.log('all_edges_num:', this.all_edges.length)
      

      if (Number.isInteger(N) && this.all_edges.every(edgeList => Array.isArray(edgeList) && edgeList.every(Number.isInteger))) {
        const pos = await get_new_pos(N, this.all_edges);
        this.pos = pos;
        this.isDataReady = true; 
      } else {
        console.error('Invalid data format for N or edges.');
      }
    } catch (error) {
      console.error("Error loading graph data:", error);
    }

  },

}
</script>

<style lang="less" scoped>

</style>

<style lang="less">
html {
  font-size: 100%;
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  position: absolute;
  top: 0%;
  bottom: 0%;
  left: 0%;
  right: 0%;

  .content-container {
    width: 100vw; 
    height: 100vh; 
    justify-content: center; 
    align-items: center; 
    position: absolute;
    top: 0%;
    left: 0%;
    bottom: 0%;
    right: 0%;
  }
  .Vis {
    width: 100%;
    height: 100%;
  }

  .loading-container {

    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    font-size: 32px;
  }
}
</style>
