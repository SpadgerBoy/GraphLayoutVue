<template>
  <div class="content-container">
      <Vis></Vis>
  </div>
</template>

<script>
import Vis from './components/Vis.vue'
export default {
  name: 'App',
  components: {
    Vis
  }
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
    width: 100vw; /* viewport width, makes it full width of the screen */
    height: 100vh; /* viewport height, makes it full height of the screen */
    justify-content: center; /* centers content horizontally */
    align-items: center; /* centers content vertically */
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
}
</style>
