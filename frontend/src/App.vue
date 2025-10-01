<script setup lang="ts">
import CameraController from './components/CameraController.vue'
</script>

<template>
  <div id="app">
    <header>
      <h1>🪲 Insect Trap Camera System</h1>
    </header>
    <main>
      <CameraController />
    </main>
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.6;
  color: #333;
  background: white;
  min-height: 100vh;
}

header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem;
  text-align: center;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

header h1 {
  font-size: 2rem;
  font-weight: 600;
}

main {
  padding: 0;
}
</style>
