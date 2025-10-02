<script setup lang="ts">
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

function navigateTo(routeName: string) {
  router.push({ name: routeName })
}
</script>

<template>
  <div id="app">
    <header>
      <h1>🪲 Insect Trap Camera System</h1>
      <nav class="nav-bar">
        <button 
          @click="navigateTo('VideoCapture')" 
          class="nav-btn"
          :class="{ active: route.name === 'VideoCapture' }"
        >
          🎥 Video Capture
        </button>
        <button 
          @click="navigateTo('StillCapture')" 
          class="nav-btn"
          :class="{ active: route.name === 'StillCapture' }"
        >
          📷 Still Capture
        </button>
      </nav>
    </header>
    <main>
      <router-view />
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
  margin-bottom: 1rem;
}

.nav-bar {
  display: flex;
  justify-content: center;
  gap: 1rem;
}

.nav-btn {
  background: rgba(255, 255, 255, 0.1);
  color: white;
  border: 2px solid rgba(255, 255, 255, 0.3);
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);
}

.nav-btn:hover {
  background: rgba(255, 255, 255, 0.2);
  border-color: rgba(255, 255, 255, 0.5);
  transform: translateY(-2px);
}

.nav-btn.active {
  background: rgba(255, 255, 255, 0.25);
  border-color: white;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

main {
  padding: 1rem;
}
</style>
