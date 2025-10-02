import { createRouter, createWebHistory } from 'vue-router'
import StillCapture from '../views/StillCapture.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'StillCapture',
      component: StillCapture
    }
  ]
})

export default router