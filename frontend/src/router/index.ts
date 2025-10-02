import { createRouter, createWebHistory } from 'vue-router'
import VideoCapture from '../views/VideoCapture.vue'
import StillCapture from '../views/StillCapture.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      redirect: '/still'
    },
    {
      path: '/video',
      name: 'VideoCapture',
      component: VideoCapture
    },
    {
      path: '/still',
      name: 'StillCapture',
      component: StillCapture
    }
  ]
})

export default router