<template>
  <div class="camera-controller">
    <!-- Left Panel - Controls -->
    <div class="left-panel">
      <!-- Camera Status Section -->
      <div class="control-section">
        <h3>📸 Camera Status</h3>
        <button 
          @click="checkCameras" 
          :disabled="isCheckingCameras"
          class="btn btn-primary full-width"
        >
          {{ isCheckingCameras ? 'Checking...' : 'Check Cameras' }}
        </button>
        
        <div v-if="cameraStatus" class="camera-status-compact">
          <div 
            v-for="(status, camera) in cameraStatus" 
            :key="camera"
            class="camera-status-item"
            :class="status.status"
          >
            <span class="camera-name">{{ formatCameraName(camera) }}</span>
            <span class="status-indicator">
              {{ status.status === 'ok' ? '✅' : '❌' }}
            </span>
          </div>
        </div>
      </div>

      <!-- Folder Selection Section -->
      <div class="control-section">
        <h3>📁 Destination</h3>
        <input 
          type="text" 
          v-model="destinationFolder" 
          placeholder="/home/traps/Desktop/2025-10-01"
          class="folder-path compact"
        >
        <button @click="setFolder" :disabled="!destinationFolder" class="btn btn-secondary full-width">
          Set Folder
        </button>
        <div v-if="folderStatus" class="status-message" :class="folderStatus.status">
          {{ folderStatus.message }}
        </div>
      </div>

      <!-- Recording Controls -->
      <div class="control-section">
        <h3>🎥 Recording</h3>
        
        <!-- Interval Slider -->
        <div class="control-group compact">
          <label>Interval: {{ captureInterval }}ms</label>
          <input 
            type="range" 
            v-model="captureInterval" 
            min="100" 
            max="10000" 
            step="100"
            class="slider"
          >
          <div class="preset-buttons">
            <button @click="captureInterval = 500" class="btn btn-mini">500ms</button>
            <button @click="captureInterval = 1000" class="btn btn-mini">1s</button>
            <button @click="captureInterval = 2000" class="btn btn-mini">2s</button>
            <button @click="captureInterval = 5000" class="btn btn-mini">5s</button>
          </div>
        </div>

        <!-- Resolution Selection -->
        <div class="control-group compact">
          <label>Resolution:</label>
          <select v-model="selectedResolution" class="select compact">
            <option value="small">Small</option>
            <option value="medium_4:3">Medium 4:3</option>
            <option value="medium_16:9">Medium 16:9</option>
            <option value="max">Max</option>
          </select>
        </div>

        <!-- Record Button -->
        <button 
          @click="toggleRecording" 
          :disabled="!currentFolder"
          class="btn btn-record full-width"
          :class="{ recording: isRecording }"
        >
          {{ isRecording ? '⏹️ Stop' : '🔴 Record' }}
        </button>
        
        <div v-if="!currentFolder" class="warning compact">
          ⚠️ Set folder first
        </div>
      </div>

      <!-- Recording Stats -->
      <div v-if="isRecording || captureCount > 0" class="control-section">
        <h3>📊 Status</h3>
        <div class="stats-compact">
          <div class="stat-item">
            <strong>Status:</strong> {{ isRecording ? '🔴 Recording' : '⏸️ Stopped' }}
          </div>
          <div class="stat-item">
            <strong>Captures:</strong> {{ captureCount }}
          </div>
          <div class="stat-item">
            <strong>Interval:</strong> {{ captureInterval }}ms
          </div>
        </div>
      </div>
    </div>

    <!-- Center Panel - Camera Mosaic -->
    <div class="center-panel">
      <div class="mosaic-container">
        <div class="camera-grid">
          <div 
            v-for="camIdx in [0, 1, 2, 3]" 
            :key="camIdx"
            class="camera-view"
          >
            <div class="camera-header">
              <h4>Camera {{ camIdx }}</h4>
              <span class="camera-counter" v-if="imageCounters[camIdx]">
                #{{ imageCounters[camIdx] }}
              </span>
            </div>
            <div class="image-container">
              <img 
                v-if="latestImages[`camera_${camIdx}`]" 
                :src="latestImages[`camera_${camIdx}`].url + '?t=' + refreshTimestamp"
                :alt="`Camera ${camIdx}`"
                class="camera-image"
                @error="handleImageError"
              />
              <div v-else class="no-image">
                <div class="placeholder-icon">📷</div>
                <div class="placeholder-text">No image yet</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right Panel - Image List -->
    <div class="right-panel">
      <div class="image-list-container">
        <h3>📋 Captured Images</h3>
        <div class="image-list" ref="imageListRef">
          <div 
            v-for="image in imageList" 
            :key="image.filename"
            class="image-item"
            @click="previewImage(image)"
          >
            <div class="image-thumbnail">
              <img 
                :src="image.url + '?t=' + refreshTimestamp" 
                :alt="image.filename"
                class="thumbnail"
                @error="handleImageError"
              />
            </div>
            <div class="image-info">
              <div class="filename">{{ image.filename }}</div>
              <div class="image-meta">
                <span class="size">{{ formatFileSize(image.size) }}</span>
                <span class="time">{{ formatTime(image.timestamp * 1000) }}</span>
              </div>
            </div>
          </div>
          <div v-if="imageList.length === 0" class="no-images">
            <div class="placeholder-icon">📁</div>
            <div class="placeholder-text">No images captured yet</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

// Reactive state
const isCheckingCameras = ref(false)
const cameraStatus = ref<any>(null)
const destinationFolder = ref('')
const folderStatus = ref<any>(null)
const currentFolder = ref('')
const captureInterval = ref(1000)
const selectedResolution = ref('medium_4:3')
const isRecording = ref(false)
const captureCount = ref(0)
const recentCaptures = ref<any[]>([])
const latestImages = ref<any>({})
const imageList = ref<any[]>([])
const imageCounters = ref<any>({})
const refreshTimestamp = ref(Date.now())

// Refs for DOM elements
const imageListRef = ref<HTMLElement>()

let recordingInterval: number | null = null
let imageUpdateInterval: number | null = null

// API base URL
const API_BASE = window.location.origin

// Initialize default folder
onMounted(() => {
  const today = new Date().toISOString().split('T')[0]
  destinationFolder.value = `/home/traps/Desktop/${today}`
  loadStatus()
  loadLatestImages()
  loadImageList()
  
  // Start periodic updates for images
  imageUpdateInterval = window.setInterval(() => {
    if (currentFolder.value) {
      loadLatestImages()
      loadImageList()
    }
  }, 2000) // Update every 2 seconds
})

onUnmounted(() => {
  if (recordingInterval) {
    clearInterval(recordingInterval)
  }
  if (imageUpdateInterval) {
    clearInterval(imageUpdateInterval)
  }
})

// Camera checking
async function checkCameras() {
  isCheckingCameras.value = true
  cameraStatus.value = null
  
  try {
    const response = await fetch(`${API_BASE}/api/check-cameras`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    })
    
    if (response.ok) {
      cameraStatus.value = await response.json()
    } else {
      throw new Error('Failed to check cameras')
    }
  } catch (error) {
    console.error('Error checking cameras:', error)
    cameraStatus.value = {
      error: { status: 'error', message: 'Failed to connect to server' }
    }
  } finally {
    isCheckingCameras.value = false
  }
}

// Folder management
async function setFolder() {
  folderStatus.value = null
  
  try {
    const response = await fetch(`${API_BASE}/api/set-folder`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        folder_path: destinationFolder.value
      })
    })
    
    const result = await response.json()
    
    if (response.ok) {
      folderStatus.value = { status: 'ok', message: `✅ Folder set: ${result.folder}` }
      currentFolder.value = result.folder
      captureCount.value = 0
      recentCaptures.value = []
    } else {
      folderStatus.value = { status: 'error', message: `❌ ${result.message}` }
    }
  } catch (error) {
    console.error('Error setting folder:', error)
    folderStatus.value = { status: 'error', message: '❌ Failed to connect to server' }
  }
}

// Recording control
function toggleRecording() {
  if (isRecording.value) {
    stopRecording()
  } else {
    startRecording()
  }
}

function startRecording() {
  if (!currentFolder.value) {
    alert('Please set a destination folder first')
    return
  }
  
  isRecording.value = true
  recordingInterval = window.setInterval(captureImages, captureInterval.value)
}

function stopRecording() {
  isRecording.value = false
  if (recordingInterval) {
    clearInterval(recordingInterval)
    recordingInterval = null
  }
}

// Image capture
async function captureImages() {
  try {
    const response = await fetch(`${API_BASE}/api/capture-images`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        resolution: selectedResolution.value
      })
    })
    
    if (response.ok) {
      const result = await response.json()
      captureCount.value++
      
      // Update image counters
      for (const [camera, data] of Object.entries(result)) {
        if ((data as any).status === 'ok') {
          const camIdx = camera.replace('camera_', '')
          imageCounters.value[camIdx] = (data as any).counter
        }
      }
      
      // Count successful captures
      const successCount = Object.values(result).filter((r: any) => r.status === 'ok').length
      
      recentCaptures.value.push({
        timestamp: Date.now(),
        success: successCount,
        total: 4,
        result
      })
      
      // Immediately refresh images
      setTimeout(() => {
        loadLatestImages()
        loadImageList()
      }, 500) // Small delay to ensure files are written
      
      console.log(`Capture ${captureCount.value}: ${successCount}/4 cameras successful`)
    } else {
      console.error('Failed to capture images')
    }
  } catch (error) {
    console.error('Error capturing images:', error)
  }
}

// Load current status
async function loadStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/get-status`)
    if (response.ok) {
      const status = await response.json()
      if (status.current_folder) {
        currentFolder.value = status.current_folder
        destinationFolder.value = status.current_folder
      }
      if (status.image_counters) {
        imageCounters.value = status.image_counters
        captureCount.value = Math.max(...Object.values(status.image_counters) as number[])
      }
    }
  } catch (error) {
    console.error('Error loading status:', error)
  }
}

// Load latest images for mosaic
async function loadLatestImages() {
  try {
    const response = await fetch(`${API_BASE}/api/get-latest-images`)
    if (response.ok) {
      latestImages.value = await response.json()
      refreshTimestamp.value = Date.now()
    }
  } catch (error) {
    console.error('Error loading latest images:', error)
  }
}

// Load image list for right panel
async function loadImageList() {
  try {
    const response = await fetch(`${API_BASE}/api/get-image-list`)
    if (response.ok) {
      const newImageList = await response.json()
      
      // Auto-scroll to top when new images are added
      if (newImageList.length > imageList.value.length && imageListRef.value) {
        setTimeout(() => {
          imageListRef.value?.scrollTo({ top: 0, behavior: 'smooth' })
        }, 100)
      }
      
      imageList.value = newImageList
    }
  } catch (error) {
    console.error('Error loading image list:', error)
  }
}

// Handle image preview (could be expanded later)
function previewImage(image: any) {
  // For now, just open in new tab
  window.open(image.url, '_blank')
}

// Handle image loading errors
function handleImageError(event: Event) {
  const img = event.target as HTMLImageElement
  img.style.display = 'none'
}

// Utility functions
function formatCameraName(camera: string | number): string {
  return String(camera).replace('camera_', 'Camera ')
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString()
}
</script>

<style scoped>
.camera-controller {
  display: grid;
  grid-template-columns: 280px 1fr 250px;
  height: calc(100vh - 80px);
  gap: 0;
  margin: 0;
  padding: 0;
}

/* Left Panel - Controls */
.left-panel {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.control-section {
  border-bottom: 1px solid #eee;
  padding-bottom: 1.5rem;
}

.control-section:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

.control-section h3 {
  margin-bottom: 1rem;
  color: #333;
  font-size: 1.1rem;
  font-weight: 600;
}

/* Buttons */
.btn {
  padding: 0.6rem 1.2rem;
  border: none;
  border-radius: 8px;
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  text-decoration: none;
  display: inline-block;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn.full-width {
  width: 100%;
  margin-bottom: 0.5rem;
}

.btn-primary {
  background: #667eea;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #5a6fd8;
}

.btn-secondary {
  background: #6c757d;
  color: white;
}

.btn-secondary:hover:not(:disabled) {
  background: #5a6268;
}

.btn-mini {
  padding: 0.3rem 0.6rem;
  font-size: 0.75rem;
  margin: 0.1rem;
}

.btn-record {
  background: #dc3545 !important;
  color: white !important;
  font-size: 1rem;
  padding: 0.8rem 1.5rem;
  border: none !important;
}

.btn-record:hover:not(:disabled) {
  background: #c82333 !important;
  color: white !important;
}

.btn-record.recording {
  background: #28a745 !important;
  color: white !important;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}

/* Camera Status Compact */
.camera-status-compact {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.camera-status-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem;
  border-radius: 6px;
  background: #f8f9fa;
  border: 1px solid transparent;
  font-size: 0.8rem;
}

.camera-status-item.ok {
  border-color: #28a745;
  background: #d4edda;
}

.camera-status-item.error {
  border-color: #dc3545;
  background: #f8d7da;
}

.camera-name {
  font-weight: 600;
  font-size: 0.75rem;
}

/* Compact Controls */
.folder-path.compact {
  width: 100%;
  padding: 0.6rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 0.85rem;
  margin-bottom: 0.5rem;
}

.status-message {
  padding: 0.5rem;
  border-radius: 6px;
  font-size: 0.8rem;
  margin-top: 0.5rem;
}

.status-message.ok {
  background: #d4edda;
  color: #155724;
}

.status-message.error {
  background: #f8d7da;
  color: #721c24;
}

.control-group.compact {
  margin-bottom: 1rem;
}

.control-group.compact label {
  display: block;
  margin-bottom: 0.3rem;
  font-weight: 500;
  color: #333;
  font-size: 0.85rem;
}

.slider {
  width: 100%;
  height: 4px;
  border-radius: 2px;
  background: #ddd;
  outline: none;
  margin-bottom: 0.5rem;
}

.preset-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.2rem;
}

.select.compact {
  padding: 0.6rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 0.85rem;
  background: white;
  width: 100%;
}

.warning.compact {
  color: #856404;
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  padding: 0.5rem;
  border-radius: 6px;
  font-size: 0.8rem;
  text-align: center;
  margin-top: 0.5rem;
}

.stats-compact {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.stat-item {
  padding: 0.5rem;
  background: #f8f9fa;
  border-radius: 6px;
  font-size: 0.8rem;
}

/* Center Panel - Camera Mosaic */
.center-panel {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.mosaic-container {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.camera-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 1rem;
  height: 100%;
}

.camera-view {
  display: flex;
  flex-direction: column;
  border: 2px solid #e9ecef;
  border-radius: 12px;
  overflow: hidden;
  background: #f8f9fa;
}

.camera-header {
  background: #667eea;
  color: white;
  padding: 0.75rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.camera-header h4 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
}

.camera-counter {
  background: rgba(255, 255, 255, 0.2);
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
}

.image-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #000;
  position: relative;
}

.camera-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: 0;
}

.no-image {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #666;
  height: 100%;
}

.placeholder-icon {
  font-size: 3rem;
  margin-bottom: 0.5rem;
  opacity: 0.5;
}

.placeholder-text {
  font-size: 0.9rem;
  opacity: 0.7;
}

/* Right Panel - Image List */
.right-panel {
  background: white;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.image-list-container {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.image-list-container h3 {
  margin-bottom: 1rem;
  color: #333;
  font-size: 1.1rem;
  font-weight: 600;
}

.image-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.image-item {
  display: flex;
  gap: 0.75rem;
  padding: 0.75rem;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  background: #fafafa;
}

.image-item:hover {
  background: #f0f8ff;
  border-color: #667eea;
  transform: translateY(-1px);
}

.image-thumbnail {
  width: 60px;
  height: 45px;
  border-radius: 4px;
  overflow: hidden;
  background: #000;
  flex-shrink: 0;
}

.thumbnail {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.filename {
  font-weight: 500;
  font-size: 0.8rem;
  color: #333;
  margin-bottom: 0.2rem;
  word-break: break-all;
}

.image-meta {
  display: flex;
  justify-content: space-between;
  font-size: 0.7rem;
  color: #666;
}

.no-images {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #666;
  height: 200px;
}

/* Responsive */
@media (max-width: 1200px) {
  .camera-controller {
    grid-template-columns: 250px 1fr 220px;
  }
}

@media (max-width: 992px) {
  .camera-controller {
    grid-template-columns: 1fr;
    grid-template-rows: auto auto auto;
    height: auto;
    gap: 1rem;
  }
  
  .left-panel, .right-panel {
    max-height: 400px;
  }
  
  .center-panel {
    min-height: 500px;
  }
  
  .camera-grid {
    min-height: 400px;
  }
}

@media (max-width: 768px) {
  .camera-controller {
    padding: 0.5rem;
    gap: 0.5rem;
  }
  
  .left-panel,
  .center-panel,
  .right-panel {
    padding: 1rem;
  }
  
  .camera-grid {
    gap: 0.5rem;
  }
  
  .preset-buttons {
    grid-template-columns: 1fr 1fr 1fr 1fr;
  }
  
  .camera-status-compact {
    grid-template-columns: 1fr;
  }
}
</style>