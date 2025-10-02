<template>
  <div class="still-capture">
    <!-- Left Panel - Controls -->
    <div class="left-panel">
      <!-- Capture Controls -->
      <div class="control-section">
        <h3>📸 Still Capture</h3>
        
        <!-- Wait Time Slider -->
        <div class="control-group compact">
          <label>Wait Time: {{ waitTime }}ms</label>
          <input 
            type="range" 
            v-model="waitTime" 
            min="100" 
            max="5000" 
            step="50"
            class="slider"
          >
          <div class="preset-buttons">
            <button @click="waitTime = 100" class="btn btn-mini">100ms</button>
            <button @click="waitTime = 500" class="btn btn-mini">500ms</button>
            <button @click="waitTime = 1000" class="btn btn-mini">1s</button>
            <button @click="waitTime = 2000" class="btn btn-mini">2s</button>
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

        <!-- Capture Button -->
        <button 
          @click="captureImages" 
          :disabled="isCapturing || !currentFolder"
          class="btn btn-capture full-width"
          :class="{ capturing: isCapturing }"
        >
          {{ isCapturing ? '⏳ Capturing...' : '📸 Capture Now' }}
        </button>
        
        <div v-if="!currentFolder" class="warning compact">
          ⚠️ Set folder first
        </div>
        
        <!-- Success notification -->
        <div v-if="showSuccessNotification" class="success-toast">
          ✅ Capture completed! {{ lastCaptureResult?.success || 0 }}/4 cameras successful
        </div>
      </div>

      <!-- Folder Selection Section -->
      <div class="control-section">
        <h3>📁 Destination</h3>
        <input 
          type="text" 
          v-model="destinationFolder" 
          placeholder="/home/traps/Desktop/2025-10-02"
          class="folder-path compact"
        >
        <button @click="setFolder" :disabled="!destinationFolder" class="btn btn-secondary full-width">
          Set Folder
        </button>
        <div v-if="folderStatus" class="status-message" :class="folderStatus.status">
          {{ folderStatus.message }}
        </div>
      </div>

      <!-- Camera Status Section -->
      <div class="control-section">
        <h3>📷 Camera Status</h3>
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

      <!-- Capture Stats -->
      <div v-if="captureCount > 0" class="control-section">
        <h3>📊 Status</h3>
        <div class="stats-compact">
          <div class="stat-item">
            <strong>Total Captures:</strong> {{ captureCount }}
          </div>
          <div class="stat-item">
            <strong>Wait Time:</strong> {{ waitTime }}ms
          </div>
          <div v-if="lastCaptureResult" class="stat-item">
            <strong>Last Result:</strong> {{ lastCaptureResult.success }}/{{ lastCaptureResult.total }} cameras
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
              <span class="camera-counter" v-if="imageCounters[camIdx] > 0">
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
            v-for="image in sortedImageList" 
            :key="image.filename"
            class="file-item"
            @click="previewImage(image)"
          >
            <div class="file-info">
              <div class="file-header">
                <span class="acquisition-number">#{{ getAcquisitionNumber(image.filename) }}</span>
                <span class="camera-badge">{{ getCameraFromFilename(image.filename) }}</span>
              </div>
              <div class="filename">{{ image.filename }}</div>
              <div class="file-meta">
                <span class="time">{{ formatTime(image.timestamp * 1000) }}</span>
                <span class="size">{{ formatFileSize(image.size) }}</span>
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
import { ref, computed, onMounted, onUnmounted } from 'vue'

// Reactive state
const isCheckingCameras = ref(false)
const cameraStatus = ref<any>(null)
const destinationFolder = ref('')
const folderStatus = ref<any>(null)
const currentFolder = ref('')
const waitTime = ref(100)
const selectedResolution = ref('max')
const isCapturing = ref(false)
const captureCount = ref(0)
const latestImages = ref<any>({})
const imageList = ref<any[]>([])
const imageCounters = ref<any>({})
const refreshTimestamp = ref(Date.now())
const lastCaptureResult = ref<any>(null)
const showSuccessNotification = ref(false)

// Refs for DOM elements
const imageListRef = ref<HTMLElement>()

let imageUpdateInterval: number | null = null

// API base URL
const API_BASE = window.location.origin

// Computed property for sorted image list
const sortedImageList = computed(() => {
  return [...imageList.value].sort((a, b) => {
    const aNumber = getAcquisitionNumber(a.filename)
    const bNumber = getAcquisitionNumber(b.filename)
    const aCamera = getCameraNumber(a.filename)
    const bCamera = getCameraNumber(b.filename)
    
    // First sort by acquisition number (descending - most recent first)
    if (aNumber !== bNumber) {
      return bNumber - aNumber
    }
    
    // If same acquisition number, sort by camera number
    return aCamera - bCamera
  })
})

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
      
      // Update image counters from backend
      if (result.image_counters) {
        imageCounters.value = result.image_counters
        // Set capture count to the maximum counter value across all cameras
        const maxCounter = Math.max(...Object.values(result.image_counters) as number[])
        captureCount.value = maxCounter
        console.log(`Folder set with existing counters:`, result.image_counters, `Max: ${maxCounter}`)
      } else {
        // Reset counters if no existing images
        imageCounters.value = { '0': 0, '1': 0, '2': 0, '3': 0 }
        captureCount.value = 0
      }
    } else {
      folderStatus.value = { status: 'error', message: `❌ ${result.message}` }
    }
  } catch (error) {
    console.error('Error setting folder:', error)
    folderStatus.value = { status: 'error', message: '❌ Failed to connect to server' }
  }
}

// Image capture
async function captureImages() {
  if (isCapturing.value) return
  
  isCapturing.value = true
  
  try {
    // Wait for the specified time
    await new Promise(resolve => setTimeout(resolve, waitTime.value))
    
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
      
      // Update image counters from the backend response
      for (const [camera, data] of Object.entries(result)) {
        if ((data as any).status === 'ok') {
          const camIdx = camera.replace('camera_', '')
          imageCounters.value[camIdx] = (data as any).counter
        }
      }
      
      // Update capture count to match the highest counter
      const maxCounter = Math.max(...Object.values(imageCounters.value) as number[])
      captureCount.value = maxCounter
      
      // Count successful captures
      const successCount = Object.values(result).filter((r: any) => r.status === 'ok').length
      
      lastCaptureResult.value = {
        success: successCount,
        total: 4,
        result
      }
      
      // Show success notification
      showSuccessNotification.value = true
      setTimeout(() => {
        showSuccessNotification.value = false
      }, 3000) // Hide after 3 seconds
      
      // Immediately refresh images
      setTimeout(() => {
        loadLatestImages()
        loadImageList()
      }, 500) // Small delay to ensure files are written
      
      console.log(`Capture completed - Total captures: ${captureCount.value}, Success: ${successCount}/4 cameras`)
    } else {
      console.error('Failed to capture images')
    }
  } catch (error) {
    console.error('Error capturing images:', error)
  } finally {
    isCapturing.value = false
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
        const maxCounter = Math.max(...Object.values(status.image_counters) as number[])
        captureCount.value = maxCounter
        console.log(`Status loaded with counters:`, status.image_counters, `Max: ${maxCounter}`)
      } else {
        // Initialize counters if not provided
        imageCounters.value = { '0': 0, '1': 0, '2': 0, '3': 0 }
        captureCount.value = 0
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

function getAcquisitionNumber(filename: string): number {
  // Extract number from filename like IMG_CAM0_000123.jpg
  const match = filename.match(/IMG_CAM\d+_(\d+)\.jpg$/)
  return match && match[1] ? parseInt(match[1], 10) : 0
}

function getCameraNumber(filename: string): number {
  // Extract camera number from filename like IMG_CAM0_000123.jpg
  const match = filename.match(/IMG_CAM(\d+)_\d+\.jpg$/)
  return match && match[1] ? parseInt(match[1], 10) : 0
}

function getCameraFromFilename(filename: string): string {
  // Extract camera name from filename like IMG_CAM0_000123.jpg
  const cameraNum = getCameraNumber(filename)
  return `CAM${cameraNum}`
}
</script>

<style scoped>
.still-capture {
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

/* Success toast notification */
.success-toast {
  background: #d4edda;
  color: #155724;
  padding: 0.75rem;
  border-radius: 6px;
  font-size: 0.85rem;
  margin-top: 0.5rem;
  border: 1px solid #c3e6cb;
  animation: slideIn 0.3s ease-in-out;
}

@keyframes slideIn {
  from { 
    opacity: 0; 
    transform: translateY(-10px); 
  }
  to { 
    opacity: 1; 
    transform: translateY(0); 
  }
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
  background: #27ae60;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: #219a52;
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

.btn-capture {
  background: #27ae60 !important;
  color: white !important;
  font-size: 1rem;
  padding: 0.8rem 1.5rem;
  border: none !important;
}

.btn-capture:hover:not(:disabled) {
  background: #219a52 !important;
  color: white !important;
}

.btn-capture.capturing {
  background: #f39c12 !important;
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
  grid-template-columns: 1fr 1fr 1fr 1fr;
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
  background: #27ae60;
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

.file-item {
  padding: 0.75rem;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  background: #fafafa;
}

.file-item:hover {
  background: #e8f5e8;
  border-color: #27ae60;
  transform: translateY(-1px);
}

.file-info {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.file-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.acquisition-number {
  font-weight: 700;
  font-size: 0.9rem;
  color: #27ae60;
  background: #e8f5e8;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
}

.camera-badge {
  font-size: 0.7rem;
  font-weight: 600;
  color: #666;
  background: #f8f9fa;
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  border: 1px solid #dee2e6;
}

.filename {
  font-weight: 500;
  font-size: 0.75rem;
  color: #333;
  word-break: break-all;
  opacity: 0.8;
}

.file-meta {
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
  .still-capture {
    grid-template-columns: 250px 1fr 220px;
  }
}

@media (max-width: 992px) {
  .still-capture {
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
  .still-capture {
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
    grid-template-columns: 1fr 1fr;
  }
  
  .camera-status-compact {
    grid-template-columns: 1fr;
  }
}
</style>