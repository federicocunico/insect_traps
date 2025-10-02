# Insect Trap Camera System 🪲

A multi-camera system for insect trap monitoring using Raspberry Pi and multiple cameras.

## Raspberry Pi Setup Guide

This section covers the complete setup of a Raspberry Pi for multi-camera functionality. Follow these steps carefully for proper hardware configuration.

### 1. Initial Raspberry Pi Configuration

First, update your system and enable necessary interfaces:

```bash
# Update the system
sudo apt update && sudo apt upgrade -y

# Run Raspberry Pi configuration tool
sudo raspi-config
```

In `raspi-config`, configure the following:
- **Interface Options → Camera**: Enable camera interface
- **Interface Options → I2C**: Enable I2C interface (required for multiple cameras)
- **Interface Options → SPI**: Enable SPI interface (if needed)
- **Advanced Options → Expand Filesystem**: Ensure full SD card usage
- **System Options → Boot Options**: Set to desktop autologin (optional)

Reboot after making these changes:
```bash
sudo reboot
```

### 2. Install Required System Packages

Install essential packages for camera functionality:

```bash
# Install camera and I2C tools (rpicam tools should be pre-installed)
sudo apt install -y \
    python3-picamera2 \
    python3-libcamera \
    python3-kms++ \
    python3-pyqt5 \
    python3-prctl \
    libatlas-base-dev \
    i2c-tools \
    python3-smbus \
    git \
    curl \
    build-essential

# Verify rpicam tools are available
rpicam-hello --version

# Verify I2C is working
sudo i2cdetect -y 1
```

### 3. Multi-Camera Hardware Setup

For this specific multi-camera setup, you'll need:
- **Raspberry Pi 4** (required for multi-camera support)
- **4x IMX477 cameras** (12MP high-quality camera sensors)
- **4-port CSI camera multiplexer board** (e.g., Arducam Multi-Camera Adapter)
- **Proper I2C addressing** for camera identification

#### IMX477 Camera Setup with 4-Port Multiplexer:

The system uses **Sony IMX477 sensors** (12.3MP, 1/2.3" format) with the following specifications:
- **Resolution**: 4056 × 3040 pixels (12MP)
- **Sensor Size**: 1/2.3" (7.9mm diagonal) 
- **Pixel Size**: 1.55μm × 1.55μm
- **Available Modes**:
  - `4056x3040` @ 10fps (full resolution)
  - `2028x1520` @ 40fps (2x2 binned)
  - `2028x1080` @ 50fps (cropped)
  - `1332x990` @ 120fps (4x4 binned, cropped)

```bash
# Check camera detection via I2C
sudo i2cdetect -y 1
# Should show devices at addresses 0x1a on buses 23, 24, 25, 26

# Verify camera overlay is loaded
dmesg | grep imx477
# Should show "Device found is imx477" for each camera

# Test individual cameras using rpicam tools
# Note: Camera indices may vary based on multiplexer configuration
rpicam-still --camera 0 -o test_cam0.jpg --immediate
rpicam-still --camera 1 -o test_cam1.jpg --immediate
rpicam-still --camera 2 -o test_cam2.jpg --immediate  
rpicam-still --camera 3 -o test_cam3.jpg --immediate
```

#### Camera Permissions:
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Create camera access rules
echo 'SUBSYSTEM=="video4linux", GROUP="video", MODE="0664"' | sudo tee /etc/udev/rules.d/99-camera.rules

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 4. Install Node.js and NVM

Install Node Version Manager (NVM) and Node.js LTS:

```bash
# Install NVM
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash

# Reload shell configuration
source ~/.bashrc

# Install Node.js LTS (currently v20.x)
nvm install --lts
nvm use --lts
nvm alias default node

# Verify installation
node --version  # Should show v20.x.x or later
npm --version   # Should show v10.x.x or later
```

### 5. Install Python Dependencies

Set up Python environment:

```bash
# Install pip if not already installed
sudo apt install -y python3-pip python3-venv

# Install system-wide Python packages for camera support
sudo apt install -y \
    python3-opencv \
    python3-numpy \
    python3-pillow
```

### 6. Camera Configuration File

Create a camera configuration file for your setup:

```bash
# Create config directory
mkdir -p ~/.config/camera_system

# Create camera config for IMX477 setup
cat > ~/.config/camera_system/cameras.conf << 'EOF'
# Camera configuration for IMX477 cameras with multiplexer
# Format: camera_id:i2c_address:name
0:23-001a:Camera 0 (IMX477)
1:24-001a:Camera 1 (IMX477)  
2:25-001a:Camera 2 (IMX477)
3:26-001a:Camera 3 (IMX477)
EOF
```

### 7. Performance Optimization

Optimize your Pi for camera operations:

```bash
# Increase GPU memory split
echo 'gpu_mem=128' | sudo tee -a /boot/firmware/config.txt

# Optimize camera settings (disable auto-detect for manual config)
echo 'camera_auto_detect=0' | sudo tee -a /boot/firmware/config.txt

# Configure 4-port camera multiplexer with IMX477 cameras
echo 'dtoverlay=camera-mux-4port,cam0-imx477,cam1-imx477,cam2-imx477,cam3-imx477' | sudo tee -a /boot/firmware/config.txt

# Increase USB current (for USB cameras)
echo 'max_usb_current=1' | sudo tee -a /boot/firmware/config.txt

# Reboot to apply changes
sudo reboot
```

### 8. Test Multi-Camera Setup

After reboot, test your camera setup:

```bash
# Verify all 4 IMX477 cameras are detected
dmesg | grep "Device found is imx477"
# Should show 4 entries (one for each camera)

# Check I2C camera addresses
sudo i2cdetect -y 1
# Should show devices at 0x1a (cameras) and 0x70 (multiplexer)

# Test capture from all cameras simultaneously
for i in {0..3}; do
    echo "Testing camera $i..."
    rpicam-still --camera $i -o test_imx477_camera_$i.jpg --immediate &
done
wait

# Verify images were captured successfully
ls -la test_imx477_camera_*.jpg
file test_imx477_camera_*.jpg  # Should show JPEG image data

# Check image dimensions (should be 4056x3040 for full resolution)
identify test_imx477_camera_*.jpg 2>/dev/null || echo "Install imagemagick: sudo apt install imagemagick"
```

## Software Requirements

- **Node.js**: LTS version (>= v20.x) with npm (>= v10.x)
- **Python**: 3.8 or higher
- **Raspberry Pi OS**: Bullseye or newer (64-bit recommended)

## Installation and Setup

### Prerequisites Check

Before proceeding, ensure you have completed the Raspberry Pi setup above. Verify your installation:

```bash
# Check Node.js version (should be >= v20.x)
node --version

# Check npm version (should be >= v10.x) 
npm --version

# Check Python version (should be >= 3.8)
python3 --version

# Test camera access
rpicam-hello --list-cameras
```

### 1. Clone and Setup Frontend

```bash
# Clone the repository (if not already done)
git clone https://github.com/federicocunico/insect_traps.git
cd insect_traps

# Setup frontend
cd frontend
npm install
npm run build
```

### 2. Setup Backend Environment

```bash
cd ../backend

# Create a Python virtual environment (recommended)
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Run the Application

Start both backend and frontend services:

```bash
# Terminal 1: Start backend server
cd backend
source venv/bin/activate  # If using virtual environment
python server.py

# Terminal 2: Start frontend development server (optional, for development)
cd frontend
npm run dev
```

The application will be accessible at:
- **Production build**: `http://localhost:8000` (served by Flask backend)
- **Development server**: `http://localhost:5173` (if running npm run dev)

### 4. Camera System Configuration

On first run, configure your camera setup:

1. Open the web interface
2. Go to camera settings
3. Set your target folder for image storage
4. Test each camera individually
5. Adjust camera parameters as needed

### Troubleshooting

**Camera not detected:**
```bash
# Check camera connections
rpicam-hello --list-cameras

# Check permissions
groups $USER  # Should include 'video' group

# Check device files
ls -la /dev/video*
```

**I2C issues:**
```bash
# Check I2C devices
sudo i2cdetect -y 1

# Verify I2C is enabled
lsmod | grep i2c
```

**Performance issues:**
```bash
# Check GPU memory
vcgencmd get_mem gpu  # Should show at least 128MB

# Monitor system resources
htop
```