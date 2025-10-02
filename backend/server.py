from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import os
import tempfile
import time
import threading
from datetime import datetime
from raspi_utils.capture import capture_image, CAMERA_IDXS, RESOLUTIONS

app = Flask(__name__, static_folder="../frontend/dist", static_url_path="/")
CORS(app)  # Enable CORS for all routes

# Global variables for tracking images
image_counters = {}
current_folder = None

# Lock to prevent concurrent camera operations
camera_lock = threading.Lock()


@app.route('/api/check-cameras', methods=['POST'])
def check_cameras():
    """Check if all cameras are working by taking test images in /tmp/ sequentially"""
    # Use lock to prevent concurrent camera operations
    with camera_lock:
        results = {}
        temp_dir = tempfile.gettempdir()
        
        print("Starting camera check sequence...")
        # Test each camera sequentially to avoid rpicam-still conflicts
        for i, cam_idx in enumerate(CAMERA_IDXS):
            # Add a small delay between cameras to ensure pipeline is released
            if i > 0:
                print(f"Waiting 1s before testing next camera...")
                time.sleep(1.0)
                
            try:
                test_path = os.path.join(temp_dir, f"test_cam_{cam_idx}.jpg")
                width, height = RESOLUTIONS["max"]
                
                print(f"Testing camera {cam_idx}...")
                capture_image(cam_idx, test_path, width, height, timeout=1000)
                print(f"Camera {cam_idx} test completed")
                
                # Check if file was created and has reasonable size
                if os.path.exists(test_path):
                    file_size = os.path.getsize(test_path)
                    if file_size > 1000:  # At least 1KB
                        results[f"camera_{cam_idx}"] = {"status": "ok", "size": file_size}
                        # Clean up test file
                        os.remove(test_path)
                        print(f"Camera {cam_idx} test passed ({file_size} bytes)")
                    else:
                        results[f"camera_{cam_idx}"] = {"status": "error", "message": "File too small"}
                        print(f"Camera {cam_idx} test failed: File too small ({file_size} bytes)")
                else:
                    results[f"camera_{cam_idx}"] = {"status": "error", "message": "File not created"}
                    print(f"Camera {cam_idx} test failed: File not created")
                    
            except Exception as e:
                results[f"camera_{cam_idx}"] = {"status": "error", "message": str(e)}
                print(f"Camera {cam_idx} test failed: {str(e)}")
    
        print(f"Camera testing completed for all {len(CAMERA_IDXS)} cameras")
        return jsonify(results)


@app.route('/api/set-folder', methods=['POST'])
def set_folder():
    """Set the destination folder for images"""
    global current_folder, image_counters
    
    data = request.get_json()
    folder_path = data.get('folder_path')
    
    if not folder_path:
        return jsonify({"status": "error", "message": "No folder path provided"}), 400
    
    try:
        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        current_folder = folder_path
        
        # Scan existing images to determine starting counters
        image_counters = {cam_idx: 0 for cam_idx in CAMERA_IDXS}
        
        if os.path.exists(folder_path):
            try:
                files = os.listdir(folder_path)
                image_files = [f for f in files if f.startswith('IMG_CAM') and f.endswith('.jpg')]
                
                # Extract counter values from existing files
                for filename in image_files:
                    # Format: IMG_CAM{cam_idx}_{counter:06d}.jpg
                    try:
                        parts = filename.replace('IMG_CAM', '').replace('.jpg', '').split('_')
                        if len(parts) == 2:
                            cam_idx = int(parts[0])
                            counter = int(parts[1])
                            if cam_idx in image_counters:
                                image_counters[cam_idx] = max(image_counters[cam_idx], counter)
                    except (ValueError, IndexError):
                        continue  # Skip malformed filenames
                        
                print(f"Scanned folder {folder_path}, found counters: {image_counters}")
            except Exception as e:
                print(f"Error scanning folder: {e}")
                # Keep default counters if scanning fails
        
        return jsonify({
            "status": "ok", 
            "folder": current_folder,
            "image_counters": image_counters
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/capture-images', methods=['POST'])
def capture_images():
    """Capture images from all cameras sequentially"""
    global current_folder, image_counters
    
    # Use lock to prevent concurrent camera operations
    with camera_lock:
        if not current_folder:
            return jsonify({"status": "error", "message": "No folder set"}), 400
        
        data = request.get_json()
        resolution = data.get('resolution', 'max')
        
        if resolution not in RESOLUTIONS:
            resolution = 'max'
        
        width, height = RESOLUTIONS[resolution]
        results = {}
        
        print("Starting capture sequence...")
        # Capture from each camera sequentially to avoid rpicam-still conflicts
        for i, cam_idx in enumerate(CAMERA_IDXS):
            # Add a small delay between cameras to ensure pipeline is released
            if i > 0:
                print(f"Waiting 1s before next camera...")
                time.sleep(1.0)
                
            image_counters[cam_idx] += 1
            
            try:
                filename = f"IMG_CAM{cam_idx}_{image_counters[cam_idx]:06d}.jpg"
                output_path = os.path.join(current_folder, filename)
                
                # Capture image and wait for completion before moving to next camera
                print(f"Capturing from camera {cam_idx}...")
                capture_image(cam_idx, output_path, width, height, timeout=1000)
                print(f"Camera {cam_idx} capture completed")
                
                # Verify file was created
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    results[f"camera_{cam_idx}"] = {
                        "status": "ok", 
                        "filename": filename,
                        "size": file_size,
                        "counter": image_counters[cam_idx]
                    }
                    print(f"Camera {cam_idx} image saved: {filename} ({file_size} bytes)")
                else:
                    results[f"camera_{cam_idx}"] = {"status": "error", "message": "File not created"}
                    print(f"Camera {cam_idx} error: File not created")
                    
            except Exception as e:
                results[f"camera_{cam_idx}"] = {"status": "error", "message": str(e)}
                print(f"Camera {cam_idx} error: {str(e)}")
    
        print(f"Sequential capture completed for all {len(CAMERA_IDXS)} cameras")
        return jsonify(results)


@app.route('/api/get-status', methods=['GET'])
def get_status():
    """Get current status including folder and image counters"""
    return jsonify({
        "current_folder": current_folder,
        "image_counters": image_counters,
        "available_resolutions": list(RESOLUTIONS.keys())
    })


@app.route('/api/get-latest-images', methods=['GET'])
def get_latest_images():
    """Get the latest captured image for each camera"""
    if not current_folder or not os.path.exists(current_folder):
        return jsonify({"error": "No folder set or folder doesn't exist"}), 400
    
    latest_images = {}
    
    for cam_idx in CAMERA_IDXS:
        if cam_idx in image_counters and image_counters[cam_idx] > 0:
            filename = f"IMG_CAM{cam_idx}_{image_counters[cam_idx]:06d}.jpg"
            filepath = os.path.join(current_folder, filename)
            if os.path.exists(filepath):
                latest_images[f"camera_{cam_idx}"] = {
                    "filename": filename,
                    "url": f"/api/image/{filename}",
                    "counter": image_counters[cam_idx]
                }
    
    return jsonify(latest_images)


@app.route('/api/get-image-list', methods=['GET'])
def get_image_list():
    """Get list of all captured images"""
    if not current_folder or not os.path.exists(current_folder):
        return jsonify({"error": "No folder set or folder doesn't exist"}), 400
    
    images = []
    try:
        files = os.listdir(current_folder)
        image_files = [f for f in files if f.startswith('IMG_CAM') and f.endswith('.jpg')]
        image_files.sort(reverse=True)  # Most recent first
        
        for filename in image_files:
            filepath = os.path.join(current_folder, filename)
            if os.path.exists(filepath):
                stat = os.stat(filepath)
                images.append({
                    "filename": filename,
                    "url": f"/api/image/{filename}",
                    "size": stat.st_size,
                    "timestamp": stat.st_mtime
                })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    return jsonify(images)


@app.route('/api/image/<filename>')
def serve_image(filename):
    """Serve captured images"""
    if not current_folder:
        return "No folder set", 404
    
    # Security check: only allow IMG_CAM*.jpg files
    if not filename.startswith('IMG_CAM') or not filename.endswith('.jpg'):
        return "Invalid filename", 403
    
    return send_from_directory(current_folder, filename)


# Serve static files (JS, CSS, images, etc.)
@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory(os.path.join(app.static_folder, 'assets'), filename)

# Serve the main page and SPA fallback
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    # If the file exists, serve it
    file_path = os.path.join(app.static_folder, path)
    if path and os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    # Otherwise, serve index.html (SPA fallback)
    return send_from_directory(app.static_folder, 'index.html')



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
