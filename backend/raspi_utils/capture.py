"""
Available cameras
-----------------
0 : imx477 [4056x3040 12-bit RGGB] (/base/soc/i2c0mux/i2c@1/pca@70/i2c@3/imx477@1a)
    Modes: 'SRGGB10_CSI2P' : 1332x990 [120.05 fps - (696, 528)/2664x1980 crop]
           'SRGGB12_CSI2P' : 2028x1080 [50.03 fps - (0, 440)/4056x2160 crop]
                             2028x1520 [40.01 fps - (0, 0)/4056x3040 crop]
                             4056x3040 [10.00 fps - (0, 0)/4056x3040 crop]

1 : imx477 [4056x3040 12-bit RGGB] (/base/soc/i2c0mux/i2c@1/pca@70/i2c@2/imx477@1a)
    Modes: 'SRGGB10_CSI2P' : 1332x990 [120.05 fps - (696, 528)/2664x1980 crop]
           'SRGGB12_CSI2P' : 2028x1080 [50.03 fps - (0, 440)/4056x2160 crop]
                             2028x1520 [40.01 fps - (0, 0)/4056x3040 crop]
                             4056x3040 [10.00 fps - (0, 0)/4056x3040 crop]

2 : imx477 [4056x3040 12-bit RGGB] (/base/soc/i2c0mux/i2c@1/pca@70/i2c@1/imx477@1a)
    Modes: 'SRGGB10_CSI2P' : 1332x990 [120.05 fps - (696, 528)/2664x1980 crop]
           'SRGGB12_CSI2P' : 2028x1080 [50.03 fps - (0, 440)/4056x2160 crop]
                             2028x1520 [40.01 fps - (0, 0)/4056x3040 crop]
                             4056x3040 [10.00 fps - (0, 0)/4056x3040 crop]

3 : imx477 [4056x3040 12-bit RGGB] (/base/soc/i2c0mux/i2c@1/pca@70/i2c@0/imx477@1a)
    Modes: 'SRGGB10_CSI2P' : 1332x990 [120.05 fps - (696, 528)/2664x1980 crop]
           'SRGGB12_CSI2P' : 2028x1080 [50.03 fps - (0, 440)/4056x2160 crop]
                             2028x1520 [40.01 fps - (0, 0)/4056x3040 crop]
                             4056x3040 [10.00 fps - (0, 0)/4056x3040 crop]
"""

import os
import subprocess


CAMERA_IDXS = [0, 1, 2, 3]
CAMERA_NAMES = ["Camera 0", "Camera 1", "Camera 2", "Camera 3"]

RESOLUTIONS = {
    "max": (4056, 3040),  # 4:3
    "medium_4:3": (2028, 1520),  # 4:3
    "medium_16:9": (2028, 1080),  # 16:9
    "small": (1332, 990),  # 4:3
}

def build_still_cmd(cam_idx: int, output_path: str, width: int, height: int, timeout: int = 500) -> list[str]:
    '''
    # DOCS: https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-still
    # Example: rpicam-still -o long_exposure.jpg --shutter 100000000 --gain 1 --awbgains 1,1 --immediate
    '''
    cmd = [
        "rpicam-still",
        "--camera", str(cam_idx),
        "-o", output_path,
        "-t", str(timeout),
        "--width", str(width),
        "--height", str(height),
        "-q", "100", "--immediate"
    ]
    return cmd


def capture_image(cam_idx: int, output_path: str, width: int, height: int, timeout: int = 500) -> None:
    cmd = build_still_cmd(cam_idx, output_path, width, height, timeout)
    cmd_str = ' '.join(cmd)
    print(f"Running command: {cmd_str}")
    
    # Try with shell=True first
    try:
        result = subprocess.run(cmd_str, shell=True, capture_output=True, check=True, text=True)
        print(f"Command succeeded for camera {cam_idx}")
        if result.stdout:
            print(f"stdout: {result.stdout.strip()}")
        return
    except subprocess.CalledProcessError as e:
        print(f"Shell command failed for camera {cam_idx}: return code {e.returncode}")
        if e.stderr:
            print(f"stderr: {e.stderr.strip()}")
    
    # If shell=True failed, try with list command
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        print(f"List command succeeded for camera {cam_idx}")
        if result.stdout:
            print(f"stdout: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error capturing image from camera {cam_idx}: {e}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout.strip()}")
        if e.stderr:
            print(f"stderr: {e.stderr.strip()}")
        raise
    except FileNotFoundError:
        print(f"Error: rpicam-still command not found. Make sure it's installed and in PATH.")
        raise


def __test__():
    for cam_idx in CAMERA_IDXS:
        output_path = f"camera_{cam_idx}_test.jpg"
        width, height = RESOLUTIONS["max"]
        print(f"Capturing image from Camera {cam_idx} to {output_path} at resolution {width}x{height}")
        capture_image(cam_idx, output_path, width, height)


if __name__ == "__main__":
    __test__()
