import numpy as np
import cv2
import glob
import argparse

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate(dirpath, prefix, image_format, square_size, width=9, height=6):
    """ Apply camera calibration operation for images in the given directory path. """
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane.

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    images = glob.glob(f"{dirpath}/{prefix}*.{image_format}")
    if len(images) == 0:
        raise RuntimeError(f"No images found with prefix '{prefix}' and format '{image_format}' in {dirpath}")

    print(f"[INFO] Found {len(images)} images. Starting calibration...")

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[Warning] Cannot read image {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

            # Uncomment to visualize detected corners
            # cv2.imshow('Corners', img)
            # cv2.waitKey(300)
        else:
            print(f"[Warning] Chessboard not found in image: {fname}")

    # cv2.destroyAllWindows()

    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise RuntimeError("No valid chessboard corners detected in any image. Check your input or chessboard size.")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return [ret, mtx, dist, rvecs, tvecs]

def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.release()

def load_coefficients(path):
    """ Load camera matrix and distortion coefficients. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()
    cv_file.release()
    return [camera_matrix, dist_matrix]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--image_dir', type=str, required=True, help='Image directory path')
    parser.add_argument('--image_format', type=str, required=True, help='Image format, png/jpg')
    parser.add_argument('--prefix', type=str, required=True, help='Image filename prefix')
    parser.add_argument('--square_size', type=float, required=True, help='Chessboard square size (in meters or cm)')
    parser.add_argument('--width', type=int, default=9, help='Chessboard inner width (corners)')
    parser.add_argument('--height', type=int, default=6, help='Chessboard inner height (corners)')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')

    args = parser.parse_args()
    ret, mtx, dist, rvecs, tvecs = calibrate(args.image_dir, args.prefix, args.image_format, args.square_size, args.width, args.height)
    save_coefficients(mtx, dist, args.save_file)
    print("[DONE] Calibration is finished.")
    print(f"RMS Reprojection Error: {ret}")
    print(f"Camera Matrix (K):\n{mtx}")
    print(f"Distortion Coefficients (D):\n{dist}")
