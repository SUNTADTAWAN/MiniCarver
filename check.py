import cv2

def list_available_cameras(max_index=50):
    available = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            print(f"Camera found at index {index}")
            available.append(index)
        cap.release()
    if not available:
        print("No cameras found.")
    return available

list_available_cameras()
