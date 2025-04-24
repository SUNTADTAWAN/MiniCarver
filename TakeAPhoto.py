import cv2
import os

# === Folder and base name ===
folder_name = "calibration_images"
base_name = "calib_"

# === Create folder if not exists ===
os.makedirs(folder_name, exist_ok=True)

# === Open the default camera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access the camera.")
    exit()

print("✅ Camera is on. Press SPACE to capture, ESC to exit.")

img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        filename = os.path.join(folder_name, f"{base_name}_{img_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()
