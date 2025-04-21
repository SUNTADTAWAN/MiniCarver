# Camera Calibration
This branch for camera calibration.

### Clone Github
```
https://github.com/SUNTADTAWAN/MiniCarver.git
```

### Activate Environment
```
source aruco_env/bin/activate
```

### Run
```
python3 Camera_Calibration.py \
  --image_dir "/Users/SUN_Tadtawan/Documents/GitHub/Mini_Carver/calibration_images" \
  --image_format "jpg" \
  --prefix "calib_" \
  --square_size 0.024 \
  --width 9 \
  --height 6 \
  --save_file "camera_intrinsics.yml"
```
- **image_dir** - Path of your image
- **image_format** - Format of your image
- **prefix** - Your prefix file name
- **square_sizesquare_size** - Your square size 
- **width** - width of your square 
- **height** - height of your square 
- **save_file** - Path for save parameter as yml file