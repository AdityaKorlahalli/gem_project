import torch
import cv2
import numpy as np

from dataset import CaptureDataset

def mask_by_hsv(image, target_hsv, tolerance):
    if isinstance(tolerance, int):
        tol_h, tol_s, tol_v = tolerance, tolerance, tolerance
    else:
        tol_h, tol_s, tol_v = tolerance
    h, s, v = target_hsv
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    s_min = max(s - tol_s, 0)
    s_max = min(s + tol_s, 255)
    v_min = max(v - tol_v, 0)
    v_max = min(v + tol_v, 255)

    if h - tol_h < 0:
        lower1 = np.array([0, s_min, v_min])
        upper1 = np.array([h + tol_h, s_max, v_max])

        lower2 = np.array([179 + (h - tol_h), s_min, v_min])
        upper2 = np.array([179, s_max, v_max])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif h + tol_h > 179:
        lower1 = np.array([h - tol_h, s_min, v_min])
        upper1 = np.array([179, s_max, v_max])

        lower2 = np.array([0, s_min, v_min])
        upper2 = np.array[(h + tol_h) - 179, s_max, v_max]

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower = np.array([h - tol_h, s_min, v_min])
        upper = np.array([h + tol_h, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
    return mask


if __name__ == "__main__":
    ds = CaptureDataset("data/capture")

    yellow_lane = [30, 255, 255]

    for i in range(len(ds)):
        image, _ = ds.read(i)
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. Yellow Lanes
        yellow_mask = mask_by_hsv(image, yellow_lane, [10, 100, 150])    
        
        # 2. Strict White Lanes
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        thresh = cv2.bitwise_or(yellow_mask, white_mask)

        # 3. AGGRESSIVE HORIZON CROP
        thresh[:int(height * 0.55), :] = 0
        
        # 4. SUPERCHARGED ANTI-CONE MASK
        # Cones span the red/orange hue wrap-around and can have shadows (lower sat/val)
        lower_orange1 = np.array([0, 80, 80])
        upper_orange1 = np.array([25, 255, 255])
        lower_orange2 = np.array([170, 80, 80])
        upper_orange2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)
        orange_mask = cv2.bitwise_or(mask1, mask2)
        
        # Cones are tall. Use a massive vertically biased kernel to stretch the mask 
        # up and down over the white stripes and the base.
        kernel = np.ones((80, 40), np.uint8) 
        cone_mask = cv2.dilate(orange_mask, kernel, iterations=1)
        
        # Erase!
        thresh[cone_mask > 0] = 0

        ds.write_mask(thresh, i)
        print(f"Processed image {i+1}/{len(ds)}")