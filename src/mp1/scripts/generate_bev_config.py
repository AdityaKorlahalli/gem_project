# import os
# import json
# import numpy as np


# def main():
#     # intrinsics (K) from /front_single_camera/camera_info
#     K = np.array([
#         [476.7030836014194, 0.0, 400.0],
#         [0.0, 476.7030836014194, 300.0],
#         [0.0, 0.0, 1.0]
#     ])
#     # extrinsics (R,t): base_footprint -> front_single_camera_link
#     R = np.array([
#         [0, -1, 0],
#         [ 0, 0, 1],
#         [ -1, 0, 0]
#     ])
#     t = np.array([0.160, -0.110, 1.546])

#     # bird eye view image size (px); 
#     # NOTE: DO NOT CHANGE THIS
#     bev_img_height, bev_img_width = 600, 800

#     # HYPERPARAMETERS: BEV rectangle configuration (m)
#     # NOTE: Revert this to the original values for submission
#     bev_height, bev_width = 15, 20

#     # px -> m conversion factor
#     unit_conversion_factor = (bev_height/bev_img_height, bev_width/bev_img_width)
#     bev_world_coords = np.float32([
#         [bev_height, -bev_width/2, 0],
#         [0, -bev_width/2, 0],
#         [0, bev_width/2, 0],
#         [bev_height, bev_width/2, 0],
#     ])

#     # convert the bev_world_coords into pixel coordinates
#     src = []
#     for pt in bev_world_coords:
#         print("--- BEV Projection Debug ---")
#         ##### YOUR CODE STARTS HERE #####
#         pt_cam = R @ pt + (-R @ t)
#         pt_2d = K @ pt_cam

#         u = pt_2d[0] / pt_2d[2]
#         v = pt_2d[1] / pt_2d[2]

#         print(f"World {pt} -> Pixel (u: {u:.1f}, v: {v:.1f})")
#         src.append([u,v])
#         ##### YOUR CODE ENDS HERE #####
#         pass
#     src = np.float32(src)

#     output = {
#         "bev_world_dim": (bev_height, bev_width),
#         "unit_conversion_factor": unit_conversion_factor,
#         "src": src.tolist(),
#     }
#     # save config to json
#     save_fn = 'data/bev_config.json'
#     if not os.path.isdir('data/'):
#         print(f"Data directory not found. Generating...")
#         os.makedirs('data/', exist_ok=False)
#     if os.path.isfile(save_fn):
#         if input("File already exists. Overwrite? (y/n):").lower() != 'y':
#             print("Exiting...")
#             import sys
#             sys.exit()
#     with open(save_fn, "w") as f:
#         json.dump(output, f, indent=2)
#     print(f"Saved BEV config to {save_fn}.")


# if __name__ == "__main__":
#     main()

import os
import json
import numpy as np

def main():
    # 1. REAL ZED Intrinsics (K) from /zed/zed_node/rgb/camera_info
    # Extracted from the 'k' array: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    K = np.array([
        [263.7889709472656, 0.0, 308.12298583984375],
        [0.0, 263.7889709472656, 179.60777282714844],
        [0.0, 0.0, 1.0]
    ])

    # 2. Extrinsics (R,t): base_footprint -> zed_left_camera_optical_frame
    # NOTE: These values define the physical mounting of the camera on the GEM.
    # If the camera is mounted differently than in simulation, these will need adjustment.
    R = np.array([
        [0, -1, 0],
        [ 0, 0, 1],
        [ -1, 0, 0]
    ])
    t = np.array([0.160, -0.110, 1.546])

    # 3. Bird's Eye View image size (px)
    # NOTE: DO NOT CHANGE THIS
    bev_img_height, bev_img_width = 600, 800

    # 4. HYPERPARAMETERS: BEV physical dimensions (m)
    # We'll use 15m forward and 20m wide (10m left, 10m right)
    bev_height, bev_width = 15, 20

    # px -> m conversion factor
    unit_conversion_factor = (bev_height/bev_img_height, bev_width/bev_img_width)
    
    # Define the 4 corners of the BEV box in the physical world (m)
    bev_world_coords = np.float32([
        [bev_height, -bev_width/2, 0], # Top Left
        [0, -bev_width/2, 0],          # Bottom Left
        [0, bev_width/2, 0],           # Bottom Right
        [bev_height, bev_width/2, 0],  # Top Right
    ])

    # Convert the bev_world_coords into pixel coordinates using the camera model
    src = []
    print("--- BEV Projection Debug (Real ZED) ---")
    for pt in bev_world_coords:
        # Transform world point to camera frame: pt_cam = R(pt - t)
        pt_cam = R @ pt + (-R @ t)
        
        # Project 3D camera point to 2D image plane: pt_2d = K * pt_cam
        pt_2d = K @ pt_cam

        # Normalize by depth (z) to get pixel coordinates (u, v)
        u = pt_2d[0] / pt_2d[2]
        v = pt_2d[1] / pt_2d[2]

        print(f"World {pt} -> Pixel (u: {u:.1f}, v: {v:.1f})")
        src.append([u,v])
        
    src = np.float32(src)

    output = {
        "bev_world_dim": (bev_height, bev_width),
        "unit_conversion_factor": unit_conversion_factor,
        "src": src.tolist(),
    }

    # Save config to json
    save_fn = 'data/bev_config_real.json'
    if not os.path.isdir('data/'):
        print(f"Data directory not found. Generating...")
        os.makedirs('data/', exist_ok=False)

    if os.path.isfile(save_fn):
        if input(f"File {save_fn} already exists. Overwrite? (y/n):").lower() != 'y':
            print("Exiting...")
            import sys
            sys.exit()

    with open(save_fn, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSuccessfully saved REAL BEV config to {save_fn}.")

if __name__ == "__main__":
    main()