import os
import math 
import torch
import json
import numpy as np
import time
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.parameter import Parameter

from worldgt import WorldGT
from line_fit import lane_fit, final_viz, perspective_transform, closest_point_on_polynomial
from model_utils import load_model, inference
import rich
import cv2
from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDrive

# Add this line:
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

class LaneVisualizer(Node):
    def __init__(self):
        super().__init__("lane_visualizer")

        sim_time_param = Parameter('use_sim_time', Parameter.Type.BOOL, True)
        self.set_parameters([sim_time_param])

        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._model = load_model()
            if self._model is not None:
                self._model = self._model.to(self._dev)
                self._model = self._model.eval()
                rich.print("[green]loaded SimpleEnet :o")
            else:
                self.get_logger().error(f"could not load SimpleEnet model x_X: {e}")
                exit(1)
        except Exception as e:
            self.get_logger().error(f"could not load SimpleEnet model x_X: {e}")
            exit(1)
        
        try: 
            with open(os.path.join("data", "bev_config.json")) as f:
                self._bev_cfg = json.load(f)
        except FileNotFoundError:
            self.get_logger().error(f"could not load bev config x_X: {e}")
            exit(1)

        self._world = WorldGT("HighBay")
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)
        self._path_pub = self.create_publisher(Path, "/lane_path", 10)
        self._image_msg = None
        self._cv_bridge = CvBridge()
        
        self.create_subscription(
            Image,
            "/camera/image_raw",
            self._on_image,
            10
        )
        # Copying the QoS profile from drive.py
        streaming_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE 
        )
        self._control_pub = self.create_publisher(AckermannDrive, "/ackermann_cmd", streaming_qos)
        #self._control_pub = self.create_publisher(AckermannDrive, "/ackermann_cmd", 1)
        self.L = 1.75
        self._prev_steer = 0.0

    def stanley_lateral_controller(self, current_he, current_xte, current_speed):
            """
            Calculates steering based on Heading Error and Cross-Track Error.
            """
            k = 0.5 # Softened the gain slightly
            v = max(current_speed, 0.1) 
            
            # FLIPPED THE SIGN ON XTE:
            # If XTE is positive (lane is to the right), we need a negative steering angle (steer right)
            steering = current_he + math.atan2(-k * current_xte, v)
            
            steering = np.clip(steering, -0.8, 0.8)
            return steering

    def longitudinal_controller(self, current_he):
        """
        Slow, safe speeds for testing in the Highbay and Sim.
        """
        straight_speed = 3.0 # Significantly reduced from 12.0
        corner_speed = 1.5   # Significantly reduced from 8.0

        if abs(current_he) > 0.1: # Increased threshold to ~5.7 degrees
            return corner_speed
        else:
            return straight_speed

    def _on_image(self, msg) -> None:
        self._image_msg = msg
        if self._model is None:
            return
        
        XTE, HE = "N/A", "N/A"
        lane, gt_XTE, gt_HE = "unknown", "N/A", "N/A"
        
        # 1. DO NOT CROP FOR SIMULATION
        image = self._cv_bridge.imgmsg_to_cv2(self._image_msg, "bgr8")
        
        # 2. Perception Pipeline
        mask = inference(self._model, image, self._dev)
        m = mask.astype(np.uint8) * 255

        # 3. Use 50% cut for Simulator (62% is for the Highbay buildings)
        h_mask = m.shape[0]
        m[:int(h_mask * 0.50), :] = 0 
        
        # 4. Fitting
        combine_fit_img, binary_BEV, ret = self.fit_poly_lanes(image, m)

        # Prepare BEV for drawing
        binary_BEV = np.pad(binary_BEV, ((0, 100), (0, 0)))
        binary_BEV = cv2.cvtColor(binary_BEV, cv2.COLOR_GRAY2BGR)
        
        if ret:
            left_fit = ret["left_fit"]
            right_fit = ret["right_fit"]

            # --- 1. EVALUATE POLYNOMIALS ---
            lookahead_y_px = 350
            
            left_lookahead_x = np.polyval(left_fit, lookahead_y_px)
            right_lookahead_x = np.polyval(right_fit, lookahead_y_px)
            
            # Evaluate at the bumper to see which line is closest to the car
            left_bumper_x = np.polyval(left_fit, 600)
            right_bumper_x = np.polyval(right_fit, 600)

            # --- 2. DYNAMIC LANE WEIGHTING ---
            NOMINAL_WIDTH = 170 
            MAX_WIDTH = 250 # If the lane is wider than this, a line is hallucinating
            MIN_WIDTH = 80  # If the lane is narrower than this, lines are crossing
            
            # Calculate the current observed width
            current_width = right_lookahead_x - left_lookahead_x

            # Calculate distance from center of bumper (x = 400)
            dist_left = abs(left_bumper_x - 400.0)
            dist_right = abs(right_bumper_x - 400.0)

            # HARD CUTOFF: If the lane is impossibly wide or narrow
            if current_width > MAX_WIDTH or current_width < MIN_WIDTH:
                # Force 100% trust on the line closest to the car's center
                if dist_left < dist_right:
                    weight_left = 1.0
                    weight_right = 0.0
                else:
                    weight_left = 0.0
                    weight_right = 1.0
            else:
                # NORMAL BLENDING: Use inverse distance weighting
                weight_left = 1.0 / ((dist_left + 10.0) ** 2)
                weight_right = 1.0 / ((dist_right + 10.0) ** 2)
                
                # Normalize so weights sum to 1.0
                total_weight = weight_left + weight_right
                weight_left /= total_weight
                weight_right /= total_weight

            # What is the target if we ONLY trust the left line?
            target_from_left = left_lookahead_x + (NOMINAL_WIDTH / 2.0)
            
            # What is the target if we ONLY trust the right line?
            target_from_right = right_lookahead_x - (NOMINAL_WIDTH / 2.0)

            # Blend the targets! 
            target_x_px = (weight_left * target_from_left) + (weight_right * target_from_right)

            # --- 3. CONVERT PIXELS TO METERS (Car Frame) ---
            Sy, Sx = self._bev_cfg["unit_conversion_factor"]
            
            x_forward_m = (600 - lookahead_y_px) * Sy
            y_lateral_m = -(target_x_px - 400.0) * Sx

            # --- 4. PURE PURSUIT CONTROLLER ---
            ld = math.hypot(x_forward_m, y_lateral_m)
            if ld > 0.001:
                alpha = math.atan2(y_lateral_m, x_forward_m)
                base_steering = math.atan2(2 * self.L * math.sin(alpha), ld)
                
                steering_gain = 1.6 
                raw_steering = base_steering * steering_gain
            else:
                raw_steering = 0.0
            
            raw_steering = np.clip(raw_steering, -0.8, 0.8)

            # --- UPGRADE A: TEMPORAL SMOOTHING (LOW-PASS FILTER) ---
            # Neural networks jitter frame-to-frame. We blend the new steering command 
            # with the previous one to act as "suspension" for the steering wheel.
            # 0.3 means we trust 30% of the new frame and keep 70% of the old momentum.
            alpha_filter = 0.6
            target_steering = (alpha_filter * raw_steering) + ((1.0 - alpha_filter) * self._prev_steer)
            self._prev_steer = target_steering # Save for the next frame

            # --- UPGRADE B: DYNAMIC SPEED CONTROL ---
            # Humans drive fast on straights and brake for corners. 
            # If the steering angle is near 0.0, it goes 3.0 m/s.
            # If the steering angle is sharp (e.g., 0.8 rad), it slows down safely to ~1.5 m/s.
            target_speed = max(1.5, 3.0 - (abs(target_steering) * 2.5))

            # --- 5. PUBLISH COMMAND ---
            cmd = AckermannDrive()
            cmd.speed = float(target_speed)
            cmd.steering_angle = float(target_steering)
            self._control_pub.publish(cmd)

            # --- 6. DRAWING LOGIC ---
            poly_px = (np.add(left_fit, right_fit) / 2)
            est_xte_val, est_he_val, camera_px, closest_px = self.compute_error(poly_px)
            
            ploty = ret['ploty']
            left_fitx = np.polyval(left_fit, ploty) 
            center_fitx = np.polyval(poly_px, ploty)
            right_fitx = np.polyval(right_fit, ploty) 
            
            pts_left = np.stack((left_fitx, ploty), axis=1).astype(np.int32)
            pts_center = np.stack((center_fitx, ploty), axis=1).astype(np.int32)
            pts_right = np.stack((right_fitx, ploty), axis=1).astype(np.int32)

            cv2.polylines(binary_BEV, [pts_center], isClosed=False, color=(0, 255, 255), thickness=4)                
            cv2.polylines(binary_BEV, [pts_left], isClosed=False, color=(255, 0, 0), thickness=4)
            cv2.polylines(binary_BEV, [pts_right], isClosed=False, color=(0, 0, 255), thickness=4)

            # Draw a bright purple circle where the car is ACTUALLY aiming
            cv2.circle(binary_BEV, (int(target_x_px), int(lookahead_y_px)), 12, (255, 0, 255), -1)

            XTE = f"{est_xte_val:.2f}"
            HE = f"{np.degrees(est_he_val):.2f}"

        try:
            trans = self._tf_buf.lookup_transform("world", "base_link", msg.header.stamp)
            pos = trans.transform.translation
            rotation = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
            yaw = rotation.as_euler('xyz', degrees=False)[2]
            lane_id, _, gt_xte_val, gt_he_val = self._world.get_metrics(pos.x, pos.y, yaw)
            gt_XTE = f"{gt_xte_val:.2f}"
            gt_HE = f"{np.degrees(gt_he_val):.2f}"
            lane = lane_id
        except:
            pass
            
        print(f"EST XTE: {XTE} m - HE: {HE}° -- GT XTE: {gt_XTE} m HE: {gt_HE}° - lane: {lane}")

        if combine_fit_img is None:
            combine_fit_img = image

        cv2.imshow("render_view", combine_fit_img)
        cv2.imshow("binary_BEV", binary_BEV)
        cv2.waitKey(1)
        
    # def _on_image(self, msg) -> None:
    #     self._image_msg = msg
    #     if self._model is None:
    #         return
        
    #     image = self._cv_bridge.imgmsg_to_cv2(self._image_msg, "bgr8")
    #     mask = inference(self._model, image, self._dev)
    #     m = mask.astype(np.uint8) * 255

    #     # Adding in to clean up binary mask

    #     # kernel = np.ones((5, 5), np.uint8)
    #     # m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    #     # m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    #     # h = m.shape[0]
    #     # m[:int(h*0.5), :] = 0   # zero out top 50%
    #     ##
    #     combine_fit_img, binary_BEV, ret = self.fit_poly_lanes(image, m)

    #     binary_BEV = np.pad(binary_BEV, ((0, 100), (0, 0)))
    #     binary_BEV = cv2.cvtColor(binary_BEV, cv2.COLOR_GRAY2BGR)
        
    #     if ret:                
    #         poly_px = (np.add(ret["left_fit"], ret["right_fit"]) / 2)
    #         XTE, HE, camera_px, closest_px = self.compute_error(poly_px)
            
    #         # draw lane lines
    #         ploty = ret['ploty']
    #         left_fitx = np.polyval(ret["left_fit"], ploty)
    #         center_fitx = np.polyval(poly_px, ploty)
    #         right_fitx = np.polyval(ret["right_fit"], ploty)
            
    #         pts_left = np.stack((left_fitx, ploty), axis=1).astype(np.int32)
    #         pts_center = np.stack((center_fitx, ploty), axis=1).astype(np.int32)
    #         pts_right = np.stack((right_fitx, ploty), axis=1).astype(np.int32)

    #         cv2.polylines(binary_BEV, [pts_center], isClosed=False, color=(0, 255, 255), thickness=4)                
    #         cv2.polylines(binary_BEV, [pts_left], isClosed=False, color=(255, 0, 0), thickness=4)
    #         cv2.polylines(binary_BEV, [pts_right], isClosed=False, color=(0, 0, 255), thickness=4)

    #         # draw closest point and bridge line
    #         cv2.circle(binary_BEV, (int(closest_px[0]), int(closest_px[1])), 8, (0, 255, 0), -1)
    #         cv2.line(
    #             binary_BEV,
    #             (int(camera_px[0]), int(camera_px[1])),
    #             (int(closest_px[0]), int(closest_px[1])),
    #             (0, 255, 0),
    #             4
    #         )

    #         # draw camera chevron
    #         cv2.line(
    #             binary_BEV,
    #             (int(camera_px[0]), int(camera_px[1])),
    #             (int(camera_px[0] - 20), int(camera_px[1] + 20)),
    #             (255, 0, 255),
    #             4
    #         )
    #         cv2.line(
    #             binary_BEV,
    #             (int(camera_px[0]), int(camera_px[1])),
    #             (int(camera_px[0] + 20), int(camera_px[1] + 20)),
    #             (255, 0, 255),
    #             4
    #         )

    #         XTE = f"{XTE:.2f}"
    #         HE = f"{np.degrees(HE):.2f}"
    #     else:
    #         XTE = "N/A"
    #         HE = "N/A"

    #     try:
    #         trans = self._tf_buf.lookup_transform("highbay", "stereo_camera_link", msg.header.stamp)
    #         pos = trans.transform.translation
    #         q = trans.transform.rotation
    #         rotation = R.from_quat([q.x, q.y, q.z, q.w])
    #         euler_angles = rotation.as_euler('xyz', degrees=False)
    #         yaw = euler_angles[2]
    #         lane, _, gt_XTE, gt_HE = self._world.get_metrics(pos.x, pos.y, yaw)
    #         gt_XTE = f"{gt_XTE:.2f}"
    #         gt_HE = f"{np.degrees(gt_HE):.2f}"
    #     except:
    #         lane = "unknown"
    #         gt_XTE = "N/A"
    #         gt_HE = "N/A"
            
    #     print(f"EST XTE: {XTE} m - HE: {HE}° -- GT XTE: {gt_XTE} m HE: {gt_HE}° - lane: {lane}")

    #     if combine_fit_img is None:
    #         combine_fit_img = image
            

    #     cv2.imshow("render_view", combine_fit_img)
    #     cv2.imshow("binary_BEV", binary_BEV)
    #     cv2.waitKey(1)

    
    def compute_error(self, poly_px):
        """
        Calculates Cross-Track Error (XTE) and Heading Error.

        poly_px:    polynomial coefficients defined in pixels
                    ex for 2nd order: (A, B, C) where x = Ay^2 + By + C
        """
        bev_height_m, bev_width_m = self._bev_cfg["bev_world_dim"]
        Sy, Sx = self._bev_cfg["unit_conversion_factor"]
        scale = np.array([Sx, Sy])

        camera_m = np.array([(bev_width_m / 2), bev_height_m])
        camera_px = camera_m / scale
        closest_px = closest_point_on_polynomial(camera_px, poly_px)
        closest_m = closest_px * scale

        ##### YOUR CODE STARTS HERE #####

        # calculate cross track error
        # hint: |XTE| = distance between camera and closest point
        #       on ploly_px however XTE is not a strictly positive value
        # XTE = np.sqrt(((camera_m[0] - closest_m[0]) ** 2) + ((camera_m[1] - closest_m[1]) ** 2)) # Euclidean Distance between car and lane
        # if (camera_m[0] > closest_m[0]):
        #     XTE *=-1

        XTE = -camera_m[0] + closest_m[0]

        # hint: find derivative of the polynomial at the closest point
        #       then use arctan on the scaled slope
        # HE = 0

        # der_poly = []
        # for i in range (len(poly_px) - 1): # Computes derivative and saves in a list
        #     der_poly.append(poly_px[i] * (len(poly_px) - i - 1))

        # # Plug in closest point
        # sum_val = 0

        # for i in range(len(der_poly)):
        #     sum_val += der_poly[len(der_poly) - i - 1] * (closest_px[1] ** i)

        # sum_val *= Sx/Sy
        # HE = np.arctan(sum_val)

        A, B, C = poly_px

        y_closest_px = closest_px[1]
        slope_px = 2*A*y_closest_px + B
        HE = np.arctan2(slope_px*Sx, Sy)


        ##### YOUR CODE ENDS HERE #####

        return XTE, HE, camera_px, closest_px

    def fit_poly_lanes(self, raw_img, binary_img):
        binary_warped, M, Minv = perspective_transform(binary_img, np.float32(self._bev_cfg["src"]))
        ret = lane_fit(binary_warped)
        if ret is None:
            self.get_logger().debug("ret is None; returning None for both.")
            return None, binary_warped, None
        left_fit = ret['left_fit']
        right_fit = ret['right_fit']
        
        combine_fit_img = None
        if ret is not None:
            self.get_logger().debug("Model detected lanes")
            combine_fit_img = final_viz(raw_img, left_fit, right_fit, Minv)
        else:
            self.get_logger().debug("Model unable to detect lanes")
        return combine_fit_img, binary_warped, ret

def main(args=None):
    rclpy.init(args=args)
    node = LaneVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down! Stopping the car...")
        stop_cmd = AckermannDrive()
        stop_cmd.speed = 0.0
        stop_cmd.steering_angle = 0.0
        
        # Force the message out by spinning the executor briefly
        for _ in range(10):
            node._control_pub.publish(stop_cmd)
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()


