# External libraries
import math
import numpy as np

# UDP parameters
localIP = "192.168.0.200" # Put your laptop computer's 
arduinoIP = "192.168.0.199" # Put your arduino's IP h
localPort = 4010
arduinoPort = 4010
bufferSize = 1024


# Camera parameters
# Continuity Camera default path: OpenCV webcam index.
# If you prefer an iPhone streaming app URL, set camera_stream_mode="url".
camera_stream_mode = "continuity"  # continuity | url
camera_id = 1
# When iPhone Continuity drops, index 1 may be invalid; try built-in next.
camera_fallback_indices = [0, 2]
camera_stream_url = "http://192.168.0.199:4747/video"
camera_warmup_frames = 15
# Continuity Camera sometimes returns transient read failures.
camera_read_retries = 5

# Object color detection in HSV.
# Prefer hsv_ranges when you need multiple OR bands (e.g. label + highlights).
# If hsv_ranges is empty, detection uses only hsv_lower / hsv_upper (recommended
# while tuning so `python -m laptop_stack.tune_ball_hsv` matches the detector).
#
# Preset for a saturated mostly-blue box (Hue 0–179 in OpenCV; blue ~100–125).
hsv_ranges = []

hsv_lower = (98, 129, 164)
hsv_upper = (110, 255, 255)
# If h_min > h_max in legacy mode, vision normally treats that as red hue wrap.
# Keep False for blue boxes so an accidental slider crossing does not explode the mask.
hsv_legacy_hue_wrap = False

# Shape filters: balls are round; bottles and boxes usually keep this False.
use_ball_shape_filters = False

# Basic contour filtering for ball-like targets.
min_contour_area = 2500
# One solid blue blob: merging is unnecessary; enable if lighting splits the mask.
merge_nearby_label_contours = False
merge_max_gap_px = 52
merge_fragment_min_contour_area = 400

min_circularity = 0.60
# Bounding box aspect ratio w/h; keep near 1 for a ball.
max_bbox_aspect_ratio = 1.35
# Solidity = contour_area / convex_hull_area (1 for convex blob).
min_solidity = 0.85

# Morphology on the HSV mask (noise reduction).
morph_kernel_px = 5
morph_open_iterations = 1
morph_close_iterations = 4

# Reject huge low-saturation blobs (plain walls, ceiling highlights).
reject_large_low_sat_blobs = True
large_blob_min_bbox_area_frac = 0.08
large_blob_max_mean_saturation = 55.0

# Reject low-saturation blobs (gray-blue walls; raise if you lose a dusty matte box).
min_blob_mean_saturation = 62.0

# Reject wall-sized sheets when the camera is close (huge mask blobs).
# If the real box disappears when it is large in-frame, raise these a little.
max_bbox_area_frac = 0.20
max_bbox_width_frac = 0.55
max_bbox_height_frac = 0.65
max_contour_area_frac = 0.14

# Reject contours that touch image borders (common for wall backgrounds).
reject_border_touching_contours = True
border_margin_px = 4

# Debounce to avoid one-frame false positives.
required_detection_frames = 6

# Vision-guided drive: center blob horizontally while moving forward until
# "close enough" (bbox fills enough of frame) or target lost for patience frames.
# vision_steering_gain: added to SERVO_CENTER; negative typical if blob right ⇒ turn left servo.
vision_steering_gain = 48.0
vision_approach_speed = 45
vision_approach_slow_speed = 16
vision_approach_error_slow_frac = 0.35  # |norm horizontal error| beyond this trims speed
vision_stop_bbox_area_frac = 0.09
vision_lost_patience_frames = 18
# Waypoint / PF mode needs encoder CSV lines from the Arduino. Vision chase runs
# even when serial has no line (see full_pipeline loop).
require_serial_sensor_for_explore = True

# Optional debug windows for tuning.
vision_debug_windows = True
marker_length = 0.071
camera_matrix = np.array([[1.41089024e+03, 0.00000000e+00 ,5.34757040e+02],
 [0.00000000e+00 ,1.40977771e+03, 4.63300611e+02],
 [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]], dtype=np.float32)
dist_coeffs = np.array([-0.32511173, -0.09273864 ,-0.00295959 , 0.00111094 , 0.2446519 ], dtype=np.float32)

# Robot parameters / dead reckoning stub (robot.py at repo root)
encoder_meters_per_tick = 0.00008  # scale encoder delta → meters; tune to your robot

num_robot_sensors = 2 # encoder, steering
num_robot_control_signals = 2 # speed, steering

# Logging parameters
max_num_lines_before_write = 1
filename_start = './data/robot_data'
data_name_list = ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal', 'state_mean', 'state_covariance']

# Experiment trial parameters
trial_time = 10000 # milliseconds
extra_trial_log_time = 2000 # milliseconds

# KF parameters
I3 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
covariance_plot_scale = 100

# PF parameters, modify the map and num particles as you see fit.
num_particles = 100
distance_variance = 0.05  # m^2, lidar likelihood variance for PF weighting

## need to change accoriing to out mapping 
##[x1,y1,x2,y2]
wall_corner_list = [
    [0, 0, 9.6, 0],  ##wall 1 lngest ; T top 
    [9.6, 0, 9.6, -1.63], ##wall 2
    [9.6,-1.63 , 7.1, -1.63], ##wall 3
    [7.1, -1.61, 2.5, -8.4],  ##wall4
    [2.5, -8.4, 7.1, -8.4], ##wall 5 : base
    [5.2, -8.4, 5.2, -1.63],  ##wall 6
    [5.2, -1.63, 0, -1.63], ##wall 7
    [0, -1.63, 0, 0] #3wall 8 , closing side 
    ]
















## need to change accoriing to out mapping 
##[x1,y1,x2,y2]
wall_corner_list = [
    [0, 0, 9.6, 0],  ##wall 1 lngest ; T top 
    [9.6, 0, 9.6, -1.63], ##wall 2
    [9.6,-1.63 , 7.1, -1.63], ##wall 3

    [7.1, -1.63, 7.1, -8.4],  ##wall4       
    [7.1, -8.4, 5.2, -8.4], ##wall 5 : base
    [5.2, -8.4, 5.2, -1.63],  ##wall 6
    [5.2, -1.63, 0, -1.63], ##wall 7
    [0, -1.63, 0, 0] #3wall 8 , closing side 
    ]


