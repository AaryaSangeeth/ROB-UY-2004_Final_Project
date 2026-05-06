
"""
Corridor exploration: particle-filter waypoint navigation 
plus camera-based object detection and vision-guided approach.


"""
from __future__ import annotations

import glob
import math
import os
import sys
import time
import tty
import termios

import cv2
import numpy as np
import serial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import parameters
import robot_python_code
from laptop_stack.camera_util import open_cv_camera
from laptop_stack.vision_detection import detect_colored_ball
from particle_filter import Map, ParticleFilterPlot, State, angle_wrap
from robot import Robot

#waypoints for exploration
EXPLORE_WAYPOINTS = [
    (1.0, -0.8),
    (3.0, -0.8),
    (5.0, -0.8),
    (7.0, -0.8),
    (9.0, -0.8),
    (6.15, -0.8),
    (6.15, -2.5),
    (6.15, -4.5),
    (6.15, -6.5),
    (6.15, -7.8),
    (6.15, -4.5),
    (6.15, -2.0),
    (6.15, -0.8),
]

BASE_POSITION = (0.5, -0.8)

RETURN_WAYPOINTS = [
    (6.15, -0.8),
    (1.0, -0.8),
]

WAYPOINT_REACHED_RADIUS = 0.3
LOOP_WAYPOINTS = False

# control parameters
SPEED = 80
SERVO_CENTER = 104
STEER_MIN = 90.0
STEER_MAX = 118.0
STEERING_GAIN = -10.0
MAX_HEADING_ERROR_TO_DRIVE = math.radians(90)

# Path follower: servo ≈ SERVO_CENTER + (-CROSS_GAIN * y_err - HEADING_GAIN * theta)
#   y_err  = signed cross-track error (m) to the line prev_wp → target_wp (+ = left of path)
#   theta  = heading error (rad) = path_tangent_heading - robot.theta
# Set False to use legacy STEERING_GAIN * heading_error(robot, target) toward the point.
USE_PATH_TWO_TERM_STEERING = True
CROSS_GAIN = 15.0      # lateral termy
HEADING_GAIN = 5.0   # heading term (rad → servo units, same scale order as old STEERING_GAIN)

# plotting and frames
PLOT_EVERY_N = 2
PLOT_ENABLED = True

frame_dir = "pf_frames"
os.makedirs(frame_dir, exist_ok=True)
frame_count = 0

# vision state
OBJECT_DETECTED = False
CAMERA = None
DETECTION_STREAK = 0
CAMERA_READ_FAILS = 0


def compute_vision_drive_commands(fw: int, fh: int, info: dict):
    """Map blob bbox to (speed, steering_absolute). norm_x:-1=left +1=right."""
    gx = getattr(parameters, "vision_steering_gain", -48.0)
    v_hi = getattr(parameters, "vision_approach_speed", SPEED)
    v_lo = getattr(parameters, "vision_approach_slow_speed", 18)
    err_slow = getattr(parameters, "vision_approach_error_slow_frac", 0.35)
    stop_frac = getattr(parameters, "vision_stop_bbox_area_frac", 0.09)

    if "bbox" not in info:
        vs = int(round(float(getattr(parameters, "vision_approach_speed", SPEED))))
        return vs, SERVO_CENTER, False
    x, y, bw, bh = info["bbox"]
    cx = x + bw * 0.5
    half_w = max(1.0, fw * 0.5)
    norm_x = (cx - half_w) / half_w

    steer = SERVO_CENTER + gx * norm_x
    steer = max(STEER_MIN, min(STEER_MAX, steer))

    ba = float(info.get("bbox_area_frac") or ((bw * bh) / max(1, fw * fh)))
    if ba >= stop_frac:
        return 0, SERVO_CENTER, True

    err_f = abs(norm_x)
    if err_f <= err_slow:
        spd = float(v_hi)
    else:
        t = min(1.0, err_f / max(err_slow * 5.0, 1e-6))
        spd = float(v_hi) + t * (float(v_lo) - float(v_hi))
    spd = max(0.0, min(abs(float(v_hi)), spd))
    int_spd = int(round(spd))
    return int_spd, steer, False


def update_vision():
    """One camera grab + detector; overlay; returns flags and frame size."""
    global OBJECT_DETECTED, CAMERA, DETECTION_STREAK, CAMERA_READ_FAILS

    if OBJECT_DETECTED:
        return {"debounced": True, "found": True, "info": {}, "fw": 640, "fh": 480}

    if CAMERA is None:
        return {"debounced": False, "found": False, "info": {}, "fw": 1, "fh": 1}

    retries = int(getattr(parameters, "camera_read_retries", 5))
    frame = None
    ok = False
    for _ in range(max(1, retries)):
        ok, frame = CAMERA.read()
        if ok and frame is not None:
            break
        time.sleep(0.01)

    if not ok or frame is None:
        DETECTION_STREAK = 0
        CAMERA_READ_FAILS += 1
        if CAMERA_READ_FAILS >= 10:
            try:
                CAMERA.release()
            except Exception:
                pass
            CAMERA = init_camera()
            CAMERA_READ_FAILS = 0
        return {"debounced": False, "found": False, "info": {}, "fw": 1, "fh": 1}

    CAMERA_READ_FAILS = 0
    fh, fw = frame.shape[:2]
    found, info = detect_colored_ball(frame)

    if found:
        DETECTION_STREAK += 1
    else:
        DETECTION_STREAK = 0

    if getattr(parameters, "vision_debug_windows", True):
        dbg = frame.copy()
        cx = fw // 2
        cv2.line(dbg, (cx, 0), (cx, fh), (255, 255, 0), 1, cv2.LINE_AA)
        if found and "bbox" in info:
            x, y, w, h = info["bbox"]
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            mean_s = info.get("mean_saturation")
            ba = info.get("bbox_area_frac")
            cw = info.get("bbox_w_frac")
            ch = info.get("bbox_h_frac")
            ca = info.get("contour_area_frac")
            extras = ""
            if mean_s is not None and ba is not None and cw is not None and ch is not None:
                frac_bits = f" ca={ca:.2f}" if ca is not None else ""
                extras = f" ms={mean_s:.0f} ba={ba:.2f} w={cw:.2f} h={ch:.2f}{frac_bits}"
            msg = (
                f"area={info['area']:.0f} "
                f"circ={info['circularity']:.2f} "
                f"solid={info.get('solidity', 0):.2f}"
                + extras
            )
            cv2.putText(
                dbg,
                msg,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.imshow("camera", dbg)
        if "mask" in info:
            cv2.imshow("mask", info["mask"])
        cv2.waitKey(1)

    required = int(getattr(parameters, "required_detection_frames", 3))
    debounced = DETECTION_STREAK >= max(1, required)
    return {"debounced": debounced, "found": found, "info": info, "fw": fw, "fh": fh}


def reset_vision_detection_streak():
    """Clear debounce streak so a new detection can fire after pressing ``r``."""
    global DETECTION_STREAK
    DETECTION_STREAK = 0


def init_camera():
    cap, used_idx = open_cv_camera(parameters)
    if cap is None:
        return None
    if used_idx >= 0:
        print(f"Opened camera index {used_idx}", flush=True)
    warmup = int(getattr(parameters, "camera_warmup_frames", 15))
    for _ in range(max(0, warmup)):
        cap.read()
    return cap


# serial port discovery
def find_serial_port():
    env = os.environ.get("ROBOT_SERIAL_PORT") or os.environ.get("ARDUINO_SERIAL_PORT")
    if env:
        env = env.strip()
        if os.path.exists(env):
            print("Using serial port from env:", env)
            return env
        raise RuntimeError(f"Serial port from env does not exist: {env}")

    if getattr(parameters, "serial_port", None):
        p = str(parameters.serial_port).strip()
        if os.path.exists(p):
            print("Using serial port from parameters.py:", p)
            return p
        raise RuntimeError(f"parameters.serial_port does not exist: {p}")

    patterns = [
        "/dev/cu.usbmodem*",
        "/dev/cu.usbserial*",
        "/dev/cu.wchusbserial*",
        "/dev/cu.SLAB_USBtoUART*",
        "/dev/cu.usbserial-*",
    ]
    seen = set()
    candidates = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            if path not in seen:
                seen.add(path)
                candidates.append(path)

    if candidates:
        print("Using serial port:", candidates[0])
        return candidates[0]

    all_cu = sorted(glob.glob("/dev/cu.*"))
    hint = "\n".join(all_cu) if all_cu else "(none)"
    raise RuntimeError(
        "No serial port matched common patterns.\n"
        "Set ROBOT_SERIAL_PORT or parameters.serial_port.\n"
        "/dev/cu.*:\n" + hint
    )


def send_control(ser, speed, steering_absolute):
    """Arduino: steering_angle_center + relative (same as v2 / robot_serial_usb)."""
    steering_absolute = max(STEER_MIN, min(STEER_MAX, steering_absolute))
    steering_rel = int(steering_absolute) - SERVO_CENTER
    msg = f"{int(speed)},{steering_rel}\n"
    ser.write(msg.encode("utf-8"))
    ser.flush()


def read_serial_sensor(ser):
    """Drain buffer; return latest TEL: line as RobotSensorSignal."""
    latest = None
    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
        except Exception:
            break
        if not line:
            break
        if not line.startswith("TEL:"):
            continue
        try:
            parts = list(map(float, line[4:].split(",")))
            latest = robot_python_code.RobotSensorSignal(parts)
        except Exception:
            pass
    return latest


def get_key_nonblocking():
    import select

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
        if rlist:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ""


# navigation
def get_distance_to_waypoint(current_state, waypoint):
    return math.sqrt(
        (waypoint[0] - current_state.x) ** 2 + (waypoint[1] - current_state.y) ** 2
    )


def heading_error(current_state, waypoint):
    dx = waypoint[0] - current_state.x
    dy = waypoint[1] - current_state.y
    desired = math.atan2(dy, dx)
    return angle_wrap(desired - current_state.theta)


def signed_cross_track_error_m(current_state, prev_wp, target_wp):
    """Signed perpendicular distance (m) from robot to infinite line prev → target (+ = left of forward path)."""
    x0, y0 = float(prev_wp[0]), float(prev_wp[1])
    x1, y1 = float(target_wp[0]), float(target_wp[1])
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return 0.0
    rx = float(current_state.x) - x0
    ry = float(current_state.y) - y0
    return (rx * dy - ry * dx) / L


def path_heading_error_rad(current_state, prev_wp, target_wp):
    """Heading error (rad): path tangent (prev→target) minus robot heading."""
    dx = float(target_wp[0]) - float(prev_wp[0])
    dy = float(target_wp[1]) - float(prev_wp[1])
    if abs(dx) + abs(dy) < 1e-12:
        return heading_error(current_state, target_wp)
    path_h = math.atan2(dy, dx)
    return angle_wrap(path_h - float(current_state.theta))


def compute_steering(current_state, prev_wp, target_wp):
    if not USE_PATH_TWO_TERM_STEERING:
        err = heading_error(current_state, target_wp)
        servo = SERVO_CENTER + STEERING_GAIN * err
        return max(STEER_MIN, min(STEER_MAX, servo))

    y_err = signed_cross_track_error_m(current_state, prev_wp, target_wp)
    theta = path_heading_error_rad(current_state, prev_wp, target_wp)
    correction = -CROSS_GAIN * y_err - HEADING_GAIN * theta
    servo = SERVO_CENTER + correction
    return max(STEER_MIN, min(STEER_MAX, servo))


# main autonomous exploration function
def autonomous_explore():
    global OBJECT_DETECTED, CAMERA, frame_count

    last_serial_warn = 0.0
    halt_after_object = False
    prev_vis_debounced = False

    serial_port = find_serial_port()
    ser = serial.Serial(serial_port, 115200, timeout=0.05)
    time.sleep(1.0)
    ser.reset_input_buffer()

    robot = Robot()
    robot.msg_sender = None
    robot.msg_receiver = None

    print("Waiting for first sensor reading to sync encoder + init particles...")
    send_control(ser, 0, SERVO_CENTER)
    while True:
        sensor = read_serial_sensor(ser)
        if sensor is not None:
            robot.robot_sensor_signal = sensor
            start_state = State(0.5, -0.8, 0.0)
            start_stdev = State(0.15, 0.1, 0.15)
            robot.particle_filter.particle_set.particle_list = []
            robot.particle_filter.particle_set.generate_initial_state_particles(
                start_state, start_stdev
            )
            robot.particle_filter.particle_set.update_mean_state()
            robot.particle_filter.last_encoder_counts = sensor.encoder_counts
            print(f"  Synced encoder at {sensor.encoder_counts}")
            print(
                f"  Particles initialized at ({start_state.x}, {start_state.y}, {start_state.theta})"
            )
            break
        send_control(ser, 0, SERVO_CENTER)
        time.sleep(0.05)

    map_obj = Map(parameters.wall_corner_list)
    pf_plot = ParticleFilterPlot(map_obj) if PLOT_ENABLED else None

    CAMERA = init_camera()
    if CAMERA is None:
        print(
            "[WARN] Camera not available. Vision disabled; use key 'o' to simulate object."
        )
    else:
        print("[INFO] Camera stream initialized.")

    waypoints = list(EXPLORE_WAYPOINTS)
    current_wp_index = 0
    loop_count = 0
    returning_to_base = False
    object_location = None
    sensor_count = 0

    current_drive_speed = 0.0
    current_steer_cmd = float(SERVO_CENTER)
    current_pos = robot.particle_filter.particle_set.mean_state

    print(
        f"""
===========================================
  v3: VISION + PF WAYPOINTS (v2 navigation)
===========================================
  Waypoints : {len(EXPLORE_WAYPOINTS)}
  Speed     : {SPEED}
  Servo ctr : {SERVO_CENTER} (match Arduino steering_angle_center)
  Map/PF    : parameters.wall_corner_list
  Initial   : x={current_pos.x:.2f} y={current_pos.y:.2f}

  q = emergency stop
  o = simulate object → full stop + return-to-base waypoints
  r = after vision OBJECT DETECTED full stop, resume waypoint driving
  Stable vision (debounced) → print OBJECT DETECTED + full stop (press r to continue)
===========================================
"""
    )

    keyboard_return = False
    try:
        while True:
            key = get_key_nonblocking()
            if key == "q":
                print("\n\n[!] Emergency stop.")
                break
            elif key == "o":
                keyboard_return = True
                print("\n\n[!] Key 'o': return-to-base (v2 style).")
            elif key == "r":
                if halt_after_object:
                    halt_after_object = False
                    prev_vis_debounced = False
                    reset_vision_detection_streak()
                    print("\n[RESUME] Object halt cleared; waypoint driving active again.\n", flush=True)

            send_control(ser, current_drive_speed, current_steer_cmd)

            sensor = read_serial_sensor(ser)
            sensor_ok = sensor is not None
            if sensor_ok:
                sensor_count += 1
                robot.robot_sensor_signal = sensor
                robot.update_state_estimate()
                current_pos = robot.particle_filter.particle_set.mean_state

            vis = update_vision()

            # keyboard 'o': immediate return (v2), independent of vision debounce 
            if not returning_to_base and keyboard_return:
                keyboard_return = False
                halt_after_object = False
                object_location = (current_pos.x, current_pos.y)
                print(
                    "\n\n*** OBJECT DETECTED (key 'o') ***\n"
                    f"  Estimated pose: x={object_location[0]:.2f} y={object_location[1]:.2f}\n"
                    "  Full stop, then return-to-base waypoints.\n",
                    flush=True,
                )
                current_drive_speed = 0.0
                current_steer_cmd = float(SERVO_CENTER)
                send_control(ser, 0, SERVO_CENTER)
                time.sleep(0.3)
                waypoints = list(RETURN_WAYPOINTS)
                current_wp_index = 0
                returning_to_base = True
                OBJECT_DETECTED = False

            # stable vision detection (debounced): print + full stop 
            debounced_now = vis["debounced"]
            if (
                not returning_to_base
                and not halt_after_object
                and debounced_now
                and not prev_vis_debounced
            ):
                object_location = (current_pos.x, current_pos.y)
                halt_after_object = True
                current_drive_speed = 0.0
                current_steer_cmd = float(SERVO_CENTER)
                print(
                    "\n\n*** OBJECT DETECTED (vision) ***\n"
                    f"  Estimated pose: x={object_location[0]:.2f} y={object_location[1]:.2f}\n"
                    "  Full stop: speed 0, steering centered.\n"
                    "  Press 'r' to resume waypoints, 'q' to quit.\n",
                    flush=True,
                )
                send_control(ser, 0, SERVO_CENTER)
                time.sleep(0.2)
            prev_vis_debounced = debounced_now

            if halt_after_object and not returning_to_base:
                current_drive_speed = 0.0
                current_steer_cmd = float(SERVO_CENTER)
                send_control(ser, 0, SERVO_CENTER)
                loop_count += 1
                if (
                    sensor_ok
                    and PLOT_ENABLED
                    and pf_plot
                    and loop_count % PLOT_EVERY_N == 0
                ):
                    pf_plot.update(
                        current_pos,
                        robot.particle_filter.particle_set,
                        sensor,
                        False,
                    )
                    plt.savefig(
                        os.path.join(frame_dir, f"frame_{frame_count:04d}.png"),
                        dpi=150,
                        bbox_inches="tight",
                    )
                    frame_count += 1
                time.sleep(0.1)
                continue

            need_serial = getattr(
                parameters, "require_serial_sensor_for_explore", True
            )
            if not sensor_ok and need_serial and not halt_after_object:
                tnow = time.time()
                if tnow - last_serial_warn > 2.0:
                    print(
                        "\n[WARN] No TEL: line; PF/waypoints idle. "
                        "Flash robot_serial_usb.ino.",
                        flush=True,
                    )
                    last_serial_warn = tnow
                time.sleep(0.02)
                continue

            if current_wp_index >= len(waypoints):
                if returning_to_base:
                    print(
                        f"\n\n{'='*50}"
                        f"\n  ARRIVED AT BASE"
                        f"\n  Object ~ ({object_location[0]:.2f}, {object_location[1]:.2f})"
                        f"\n{'='*50}"
                    )
                elif LOOP_WAYPOINTS:
                    current_wp_index = 0
                    print("\n\n--- Looping exploration ---")
                else:
                    print("\n\n--- Exploration complete. ---")
                current_drive_speed = 0.0
                current_steer_cmd = float(SERVO_CENTER)
                send_control(ser, 0, SERVO_CENTER)
                break

            target = waypoints[current_wp_index]
            if current_wp_index > 0:
                prev_wp = waypoints[current_wp_index - 1]
            else:
                prev_wp = BASE_POSITION

            dist = get_distance_to_waypoint(current_pos, target)
            if USE_PATH_TWO_TERM_STEERING:
                h_err = path_heading_error_rad(current_pos, prev_wp, target)
            else:
                h_err = heading_error(current_pos, target)

            if dist < WAYPOINT_REACHED_RADIUS:
                label = "RTN" if returning_to_base else "EXP"
                print(
                    f"\n  >> [{label}] Reached WP {current_wp_index} "
                    f"at ({target[0]:.1f}, {target[1]:.1f})"
                )
                current_wp_index += 1
                continue

            current_steer_cmd = compute_steering(current_pos, prev_wp, target)
            if abs(h_err) > MAX_HEADING_ERROR_TO_DRIVE:
                current_drive_speed = SPEED * 0.8
            else:
                current_drive_speed = float(SPEED)

            send_control(ser, current_drive_speed, current_steer_cmd)

            loop_count += 1
            if PLOT_ENABLED and pf_plot and sensor_ok and loop_count % PLOT_EVERY_N == 0:
                pf_plot.update(
                    current_pos,
                    robot.particle_filter.particle_set,
                    sensor,
                    False,
                )
                plt.savefig(
                    os.path.join(frame_dir, f"frame_{frame_count:04d}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                frame_count += 1

            mode = "RTN" if returning_to_base else "EXP"
            steer_rel = int(current_steer_cmd) - SERVO_CENTER
            y_ct = (
                signed_cross_track_error_m(current_pos, prev_wp, target)
                if USE_PATH_TWO_TERM_STEERING
                else 0.0
            )
            print(
                f"  [{mode}] WP {current_wp_index}/{len(waypoints)-1}"
                f" | x={current_pos.x:+.2f} y={current_pos.y:+.2f} th={math.degrees(current_pos.theta):+.1f}"
                f" | tgt ({target[0]:.1f},{target[1]:.1f})"
                f" | dist={dist:.2f}m"
                f" | yct={y_ct:+.2f}m hdg={math.degrees(h_err):+.1f}"
                f" | spd={current_drive_speed:.0f} str={steer_rel:+d}"
                f" | TEL#{sensor_count} frm={frame_count}          ",
                end="\r",
            )

            time.sleep(0.1)

    finally:
        send_control(ser, 0, SERVO_CENTER)
        if CAMERA is not None:
            try:
                CAMERA.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        ser.close()

        print(f"\n\n{frame_count} frames saved to {frame_dir}/")
        if frame_count > 0:
            try:
                from PIL import Image

                frames = []
                for i in range(frame_count):
                    img = Image.open(os.path.join(frame_dir, f"frame_{i:04d}.png"))
                    frames.append(img)
                gif_path = "pf_replay.gif"
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=200,
                    loop=0,
                )
                print(f"Animated replay saved to {gif_path}")
            except ImportError:
                print("Install Pillow for GIF: pip install Pillow")
            except Exception as e:
                print(f"GIF failed: {e}")

        print("Robot stopped. Serial closed.")


if __name__ == "__main__":
    autonomous_explore()
