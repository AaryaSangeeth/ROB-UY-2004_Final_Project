#starting position fixed to 0.5,-0.8
import time
import sys
import tty
import termios
import serial
import glob
import os
import math
import csv
from datetime import datetime

from robot import Robot
import robot_python_code
import parameters
from particle_filter import Map, ParticleFilterPlot, State, angle_wrap
import matplotlib
import matplotlib.pyplot as plt

# corrdidor mapping
wall_corner_list = [
    [0, 0, 9.6, 0],
    [9.6, 0, 9.6, -1.63],
    [9.6, -1.63, 7.1, -1.63],
    [7.1, -1.63, 7.1, -8.4],
    [7.1, -8.4, 5.2, -8.4],
    [5.2, -8.4, 5.2, -1.63],
    [5.2, -1.63, 0, -1.63],
    [0, -1.63, 0, 0],
]

# waypoints for exploration
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
RETURN_WAYPOINTS = [(6.15, -0.8), (1.0, -0.8)]
WAYPOINT_REACHED_RADIUS = 0.3
LOOP_WAYPOINTS = False

#control parameters
SPEED        = 110
SERVO_CENTER = 104
STEER_MIN    = 90.0
STEER_MAX    = 118.0
STEERING_GAIN = -27.0


PLOT_EVERY_N = 2
PLOT_ENABLED = True

frame_dir = "pf_frames"
os.makedirs(frame_dir, exist_ok=True)
frame_count = 0

# set object detection to be false for testing nav only
OBJECT_DETECTED = False

def check_for_object():
    global OBJECT_DETECTED
    return OBJECT_DETECTED

# helper function to find the serial port
def find_serial_port():
    candidates = (
        sorted(glob.glob("/dev/cu.usbmodem*"))
        + sorted(glob.glob("/dev/cu.usbserial*"))
    )
    if not candidates:
        raise RuntimeError("No serial port found.")
    print("Using serial port:", candidates[0])
    return candidates[0]


def send_control(ser, speed, steering_absolute):
    steering_absolute = max(STEER_MIN, min(STEER_MAX, steering_absolute))
    steering_rel = int(steering_absolute) - SERVO_CENTER
    msg = f"{int(speed)},{steering_rel}\n"
    ser.write(msg.encode("utf-8"))
    ser.flush()


def read_serial_sensor(ser):
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


#helper functions for navigation
def get_distance_to_waypoint(current_state, waypoint):
    return math.sqrt(
        (waypoint[0] - current_state.x) ** 2
        + (waypoint[1] - current_state.y) ** 2
    )


def heading_error(current_state, waypoint):
    dx = waypoint[0] - current_state.x
    dy = waypoint[1] - current_state.y
    return angle_wrap(math.atan2(dy, dx) - current_state.theta)


def compute_steering(current_state, waypoint):
    err = heading_error(current_state, waypoint)
    servo = SERVO_CENTER + STEERING_GAIN * err
    return max(STEER_MIN, min(STEER_MAX, servo))



class RunLogger:
    FIELDS = [
        "t", "wp_index",
        "pf_x", "pf_y", "pf_theta",
        "tgt_x", "tgt_y",
        "dist", "hdg_err",
        "speed", "steer_rel",
        "delta_enc",
    ]

    def __init__(self):
        self.rows = []
        self.t0   = time.perf_counter()

    def log(self, wp_index, pf_x, pf_y, pf_theta,
            tgt_x, tgt_y, dist, hdg_err, speed, steer_rel, delta_enc):
        self.rows.append({
            "t":         round(time.perf_counter() - self.t0, 3),
            "wp_index":  wp_index,
            "pf_x":      round(pf_x, 4),
            "pf_y":      round(pf_y, 4),
            "pf_theta":  round(math.degrees(pf_theta), 2),
            "tgt_x":     tgt_x,
            "tgt_y":     tgt_y,
            "dist":      round(dist, 4),
            "hdg_err":   round(math.degrees(hdg_err), 2),
            "speed":     speed,
            "steer_rel": steer_rel,
            "delta_enc": delta_enc,
        })

    def save(self, path="run_log.csv"):
        if not self.rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        print(f"  Run log saved → {path}  ({len(self.rows)} rows)")


# main autonomous exploration function
def autonomous_explore():
    global OBJECT_DETECTED, frame_count

    logger = RunLogger()

    # Serial connection to the robot
    serial_port = find_serial_port()
    ser = serial.Serial(serial_port, 115200, timeout=0.05)
    time.sleep(1.0)
    ser.reset_input_buffer()

    # create robot instance
    robot = Robot()
    robot.msg_sender   = None
    robot.msg_receiver = None

    
    print("Waiting for first sensor reading to sync encoder...")
    send_control(ser, 0, SERVO_CENTER)
    last_enc = 0
    while True:
        sensor = read_serial_sensor(ser)
        if sensor is not None:
            robot.robot_sensor_signal = sensor
            start_state = State(0.5, -0.8, 0.0)
            start_stdev = State(0.15, 0.1, 0.15)
            robot.particle_filter.particle_set.particle_list = []
            robot.particle_filter.particle_set.generate_initial_state_particles(
                start_state, start_stdev)
            robot.particle_filter.particle_set.update_mean_state()
            robot.particle_filter.last_encoder_counts = sensor.encoder_counts
            last_enc = sensor.encoder_counts
            print(f"  Synced encoder at {sensor.encoder_counts}")
            print(f"  Particles initialized at ({start_state.x}, {start_state.y}, {start_state.theta})")
            break
        send_control(ser, 0, SERVO_CENTER)
        time.sleep(0.05)

    # Map & plot 
    map_obj = Map(wall_corner_list)
    pf_plot = ParticleFilterPlot(map_obj) if PLOT_ENABLED else None

    #  Navigation state 
    waypoints        = list(EXPLORE_WAYPOINTS)
    current_wp_index = 0
    loop_count       = 0
    returning_to_base = False
    object_location  = None
    sensor_count     = 0

    current_drive_speed = 0.0
    current_steer_cmd   = SERVO_CENTER
    current_pos = robot.particle_filter.particle_set.mean_state

    print(f"""
===========================================
  AUTONOMOUS CORRIDOR EXPLORATION
===========================================
  Waypoints : {len(EXPLORE_WAYPOINTS)}
  Speed     : {SPEED}
  Center    : {SERVO_CENTER}
  Initial   : x={current_pos.x:.2f} y={current_pos.y:.2f}

  q = emergency stop
  o = simulate object detection
===========================================
""")

    try:
        while True:
            
            key = get_key_nonblocking()
            if key == "q":
                print("\n\n[!] Emergency stop.")
                break
            elif key == "o":
                OBJECT_DETECTED = True
                print("\n\n[!] Object detection simulated!")

            
            send_control(ser, current_drive_speed, current_steer_cmd)

            # get sensor data from the robot
            sensor    = read_serial_sensor(ser)
            delta_enc = 0

            if sensor is not None:
                sensor_count += 1
                delta_enc = int(sensor.encoder_counts - last_enc)
                last_enc  = sensor.encoder_counts
                robot.robot_sensor_signal = sensor
                robot.update_state_estimate()
                current_pos = robot.particle_filter.particle_set.mean_state

            # object detection logic
            if not returning_to_base and check_for_object():
                object_location = (current_pos.x, current_pos.y)
                print(f"\n\n{'='*50}"
                      f"\n  OBJECT DETECTED at x={object_location[0]:.2f} y={object_location[1]:.2f}"
                      f"\n  Returning to base\n{'='*50}")
                current_drive_speed = 0.0
                current_steer_cmd   = SERVO_CENTER
                send_control(ser, 0, SERVO_CENTER)
                time.sleep(1.0)
                waypoints        = list(RETURN_WAYPOINTS)
                current_wp_index = 0
                returning_to_base = True
                OBJECT_DETECTED  = False


            if current_wp_index >= len(waypoints):
                if returning_to_base:
                    print(f"\n\n{'='*50}"
                          f"\n  ARRIVED AT BASE"
                          f"\n  Object was at x={object_location[0]:.2f} y={object_location[1]:.2f}"
                          f"\n{'='*50}")
                elif LOOP_WAYPOINTS:
                    current_wp_index = 0
                    print("\n\n--- Looping exploration ---")
                else:
                    print("\n\n--- Exploration complete. No object found. ---")
                current_drive_speed = 0.0
                current_steer_cmd   = SERVO_CENTER
                send_control(ser, 0, SERVO_CENTER)
                break

            # check if all waypoints have been passed# ---------- Navigate ----------
            target = waypoints[current_wp_index]
            dist   = get_distance_to_waypoint(current_pos, target)
            h_err  = heading_error(current_pos, target)

            if dist < WAYPOINT_REACHED_RADIUS:
                label = "RTN" if returning_to_base else "EXP"
                print(f"\n  >> [{label}] Reached WP {current_wp_index} "
                      f"at ({target[0]:.1f}, {target[1]:.1f})")
                current_wp_index += 1
                continue

            # compute drive command 
            current_steer_cmd   = compute_steering(current_pos, target)
            current_drive_speed = SPEED   # constant speed, no heading gating

            steer_rel = int(current_steer_cmd) - SERVO_CENTER

            # log data 
            logger.log(
                current_wp_index,
                current_pos.x, current_pos.y, current_pos.theta,
                target[0], target[1],
                dist, h_err,
                current_drive_speed, steer_rel, delta_enc,
            )

            # plot and save frames
            loop_count += 1
            if PLOT_ENABLED and pf_plot and sensor is not None and loop_count % PLOT_EVERY_N == 0:
                pf_plot.update(
                    current_pos,
                    robot.particle_filter.particle_set,
                    sensor,
                    False,
                )
                plt.savefig(
                    os.path.join(frame_dir, f"frame_{frame_count:04d}.png"),
                    dpi=150, bbox_inches="tight",
                )
                frame_count += 1

            # debug
            mode = "RTN" if returning_to_base else "EXP"
            print(
                f"  [{mode}] WP {current_wp_index}/{len(waypoints)-1}"
                f" | x={current_pos.x:+.2f} y={current_pos.y:+.2f} th={math.degrees(current_pos.theta):+.1f}"
                f" | tgt ({target[0]:.1f},{target[1]:.1f})"
                f" | dist={dist:.2f}m hdg={math.degrees(h_err):+.1f}"
                f" | spd={current_drive_speed:.0f} str={steer_rel:+d}"
                f" | TEL#{sensor_count} frm={frame_count}"
                f"          ",
                end="\r",
            )

            time.sleep(0.1)

    finally:
        send_control(ser, 0, SERVO_CENTER)
        ser.close()

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"run_log_{ts}.csv"
        logger.save(log_path)

        print(f"\n{frame_count} frames saved to {frame_dir}/")
        if frame_count > 0:
            try:
                from PIL import Image
                frames = [Image.open(os.path.join(frame_dir, f"frame_{i:04d}.png"))
                          for i in range(frame_count)]
                gif_path = "pf_replay.gif"
                frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                               duration=200, loop=0)
                print(f"Animated replay saved to {gif_path}")
            except Exception as e:
                print(f"GIF failed: {e}")

        print(f"Run log → {log_path}")
        print("Robot stopped. Serial closed.")


if __name__ == "__main__":
    autonomous_explore()