# Autonomous corridor navigation with particle-filter localization and vision

**NYU ROB-GY 6213 — Robot Localization and Navigation** · Final project codebase

This repository contains an integrated mobile-robot stack: **particle-filter** localization using LiDAR (and encoder motion), **waypoint navigation** in a known  map, **USB serial** control of a differential platform with ackerman servo steering, and a **vision** pipeline for colored-object detection (HSV) used for approach and stop behaviors. Configuration is centralized so maps, serial ports, cameras, and detector thresholds can be tuned without touching core logic.

---

## Highlights

- **Localization:** Monte Carlo particle filter with map lines from surveyed walls; motion and sensor models wired through `particle_filter.py`, `motion_models.py`, and `data_handling.py`.
- **Planning / control:** Corridor exploration via ordered waypoints; optional path-style steering; vision mode overrides driving when a target is detected.
- **Vision:** OpenCV capture helpers (`vision/camera_util.py`), HSV-based detection (`vision/vision_detection.py`), live tuning (`python -m vision.tune_ball_hsv`), and continuity-camera checks (`python -m vision.test_continuity_camera`).
- **Robot interface:** Serial protocol and telemetry parsing in `robot.py`, `robot_python_code.py`; firmware under **`robot_arduino_code/`** (LiDAR driver sources + sketch).

---

## Repository layout

Top-level folders are **siblings** (flat layout):

| Path | Role |
|------|------|
| **`scripts/`** | Application entry points and navigation stack: `full_pipeline_v3.py`, `auto_nav_data_logging.py`, `plot_run.py`, `particle_filter.py`, `robot.py`, `data_handling.py`, etc. |
| **`vision/`** | Importable **`vision`** package: **`parameters.py`** (canonical config), `camera_util.py`, `vision_detection.py`, `tune_ball_hsv.py`, `test_continuity_camera.py`. |
| **`scripts/parameters.py`** | Thin re-export of **`vision/parameters.py`** so `import parameters` works when you run from `scripts/`. **Edit `vision/parameters.py` for real changes.** |
| **`robot_arduino_code/`** | Arduino firmware (e.g. USB bridge, `RPLidar` sources, sketches). Align baud rate, steering trim, and telemetry format with Python. |
| **`docs/`** | Project deliverables (e.g. proposal, report, figures). |
| **`req/`** | Optional place for pinned dependency files; this repo also ships **`requirements.txt`** at the root. |
| **`debug/`** | Ad-hoc debugging utilities. |
| **`base_code_lab_01/` … `base_code_lab_04/`** | Course lab reference snapshots (legacy layouts). |

---

## Prerequisites

- **Python** 3.10+  
- **Arduino IDE** (or PlatformIO) for flashing firmware  
- USB serial to the robot; port names vary (`/dev/tty.usbserial-*`, `/dev/ttyUSB*`, `COM*` on Windows)

---

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies include NumPy, OpenCV (`opencv-python`), PySerial, Matplotlib, and Pillow.

---

## Configuration

| Topic | Where to edit |
|--------|----------------|
| Map walls, particle count, camera indices, HSV ranges, vision gains, logging paths | **`vision/parameters.py`** |
| Same values visible as top-level `parameters` in scripts | Automatically via **`scripts/parameters.py`** re-export |

Before demos, confirm **serial port**, **camera index / URL**, **wall list**, and **servo / speed limits** match hardware and the flashed sketch.

---

## Running the main pipeline

Primary integrated run (PF waypoint exploration + vision):

```bash
cd scripts
python full_pipeline_v3.py
```

`full_pipeline_v3.py` adds the repo root to `sys.path` so `import vision.*` resolves. Other scripts typically load **`scripts/parameters.py`** first, which also ensures the repo root is on the path.

**Other useful commands:**

```bash
cd scripts
python auto_nav_data_logging.py    # navigation with logging
python plot_run.py                   # plot logged sessions (adjust paths inside script as needed)
```

**Vision tuning (from repo root):**

```bash
python -m vision.tune_ball_hsv
python -m vision.test_continuity_camera
```

---

## Firmware

Flash the sketch appropriate for your wiring from **`robot_arduino_code/`**. Python expects compatible **`TEL:`** telemetry and control commands at the baud rate set in code and parameters; adjust **`steering_angle_center`** (or equivalent) in firmware and in tuning notes so commanded steering matches the physical robot.

---

## Deliverables and housekeeping

- **`docs/`** holds written reports and PDFs submitted for the course.  
- Do **not** commit **`.venv/`** or huge generated artifacts (long CSV logs, frame dumps, GIFs); regenerate from runs or list paths in `.gitignore`.

---

## Course

Submitted for final course project for  **ROB-GY 6213: Robot Localization and Navigation**, NYU.
