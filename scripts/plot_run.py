"""
plot_run.py — post-run analysis for autonomous_explore.py logs (doc-15 version).

Controller:
    heading_err (rad) = atan2(dy, dx) - theta
    steer_rel = clamp(STEERING_GAIN * heading_err, STEER_MIN-CENTER, STEER_MAX-CENTER)

Usage:
    python plot_run.py                   # auto-finds latest run_log_*.csv
    python plot_run.py run_log_XYZ.csv  # specific file
"""

import sys, glob, math, csv, os, subprocess, platform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator

# controller constants must match autonomous_explore.py 
STEERING_GAIN = -27.0
SERVO_CENTER  = 104
STEER_MIN     = 90.0
STEER_MAX     = 118.0
STEER_LO      = STEER_MIN - SERVO_CENTER   # = -14  (max left relative)
STEER_HI      = STEER_MAX - SERVO_CENTER   # = +14  (max right relative)

WALLS = [
    [0,0,9.6,0], [9.6,0,9.6,-1.63], [9.6,-1.63,7.1,-1.63],
    [7.1,-1.63,7.1,-8.4], [7.1,-8.4,5.2,-8.4], [5.2,-8.4,5.2,-1.63],
    [5.2,-1.63,0,-1.63],  [0,-1.63,0,0],
]
WPS = [
    (1.0,-0.8),(3.0,-0.8),(5.0,-0.8),(7.0,-0.8),(9.0,-0.8),
    (6.15,-0.8),(6.15,-2.5),(6.15,-4.5),(6.15,-6.5),(6.15,-7.8),
    (6.15,-4.5),(6.15,-2.0),(6.15,-0.8),
]

# load log file
def load_log(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def find_latest():
    files = sorted(glob.glob("run_log_*.csv"))
    if not files:
        raise FileNotFoundError("No run_log_*.csv found.")
    return files[-1]

def col(rows, key):
    return np.array([r[key] for r in rows])

# expected heading from waypoint sequence 
def exp_theta(wp_idx):
    out = []
    for idx in wp_idx.astype(int):
        if idx == 0:
            out.append(0.0)
        else:
            p = WPS[max(0, idx-1)]; c = WPS[min(idx, len(WPS)-1)]
            out.append(math.degrees(math.atan2(c[1]-p[1], c[0]-p[0])))
    return np.array(out)

# styling helpers 
def style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=9, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(axis="both", labelsize=8, which="both",
                   direction="in", left=True, labelleft=True)
    ax.grid(True, which="major", lw=0.5, alpha=0.5, color="#cccccc")
    ax.grid(True, which="minor", lw=0.25, alpha=0.3, color="#e8e8e8")
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def fix_ylim(ax, *arrays, pad=0.12, step=None):
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    span = max(hi - lo, 1e-3)
    ymin = lo - span * pad
    ymax = hi + span * pad
    ax.set_ylim(ymin, ymax)
    if step:
        ticks = np.arange(
            np.floor(ymin / step) * step,
            np.ceil(ymax  / step) * step + step * 0.01,
            step
        )
        ax.set_yticks(ticks)
        ax.yaxis.set_minor_locator(MultipleLocator(step / 5))
    ax.yaxis.set_tick_params(which="both", left=True, labelleft=True)

def add_wp_lines(ax, t, wp_idx):
    ymin, ymax = ax.get_ylim()
    prev = wp_idx[0]
    for i, idx in enumerate(wp_idx):
        if idx != prev:
            ax.axvline(t[i], color="#999999", lw=0.9, ls="--", alpha=0.55, zorder=1)
            ax.text(t[i]+0.2, ymin + (ymax-ymin)*0.03,
                    f"WP{int(idx)}", color="#666666",
                    fontsize=6.5, va="bottom", rotation=90, alpha=0.85)
            prev = idx

# main function to plot the run
def plot_all(rows, title):

    t         = col(rows, "t")
    pf_x      = col(rows, "pf_x")
    pf_y      = col(rows, "pf_y")
    pf_th     = col(rows, "pf_theta")    # degrees
    tgt_x     = col(rows, "tgt_x")
    tgt_y     = col(rows, "tgt_y")
    hdg_err   = col(rows, "hdg_err")     # degrees
    steer     = col(rows, "steer_rel")   # servo units (clamped, sent to Arduino)
    speed     = col(rows, "speed")
    wp_idx    = col(rows, "wp_index")
    delta_enc = col(rows, "delta_enc")

    ex_th = exp_theta(wp_idx)
    ex_x  = np.array([WPS[min(int(i), len(WPS)-1)][0] for i in wp_idx])
    ex_y  = np.array([WPS[min(int(i), len(WPS)-1)][1] for i in wp_idx])

    # Reconstruct control function terms
    hdg_rad      = np.deg2rad(hdg_err)
    unclamped    = STEERING_GAIN * hdg_rad               
    clamped      = np.clip(unclamped, STEER_LO, STEER_HI)  
    is_clamped   = np.abs(unclamped - clamped) > 0.5    

    # colours
    PF    = "#1f77b4"
    EXP   = "#d62728"
    STR   = "#ff7f0e"
    SPD   = "#2ca02c"
    UNCLP = "#9467bd"   # unclamped output
    CLPD  = "#e377c2"   # clamped / actual output
    CLPZ  = "#ff0000"   # clamping region highlight
    WALL  = "#222222"
    WPCLR = "#7b2d8b"

    # layout: 4 rows × 3 cols 
    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(f"Run Analysis  ·  {title}", fontsize=15,
                 fontweight="bold", y=0.997)

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.60, wspace=0.42,
        left=0.06, right=0.97,
        top=0.965, bottom=0.05,
    )

    # panel 1: T-corridor map 
    ax_map = fig.add_subplot(gs[0:2, 0:2])
    for w in WALLS:
        ax_map.plot([w[0],w[2]], [w[1],w[3]], color=WALL, lw=2.5,
                    solid_capstyle="round")
    for i,(wx,wy) in enumerate(WPS):
        ax_map.plot(wx, wy, "o", color=WPCLR, ms=7, zorder=4)
        ax_map.annotate(f"WP{i}", (wx,wy),
                        textcoords="offset points", xytext=(5,4),
                        fontsize=7.5, color=WPCLR, zorder=5)
    pts  = np.array([pf_x, pf_y]).T.reshape(-1,1,2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap="plasma", lw=2.5, alpha=0.9, zorder=3)
    lc.set_array(t[:-1])
    ax_map.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax_map, pad=0.02, fraction=0.03)
    cb.set_label("Elapsed time  (s)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax_map.plot(pf_x[0],  pf_y[0],  "^", color="green",  ms=12, zorder=6,
                label=f"Start  ({pf_x[0]:.2f}, {pf_y[0]:.2f}) m")
    ax_map.plot(pf_x[-1], pf_y[-1], "s", color="orange", ms=12, zorder=6,
                label=f"End    ({pf_x[-1]:.2f}, {pf_y[-1]:.2f}) m")
    ax_map.set_xlabel("x  (m)", fontsize=10, labelpad=4)
    ax_map.set_ylabel("y  (m)", fontsize=10, labelpad=4)
    ax_map.set_title("PF-estimated trajectory in T-corridor",
                     fontsize=11, fontweight="bold", pad=7)
    ax_map.set_aspect("equal")
    ax_map.xaxis.set_major_locator(MultipleLocator(1))
    ax_map.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax_map.yaxis.set_major_locator(MultipleLocator(1))
    ax_map.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax_map.tick_params(labelsize=9, which="both", direction="in")
    ax_map.grid(True, which="major", lw=0.5, alpha=0.45)
    ax_map.grid(True, which="minor", lw=0.25, alpha=0.3)
    ax_map.spines["top"].set_visible(False)
    ax_map.spines["right"].set_visible(False)
    ax_map.legend(fontsize=9, loc="lower right")

    # panel 2: X position vs time 
    ax_x = fig.add_subplot(gs[0, 2])
    ax_x.plot(t, ex_x, "--", color=EXP, lw=1.6, label="Target x  (waypoint)")
    ax_x.plot(t, pf_x, "-",  color=PF,  lw=2.2, label="PF x estimate")
    fix_ylim(ax_x, ex_x, pf_x, step=1.0)
    style_ax(ax_x, "Time  (s)", "x  (m)", "X position vs time")
    ax_x.xaxis.set_major_locator(MaxNLocator(7))
    add_wp_lines(ax_x, t, wp_idx)
    ax_x.legend(fontsize=8, loc="best", framealpha=0.85)

    # panel 3: Y position vs time 
    ax_y = fig.add_subplot(gs[1, 2])
    ax_y.plot(t, ex_y, "--", color=EXP, lw=1.6, label="Target y  (waypoint)")
    ax_y.plot(t, pf_y, "-",  color=PF,  lw=2.2, label="PF y estimate")
    ax_y.axhline(0, color="black", lw=0.7, ls=":", alpha=0.55, label="y = 0")
    fix_ylim(ax_y, ex_y, pf_y, [0], step=0.5)
    style_ax(ax_y, "Time  (s)", "y  (m)", "Y position vs time")
    ax_y.xaxis.set_major_locator(MaxNLocator(7))
    add_wp_lines(ax_y, t, wp_idx)
    ax_y.legend(fontsize=8, loc="best", framealpha=0.85)

        # ── PANEL 4 — Heading θ vs time ──────────────────────────
    ax_th = fig.add_subplot(gs[2, 0])
    ax_th.plot(t, ex_th, "--", color=EXP, lw=1.6, label="Expected heading")
    ax_th.plot(t, pf_th, "-",  color=PF,  lw=2.2, label="PF θ estimate")
    ax_th.axhline(  0, color="black",   lw=0.7, ls=":", alpha=0.6)
    ax_th.axhline( 90, color="#aaaaaa", lw=0.6, ls="--", alpha=0.5, label="±90°")
    ax_th.axhline(-90, color="#aaaaaa", lw=0.6, ls="--", alpha=0.5)
    fix_ylim(ax_th, ex_th, pf_th, [-180, 180], step=30.0)
    style_ax(ax_th, "Time  (s)", "θ  (deg)",
             "Robot heading θ vs time  (0° = right, CCW+)")
    ax_th.xaxis.set_major_locator(MaxNLocator(7))
    add_wp_lines(ax_th, t, wp_idx)
    ax_th.legend(fontsize=8, loc="best", framealpha=0.85)

    # panel 5: Heading error vs time 
    ax_he = fig.add_subplot(gs[2, 1])
    ax_he.fill_between(t, hdg_err, 0, where=(hdg_err>=0),
                       alpha=0.15, color=EXP)
    ax_he.fill_between(t, hdg_err, 0, where=(hdg_err<0),
                       alpha=0.15, color=PF)
    ax_he.plot(t, hdg_err, "-", color=EXP, lw=2.0,
               label="Heading error  (desired − actual)")
    ax_he.axhline(0,   color="black",   lw=0.8, ls=":")
    ax_he.axhline( 30, color="#cc8800", lw=1.0, ls="--", alpha=0.8,
                   label="±30° reference")
    ax_he.axhline(-30, color="#cc8800", lw=1.0, ls="--", alpha=0.8)
    fix_ylim(ax_he, hdg_err, [-35, 35], step=30.0)
    style_ax(ax_he, "Time  (s)", "Heading error  (deg)",
             "Heading error to waypoint vs time\n(+ = target CCW from robot, − = CW)")
    ax_he.xaxis.set_major_locator(MaxNLocator(7))
    add_wp_lines(ax_he, t, wp_idx)
    ax_he.legend(fontsize=8, loc="best", framealpha=0.85)

    # panel 6: Steering + Speed vs time 
    ax_st = fig.add_subplot(gs[2, 2])
    ax_sp = ax_st.twinx()
    ln1, = ax_st.plot(t, steer, "-",  color=STR, lw=2.2,
                      label="Steering offset  (servo units)")
    ln2, = ax_sp.plot(t, speed, "--", color=SPD, lw=1.8,
                      label="Speed command  (PWM)")
    ax_st.axhline(0, color="black", lw=0.7, ls=":", alpha=0.6)
    sa = max(abs(steer.min()), abs(steer.max())) + 2
    ax_st.set_ylim(-sa, sa)
    ax_st.yaxis.set_major_locator(MultipleLocator(5))
    ax_st.yaxis.set_minor_locator(MultipleLocator(1))
    ax_sp.set_ylim(max(0, speed.min()-10), speed.max()+10)
    ax_sp.yaxis.set_major_locator(MultipleLocator(20))
    ax_sp.yaxis.set_minor_locator(MultipleLocator(5))
    ax_st.set_xlabel("Time  (s)", fontsize=9, labelpad=4)
    ax_st.set_ylabel("Steering offset\n(0 = straight)", fontsize=9,
                     color=STR, labelpad=4)
    ax_sp.set_ylabel("Speed command  (PWM)", fontsize=9,
                     color=SPD, labelpad=4)
    ax_st.set_title("Steering command & speed vs time",
                    fontsize=10, fontweight="bold", pad=6)
    ax_st.tick_params(axis="y", labelcolor=STR, labelsize=8,
                      which="both", direction="in")
    ax_sp.tick_params(axis="y", labelcolor=SPD, labelsize=8,
                      which="both", direction="in")
    ax_st.tick_params(axis="x", labelsize=8, direction="in")
    ax_st.xaxis.set_major_locator(MaxNLocator(7))
    ax_st.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_st.grid(True, which="major", lw=0.5, alpha=0.45, color="#cccccc")
    ax_st.grid(True, which="minor", lw=0.25, alpha=0.3, color="#e8e8e8")
    ax_st.spines["top"].set_visible(False)
    ax_st.legend([ln1, ln2], [l.get_label() for l in [ln1, ln2]],
                 fontsize=8, loc="best", framealpha=0.85)
    add_wp_lines(ax_st, t, wp_idx)

    # panel 7: Control function breakdown (full-width) 
    ax_cf = fig.add_subplot(gs[3, :])

    # Shade regions where clamping was active (output saturated)
    ax_cf.fill_between(t, STEER_LO, STEER_HI,
                       alpha=0.06, color="green",
                       label=f"Clamp range  [{STEER_LO:.0f}, +{STEER_HI:.0f}] servo units")
    ax_cf.fill_between(t, unclamped, clamped,
                       where=is_clamped, alpha=0.35, color=CLPZ,
                       label="Clamped region  (saturation active)")

    ax_cf.plot(t, unclamped, "-",  color=UNCLP, lw=1.8, alpha=0.8,
               label=f"Unclamped output  =  {STEERING_GAIN} × heading_err_rad")
    ax_cf.plot(t, clamped,  "-",  color=CLPD,  lw=2.5,
               label="Clamped output  (actual steer_rel sent)")
    ax_cf.plot(t, steer,    "--", color=STR,   lw=1.4, alpha=0.7,
               label="Logged steer_rel  (should match clamped)")

    ax_cf.axhline(0,        color="black",   lw=0.9, ls=":",  alpha=0.7,
                  label="0 = straight ahead")
    ax_cf.axhline(STEER_HI, color="#cc4400", lw=1.0, ls="--", alpha=0.7,
                  label=f"+{STEER_HI:.0f} = max right  (STEER_MAX)")
    ax_cf.axhline(STEER_LO, color="#0044cc", lw=1.0, ls="--", alpha=0.7,
                  label=f"{STEER_LO:.0f} = max left   (STEER_MIN)")

    # Y range — show both unclamped and clamped
    all_vals = np.concatenate([unclamped, clamped, [STEER_LO-2, STEER_HI+2]])
    lo = np.nanmin(all_vals); hi = np.nanmax(all_vals)
    pad = (hi - lo) * 0.12
    ax_cf.set_ylim(lo - pad, hi + pad)
    ax_cf.yaxis.set_major_locator(MultipleLocator(5))
    ax_cf.yaxis.set_minor_locator(MultipleLocator(1))

    style_ax(ax_cf,
             "Time  (s)",
             "Control output  (servo units,  + = right,  − = left)",
             f"Control function over time\n"
             f"steer_rel = clamp(  {STEERING_GAIN} × heading_error_rad,  "
             f"{STEER_LO:.0f},  +{STEER_HI:.0f}  )")
    ax_cf.xaxis.set_major_locator(MaxNLocator(12))
    add_wp_lines(ax_cf, t, wp_idx)
    ax_cf.legend(fontsize=8.5, loc="upper right", framealpha=0.92, ncol=2)

    # stats footer 
    dur         = t[-1] - t[0]
    n_wps       = int(wp_idx[-1]) + 1
    rmse_y      = float(np.sqrt(np.mean((pf_y - ex_y)**2)))
    max_he      = float(np.max(np.abs(hdg_err)))
    clamp_pct   = 100.0 * float(np.mean(is_clamped))
    mean_s      = float(np.mean(np.abs(steer)))
    stats = (
        f"Duration: {dur:.1f} s     "
        f"Waypoints reached: {n_wps} / {len(WPS)}     "
        f"RMSE y: {rmse_y:.3f} m     "
        f"Max |heading error|: {max_he:.1f}°     "
        f"Steering saturated: {clamp_pct:.1f}% of time     "
        f"Mean |steer|: {mean_s:.1f} units"
    )
    fig.text(0.5, 0.003, stats, ha="center", va="bottom", fontsize=9,
             color="#333333",
             bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc"))

    # save and open the plot
    out = title + ".png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    if platform.system() == "Darwin":
        subprocess.run(["open", out])
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", out])
    else:
        subprocess.run(["start", out], shell=True)


# entry point
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else find_latest()
    print(f"Loading: {path}")
    rows = load_log(path)
    print(f"  {len(rows)} rows   duration = {rows[-1]['t']:.1f} s")
    title = os.path.splitext(os.path.basename(path))[0]
    plot_all(rows, title)