
# analyze_0307_thesis_frequency.py
# ============================================================
# Thesis-oriented analyzer for memdiode fitdeck template (NGSpice)
#
# Main upgrades for Chapter 4.4:
#   1) TRUE frequency sweep via TIME_SCALE_LIST
#      - frequency_Hz = 1 / tstop
#      - default: 0.4, 0.2, 0.1, 0.05, 0.025 Hz
#   2) Internal-state plots for thesis discussion
#      - x_vs_time.png
#      - overlay_*_x_vs_time.png
#      - overlay_*_x_vs_phase.png
#   3) Quantitative state-recovery metrics
#      - x_start, x_end, x_recovery_signed, x_recovery_abs
#      - x_min, x_max, x_span
#   4) Quantitative hysteresis metrics
#      - loop_area_signed = \oint I dV
#      - loop_area_abs_path = \sum |I_mid * dV|
#   5) Frequency-summary outputs for direct thesis use
#      - summary_all_cases.csv
#      - summary_baseline_frequency.csv
#      - plots/metrics_vs_frequency/*.png
#   6) Overlay I-V now uses Vp (device terminal voltage) consistently
#
# Outputs:
#   analyze_result_sweep/
#     fitdeck_embedded.cir
#     cases/<case_tag>/{decks,logs,sims,plots}/...
#     overlay/*.png
#     metrics_vs_frequency/*.png
#     summary_all_cases.csv
#     summary_baseline_frequency.csv
#
# Recommended use for Chapter 4.4:
#   - Keep ENABLE_BASELINE = True
#   - Keep ENABLE_RATE_SWEEP = True
#   - Use TIME_SCALE_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]
#   - Keep RUN_NOLIMIT_MODE = True
#   - Keep RUN_LIMIT_MODE = False (unless you want compliance discussion)
# ============================================================

import re
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
CSV_MEAS = BASE_DIR / "data" / "DC-IV.csv"
NGSPICE = Path(shutil.which("ngspice") or (BASE_DIR / "ngspice.exe"))
TEMPLATE_SRC = BASE_DIR / "results" / "fit" / "fitdeck_embedded.cir"
THETA_PATH = BASE_DIR / "results" / "fit" / "theta_best.csv"

OUT_DIR = BASE_DIR / "results" / "analysis_frequency"
TEMPLATE_SNAPSHOT = OUT_DIR / "fitdeck_embedded.cir"
CASES_DIR = OUT_DIR / "cases"
OVERLAY_DIR = OUT_DIR / "overlay"
METRIC_PLOT_DIR = OUT_DIR / "metrics_vs_frequency"

# =========================
# Timing / numerical
# =========================
TSTOP_BASE = 10.0
DTMAX_SIM = 2e-4

PRINT_DIV = 6
PRINT_MIN = 2e-5
PRINT_MAX = 4e-3

# =========================
# Fixed model knobs
# =========================
KSW_FIXED = 3
RH0_FIXED = 1e3
RH_MIN_FIXED = 1.0
RH_MAX_FIXED = 1e7
VSLOPE_FIXED = 0.5

# =========================
# Compliance knobs base
# =========================
RLO_FIXED = 1.0
RHI_FIXED = 2e8
VCOMP_FIXED = 0.0
VSLOPE_POS_FIX = 0.02
ISLOPE_REL = 0.02

# =========================
# Plot settings
# =========================
I_FLOOR_ABS = 3e-10
SYMLINTHRESH = 1e-9
MAX_PLOT_POINTS = 24000

# =========================
# Theta params required
# =========================
REQUIRED_PARAMS = {
    "IMAX", "IMIN", "ALPHA_MAX", "ALPHA_MIN", "VSET", "VRES", "ETA_SET", "ETA_RES",
    "CH0", "ISCALE", "H0", "EI", "ROFF", "BETAA"
}
ALIASES = {"BETA": "BETAA"}

# ============================================================
# Sweep configuration (EDIT HERE)
# ============================================================

# ----- Enable/disable major sweep groups -----
ENABLE_BASELINE = True
ENABLE_RATE_SWEEP = True
ENABLE_AMPLITUDE_SWEEP = False
ENABLE_MULTI_NEG_SWEEP = False
ENABLE_REPEAT_NEG_SWEEP = False
ENABLE_COMPLIANCE_SWEEP = False

# Optional: staircase / discretized-input comparison
# For Chapter 4.4 frequency discussion, keep this False by default.
ENABLE_DISCRETE_LEVEL_SWEEP = False

# ----- Modes -----
RUN_LIMIT_MODE = False
RUN_NOLIMIT_MODE = True

# ----- Shared fixed baseline waveform -----
FIXED_BASELINE_VPOS = 8.0
FIXED_BASELINE_VNEG = -6.0

# ----- Rate / time / continuous linear-ramp sweeps -----
# frequency_Hz = 1 / (TSTOP_BASE * time_scale)
# With TSTOP_BASE = 10 s:
#   0.25 -> 0.4 Hz
#   0.5  -> 0.2 Hz
#   1.0  -> 0.1 Hz
#   2.0  -> 0.05 Hz
#   4.0  -> 0.025 Hz
TIME_SCALE_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]

# Continuous linear ramp sampling
PTS_PER_RAMP_LIST = [160]
HOLD_PTS_LIST = [0]

# ----- Optional: discretized-input / staircase sweeps -----
LEVELS_PER_RAMP_LIST = [3, 17, 2560]
HOLD_PTS_PER_LEVEL = 4  # set >1 if you want true visible staircase holds

# ----- Amplitude sweeps -----
VPOS_LIST = [10.0]
VNEG_LIST = [-2.0, -3.0, -4.0, -5.0]
FIXED_VNEG_FOR_VPOS_SWEEP = -5.0
FIXED_VPOS_FOR_VNEG_SWEEP = 10.0
ENABLE_FULL_VPOSxVNEG_GRID = False

# ----- Multi negative sequences -----
MULTI_NEG_SEQ_LIST = [
    [-2.0, -3.0, -5.0],
    [-5.0, -3.0, -2.0],
]

# ----- Repeat negative cycles -----
REPEAT_NEG_LIST = [-5.0]
REPEAT_COUNT_LIST = [1, 3, 5]
REPEAT_VPOS_LIST = [10.0, 12.0]

# ----- Compliance sweeps (LIMIT only) -----
ICOMP_SCALE_LIST = [0.5, 1.0, 2.0]
VCOMP_LIST = [0.0]
RHI_LIST = [2e8]

# ----- Baseline frequency-summary selection -----
# These settings define which cases are gathered into baseline frequency plots/tables.
BASELINE_SUMMARY_CASE_PREFIXES = ("fixed_vpos", "measV")
BASELINE_SUMMARY_INPUT_STYLES = ("linear",)

# ============================================================
# Utilities
# ============================================================

def ensure_dirs():
    OUT_DIR.mkdir(exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE_SRC.exists():
        raise FileNotFoundError(f"Missing template: {TEMPLATE_SRC}")

    TEMPLATE_SNAPSHOT.write_text(
        TEMPLATE_SRC.read_text(encoding="utf-8", errors="ignore"),
        encoding="utf-8",
    )


def safe_tag(s: str):
    s = s.replace(" ", "_")
    s = s.replace("/", "_").replace("\\", "_")
    s = s.replace(":", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def make_case_dirs(case_tag: str):
    case_dir = CASES_DIR / case_tag
    deck_dir = case_dir / "decks"
    log_dir = case_dir / "logs"
    sim_dir = case_dir / "sims"
    plot_dir = case_dir / "plots"
    for d in (deck_dir, log_dir, sim_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)
    return case_dir, deck_dir, log_dir, sim_dir, plot_dir


def read_meas_csv(path: Path):
    df = pd.read_csv(path, engine="python")
    if df.shape[1] < 2:
        raise ValueError("DC-IV.csv must have at least 2 columns: V, I")
    v_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    i_raw = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    m = (~v_raw.isna()) & (~i_raw.isna())
    v = v_raw[m].to_numpy(dtype=float)
    i = i_raw[m].to_numpy(dtype=float)
    if len(v) < 10:
        raise ValueError("Too few numeric rows after cleaning. Check DC-IV.csv format.")
    return v, i


def read_theta_best(path: Path):
    df = pd.read_csv(path)
    if not {"param", "value"}.issubset(df.columns):
        raise ValueError("theta_best.csv must have columns: param,value")

    theta = {}
    for p, v in zip(df["param"], df["value"]):
        k = str(p).strip()
        if k in ALIASES:
            k = ALIASES[k]
        theta[k] = float(v)

    missing = sorted([k for k in REQUIRED_PARAMS if k not in theta])
    if missing:
        raise ValueError(
            f"theta_best.csv missing required params: {missing}\n"
            f"Loaded keys: {sorted(theta.keys())}"
        )
    return theta


def estimate_icomp_pos(V: np.ndarray, I: np.ndarray):
    m = V > 0.5
    if not np.any(m):
        return 1e-3
    x = np.abs(I[m])
    ic = float(np.quantile(x, 0.98))
    return float(np.clip(ic, 1e-6, 5e-2))


def make_time_vector(N: int, TSTOP: float):
    return np.linspace(0.0, TSTOP, N) if N >= 2 else np.array([0.0])


def pick_tstep_print(N_full: int, TSTOP: float):
    if N_full <= 1:
        return 1e-3
    dt_meas = TSTOP / (N_full - 1)
    tstep = dt_meas / PRINT_DIV
    return float(max(PRINT_MIN, min(PRINT_MAX, tstep)))


def pwl_inline_from_tv(t: np.ndarray, v: np.ndarray, pairs_per_line=8):
    items = [f"{ti:.12g} {vi:.12g}" for ti, vi in zip(t, v)]
    lines = []
    for i in range(0, len(items), pairs_per_line):
        lines.append("+ " + " ".join(items[i: i + pairs_per_line]))
    return "\n".join(lines)


def run_ngspice(deck_path: Path, log_path: Path, cwd: Path, timeout_s=180):
    cmd = [str(NGSPICE), "-b", "-o", str(log_path), str(deck_path)]
    try:
        r = subprocess.run(cmd, cwd=str(cwd), timeout=timeout_s)
        return r.returncode
    except subprocess.TimeoutExpired:
        return 124


def load_wrdata(path: Path):
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise ValueError(f"wrdata has {data.shape[1]} cols (<6): {path}")
    t = data[:, 0]
    vcmd = data[:, 1]
    vp = data[:, 2]
    idev = data[:, 3]
    vx = data[:, 4]
    vxh = data[:, 5]
    return t, vcmd, vp, idev, vx, vxh


def tail_text(path: Path, n_lines=120):
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""


def _downsample_df(df: pd.DataFrame, cols, max_points=MAX_PLOT_POINTS):
    n = len(df)
    if n <= max_points:
        return [df[c].to_numpy(float) for c in cols]
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return [df[c].to_numpy(float)[idx] for c in cols]


def save_case_text_summary(plot_dir: Path, summary: dict):
    lines = []
    for k, v in summary.items():
        lines.append(f"{k},{v}")
    (plot_dir / "case_summary_metrics.csv").write_text("\n".join(lines), encoding="utf-8")


def calc_cmd_metrics(V_cmd: np.ndarray, tstop: float, num_segments: int):
    V_cmd = np.asarray(V_cmd, dtype=float)
    N = len(V_cmd)
    dt_cmd = float(tstop / max(N - 1, 1))
    dV = np.diff(V_cmd) if N >= 2 else np.array([0.0])

    return {
        "N_cmd": int(N),
        "dt_cmd_s": dt_cmd,
        "dV_cmd_mean_abs_V": float(np.mean(np.abs(dV))) if len(dV) else 0.0,
        "dV_cmd_max_abs_V": float(np.max(np.abs(dV))) if len(dV) else 0.0,
        "period_s": float(tstop),
        "frequency_Hz": float(1.0 / tstop) if tstop > 0 else np.nan,
        "num_segments": int(num_segments),
        "ramp_time_s": float(tstop / max(num_segments, 1)),
    }


def calc_sweep_rates(path_nodes: list, tstop: float):
    path_nodes = [float(v) for v in path_nodes]
    num_segments = max(len(path_nodes) - 1, 1)
    seg_time = float(tstop / num_segments)
    seg_rates = []
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        seg_rates.append((float(b) - float(a)) / seg_time)

    return {
        "sweep_rate_mean_abs_V_per_s": float(np.mean(np.abs(seg_rates))) if seg_rates else np.nan,
        "sweep_rate_max_abs_V_per_s": float(np.max(np.abs(seg_rates))) if seg_rates else np.nan,
        "sweep_rate_first_segment_V_per_s": float(seg_rates[0]) if seg_rates else np.nan,
    }


def calc_loop_metrics(V: np.ndarray, I: np.ndarray):
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    if len(V) < 2:
        return {
            "loop_area_signed_A_V": np.nan,
            "loop_area_abs_path_A_V": np.nan,
        }

    dV = np.diff(V)
    I_mid = 0.5 * (I[:-1] + I[1:])

    loop_area_signed = float(np.sum(I_mid * dV))
    loop_area_abs_path = float(np.sum(np.abs(I_mid * dV)))

    return {
        "loop_area_signed_A_V": loop_area_signed,
        "loop_area_abs_path_A_V": loop_area_abs_path,
    }


def calc_zero_crossing_times(V: np.ndarray, t: np.ndarray):
    V = np.asarray(V, dtype=float)
    t = np.asarray(t, dtype=float)
    times = []
    for i in range(len(V) - 1):
        v0, v1 = V[i], V[i + 1]
        if v0 == 0:
            times.append(float(t[i]))
        elif v0 * v1 < 0:
            # linear interpolation
            frac = abs(v0) / max(abs(v1 - v0), 1e-30)
            times.append(float(t[i] + frac * (t[i + 1] - t[i])))
    return times


def compute_return_metrics(df: pd.DataFrame):
    t = df["time"].to_numpy(float)
    V = df["Vp"].to_numpy(float)
    x = df["x"].to_numpy(float)
    xh = df["xh"].to_numpy(float)

    x_start = float(x[0])
    x_end = float(x[-1])
    xh_start = float(xh[0])
    xh_end = float(xh[-1])

    zero_times = calc_zero_crossing_times(V, t)

    metrics = {
        "x_start": x_start,
        "x_end": x_end,
        "x_recovery_signed": x_end - x_start,
        "x_recovery_abs": abs(x_end - x_start),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "x_span": float(np.max(x) - np.min(x)),
        "xh_start": xh_start,
        "xh_end": xh_end,
        "xh_recovery_signed": xh_end - xh_start,
        "xh_recovery_abs": abs(xh_end - xh_start),
        "xh_min": float(np.min(xh)),
        "xh_max": float(np.max(xh)),
        "xh_span": float(np.max(xh) - np.min(xh)),
        "x_end_fraction_of_span": float(abs(x_end - x_start) / max(np.max(x) - np.min(x), 1e-30)),
        "xh_end_fraction_of_span": float(abs(xh_end - xh_start) / max(np.max(xh) - np.min(xh), 1e-30)),
    }

    # best-effort: x at first return to approximately 0 V after + branch and after - branch
    if zero_times:
        metrics["num_zero_crossings_vp"] = int(len(zero_times))
    else:
        metrics["num_zero_crossings_vp"] = 0

    return metrics


def interpolate_y_at_x(x, y, x_target):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan

    candidates = []
    for i in range(len(x) - 1):
        x0, x1 = x[i], x[i + 1]
        if x0 == x_target:
            candidates.append(float(y[i]))
        elif (x0 - x_target) * (x1 - x_target) < 0:
            frac = (x_target - x0) / (x1 - x0)
            candidates.append(float(y[i] + frac * (y[i + 1] - y[i])))

    if not candidates:
        # nearest-point fallback
        j = int(np.argmin(np.abs(x - x_target)))
        return float(y[j])

    return float(np.mean(candidates))


def compute_branch_metrics(df: pd.DataFrame, targets=(0.0, 2.0, 5.0, 8.0, -2.0, -5.0, -8.0)):
    """
    Approximate branch separation by splitting the trajectory into monotonic Vp segments.
    For the baseline 0 -> +V -> 0 -> -V -> 0 waveform, this gives four main branches.
    """
    V = df["Vp"].to_numpy(float)
    I = df["I"].to_numpy(float)

    if len(V) < 5:
        return {}

    dV = np.diff(V)
    sign = np.sign(dV)
    # clean zeros by forward fill then backward fill
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    for i in range(len(sign) - 2, -1, -1):
        if sign[i] == 0:
            sign[i] = sign[i + 1]
    if len(sign) == 0:
        return {}

    segments = []
    start = 0
    cur = sign[0]
    for i in range(1, len(sign)):
        if sign[i] != cur:
            segments.append((start, i))
            start = i
            cur = sign[i]
    segments.append((start, len(V) - 1))

    # need at least 4 main segments for baseline loop
    if len(segments) < 2:
        return {}

    out = {}
    # compare first positive-going segment vs following negative-going segment at positive V
    for tgt in targets:
        vals = []
        for a, b in segments[:4]:
            if b <= a:
                continue
            vv = V[a:b + 1]
            ii = I[a:b + 1]
            vals.append(interpolate_y_at_x(vv, ii, tgt))
        if len(vals) >= 2 and np.sum(np.isfinite(vals[:2])) == 2:
            out[f"branch_sep_I_at_V{str(tgt).replace('.', 'p').replace('-', 'm')}"] = float(abs(vals[0] - vals[1]))
    return out


# ============================================================
# Waveform builders
# ============================================================

def _ramp(v0: float, v1: float, n: int):
    n = int(max(2, n))
    return np.linspace(v0, v1, n)


def make_voltage_sequence(nodes, pts_per_ramp: int = 220, hold_pts: int = 0):
    """
    Continuous linear-ramp waveform.
    Each ramp segment is linearly interpolated.
    hold_pts repeats every sampled point.
    """
    nodes = [float(v) for v in nodes]
    if len(nodes) < 2:
        raise ValueError("nodes must have at least 2 points")

    repeat_n = int(max(1, hold_pts + 1))
    segs = []
    first_ramp = True

    for a, b in zip(nodes[:-1], nodes[1:]):
        ramp = _ramp(a, b, pts_per_ramp)
        if not first_ramp:
            ramp = ramp[1:]
        first_ramp = False
        ramp = np.repeat(ramp, repeat_n)
        segs.append(ramp)

    return np.concatenate(segs)


def make_staircase_sequence(nodes, levels_per_ramp: int = 3, hold_pts_per_level: int = 8):
    """
    Staircase-like waveform:
    each ramp is quantized into a finite number of voltage levels,
    and each level is held for several samples.
    """
    nodes = [float(v) for v in nodes]
    if len(nodes) < 2:
        raise ValueError("nodes must have at least 2 points")

    levels_per_ramp = int(max(2, levels_per_ramp))
    hold_pts_per_level = int(max(1, hold_pts_per_level))

    segs = []
    first_seg = True

    for a, b in zip(nodes[:-1], nodes[1:]):
        levels = np.linspace(a, b, levels_per_ramp)

        if not first_seg:
            levels = levels[1:]
        first_seg = False

        seg = np.repeat(levels, hold_pts_per_level)
        segs.append(seg)

    return np.concatenate(segs)


# ============================================================
# Simulation
# ============================================================

def simulate_from_fitdeck(
    theta: dict,
    V_cmd: np.ndarray,
    case_tag: str,
    tstop: float,
    icomp_pos: float,
    vcomp: float,
    rlo: float,
    rhi: float,
    vslope_pos: float,
    islope_rel: float,
):
    case_dir, deck_dir, log_dir, sim_dir, plot_dir = make_case_dirs(case_tag)

    N = len(V_cmd)
    t = make_time_vector(N, tstop)
    pwl_inline = pwl_inline_from_tv(t, V_cmd, pairs_per_line=8)
    tstep_print = pick_tstep_print(N, tstop)

    deck_path = deck_dir / f"{case_tag}.cir"
    log_path = log_dir / f"{case_tag}.log"
    sim_path = sim_dir / f"{case_tag}.dat"

    txt = TEMPLATE_SNAPSHOT.read_text(encoding="utf-8", errors="ignore")
    islope = max(1e-12, float(islope_rel) * float(icomp_pos))

    repl = {
        "@PWL_INLINE@": pwl_inline,
        "@SIMOUT@": str(sim_path),
        "@TSTEP@": f"{tstep_print:.12g}",
        "@DTMAX@": f"{DTMAX_SIM:.12g}",
        "@TSTOP@": f"{tstop:.12g}",
        "@KSW@": f"{KSW_FIXED:.12g}",
        "@RH0@": f"{RH0_FIXED:.12g}",
        "@RH_MIN@": f"{RH_MIN_FIXED:.12g}",
        "@RH_MAX@": f"{RH_MAX_FIXED:.12g}",
        "@VSLOPE@": f"{VSLOPE_FIXED:.12g}",
        "@ICOMP_POS@": f"{float(icomp_pos):.12g}",
        "@VCOMP@": f"{float(vcomp):.12g}",
        "@RLO@": f"{float(rlo):.12g}",
        "@RHI@": f"{float(rhi):.12g}",
        "@ISLOPE@": f"{float(islope):.12g}",
        "@VSLOPE_POS@": f"{float(vslope_pos):.12g}",
    }
    for k, v in theta.items():
        repl[f"@{k}@"] = f"{float(v):.12g}"

    for k, v in repl.items():
        txt = txt.replace(k, str(v))

    leftovers = re.findall(r"@[A-Za-z0-9_]+@", txt)
    if leftovers:
        raise RuntimeError(f"Unreplaced placeholders in deck: {sorted(set(leftovers))[:30]}")

    deck_path.write_text(txt, encoding="utf-8")

    rc = run_ngspice(deck_path, log_path, cwd=case_dir, timeout_s=180)
    if rc != 0 or (not sim_path.exists()):
        print(f"[SIM] ngspice failed: {case_tag}, rc={rc}")
        print("[log tail]\n" + tail_text(log_path))
        return None

    t, vcmd, vp, idev, vx, vxh = load_wrdata(sim_path)
    df = pd.DataFrame({"time": t, "Vcmd": vcmd, "Vp": vp, "I": idev, "x": vx, "xh": vxh})
    df["phase"] = df["time"] / max(float(df["time"].iloc[-1]), 1e-30)
    df.to_csv(sim_dir / f"{case_tag}_sim.csv", index=False)
    return df, plot_dir


# ============================================================
# Thesis-friendly label utilities
# ============================================================

def _fmt_num_clean(x, nd=4):
    """Format numbers for plot legends without unnecessary trailing zeros."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return (f"{x:.{nd}g}").rstrip("0").rstrip(".")


def _case_tag_to_label(tag: str):
    """
    Convert long simulation case tags into clean thesis-style legend labels.

    Examples:
      fixed_vpos10_vneg10_return_ppr160_hold0_ts0.25 -> f = 0.4 Hz, T = 2.5 s
      amp_vpos10_vneg-4_return_ppr160_hold0_ts1       -> Vpos = 10 V, Vneg = -4 V
      stairs_vpos10_vneg-10_return_lpr17_hpl4_ts1     -> 17 levels/ramp
    """
    raw = str(tag)

    # Time-scale cases used in Chapter 4.4 frequency overlays.
    m_ts = re.search(r"(?:^|_)ts([0-9]+(?:p[0-9]+)?(?:\.[0-9]+)?)", raw)
    if m_ts:
        ts = float(m_ts.group(1).replace("p", "."))
        tstop = TSTOP_BASE * ts
        freq = 1.0 / tstop if tstop > 0 else np.nan
        return f"f = {_fmt_num_clean(freq)} Hz, T = {_fmt_num_clean(tstop)} s"

    if raw.startswith("measV"):
        return "measured waveform"

    # Staircase/discretized-input cases.
    m_lpr = re.search(r"lpr([0-9]+)", raw)
    if m_lpr:
        return f"{m_lpr.group(1)} levels/ramp"

    # Amplitude sweep cases.
    m_amp = re.search(r"vpos(-?[0-9]+(?:\.[0-9]+)?)_vneg(-?[0-9]+(?:\.[0-9]+)?)", raw)
    if m_amp and (raw.startswith("amp_") or raw.startswith("grid_")):
        return f"Vpos = {m_amp.group(1)} V, Vneg = {m_amp.group(2)} V"

    # Repeat-negative cases.
    m_rep = re.search(r"repeat_vpos(-?[0-9]+(?:\.[0-9]+)?)_vneg(-?[0-9]+(?:\.[0-9]+)?)_x([0-9]+)", raw)
    if m_rep:
        return f"Vpos = {m_rep.group(1)} V, Vneg = {m_rep.group(2)} V, repeat = {m_rep.group(3)}"

    # Multi-negative sequence cases.
    m_multi = re.search(r"multi_vpos(-?[0-9]+(?:\.[0-9]+)?)_seq_(.+)", raw)
    if m_multi:
        seq = m_multi.group(2).replace("_", ", ")
        return f"Vpos = {m_multi.group(1)} V, Vneg seq. = {seq} V"

    # Fallback: remove common technical fragments.
    label = raw
    label = re.sub(r"_?NOLIMIT", "", label)
    label = re.sub(r"_?LIMIT", "", label)
    label = re.sub(r"_?ppr[0-9]+", "", label)
    label = re.sub(r"_?hold[0-9]+", "", label)
    label = re.sub(r"_?hpl[0-9]+", "", label)
    label = label.replace("fixed_", "")
    label = label.replace("_return", "")
    label = label.replace("_", " ")
    return label.strip()

# ============================================================
# Plots
# ============================================================

def plot_vcmd_vs_time(df: pd.DataFrame, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    t, v = _downsample_df(df, ["time", "Vcmd"], max_points=22000)

    plt.figure()
    plt.plot(t, v, linewidth=1.2)
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel("time (s)")
    plt.ylabel("Vcmd (V)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_x_vs_time(df: pd.DataFrame, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    t, x = _downsample_df(df, ["time", "x"], max_points=22000)

    plt.figure()
    plt.plot(t, x, linewidth=1.2)
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel("time (s)")
    plt.ylabel("x (state)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_xh_vs_time(df: pd.DataFrame, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    t, xh = _downsample_df(df, ["time", "xh"], max_points=22000)

    plt.figure()
    plt.plot(t, xh, linewidth=1.2)
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel("time (s)")
    plt.ylabel("xh")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_x_vs_phase(df: pd.DataFrame, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    ph, x = _downsample_df(df, ["phase", "x"], max_points=22000)

    plt.figure()
    plt.plot(ph, x, linewidth=1.2)
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel("normalized phase (0~1)")
    plt.ylabel("x (state)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_iv_symlog(df: pd.DataFrame, out_png: Path, title: str, use_vp=True):
    import matplotlib.pyplot as plt
    vcol = "Vp" if use_vp else "Vcmd"
    V, I = _downsample_df(df, [vcol, "I"], max_points=22000)

    plt.figure()
    plt.yscale("symlog", linthresh=SYMLINTHRESH)
    plt.plot(V, I, ".", markersize=2)
    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.xlabel(f"{vcol} (V)")
    plt.ylabel("I (A)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_logabsI_vs_V(df: pd.DataFrame, out_png: Path, title: str, use_vp=True):
    import matplotlib.pyplot as plt
    vcol = "Vp" if use_vp else "Vcmd"
    V, I = _downsample_df(df, [vcol, "I"], max_points=22000)

    plt.figure()
    plt.semilogy(V, np.abs(I) + I_FLOOR_ABS, ".", markersize=2)
    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.xlabel(f"{vcol} (V)")
    plt.ylabel("|I| (A)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_x_vs_V(df: pd.DataFrame, out_png: Path, title: str, use_vp=True, connect=True):
    import matplotlib.pyplot as plt
    vcol = "Vp" if use_vp else "Vcmd"
    V, x = _downsample_df(df, [vcol, "x"], max_points=20000)

    plt.figure()
    if connect:
        plt.plot(V, x, linewidth=1.0)
        plt.scatter(V, x, s=8, alpha=0.5)
    else:
        plt.scatter(V, x, s=10, alpha=0.8)
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel(f"{vcol} (V)")
    plt.ylabel("x (state)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_x_vs_I(df: pd.DataFrame, out_png: Path, title: str, log_absI=True, connect=True):
    import matplotlib.pyplot as plt
    I, x = _downsample_df(df, ["I", "x"], max_points=20000)
    if log_absI:
        X = np.log10(np.abs(I) + I_FLOOR_ABS)
        xlabel = "log10(|I|) (A)"
    else:
        X = I
        xlabel = "I (A)"

    plt.figure()
    if connect:
        plt.plot(X, x, linewidth=1.0)
        plt.scatter(X, x, s=8, alpha=0.5)
    else:
        plt.scatter(X, x, s=10, alpha=0.8)
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel(xlabel)
    plt.ylabel("x (state)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_IV_colored_by_x(df: pd.DataFrame, out_png: Path, title: str, use_vp=True, symlog_y=True):
    import matplotlib.pyplot as plt
    vcol = "Vp" if use_vp else "Vcmd"
    V, I, x = _downsample_df(df, [vcol, "I", "x"], max_points=24000)

    plt.figure()
    if symlog_y:
        plt.yscale("symlog", linthresh=SYMLINTHRESH)
    sc = plt.scatter(V, I, c=x, s=10)
    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.xlabel(f"{vcol} (V)")
    plt.ylabel("I (A)")
    plt.title(title)
    cbar = plt.colorbar(sc)
    cbar.set_label("x (state)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_butterfly_with_x(df: pd.DataFrame, out_png: Path, title: str, use_vp=True):
    import matplotlib.pyplot as plt
    vcol = "Vp" if use_vp else "Vcmd"
    V, I, x = _downsample_df(df, [vcol, "I", "x"], max_points=24000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax1.set_yscale("symlog", linthresh=SYMLINTHRESH)
    sc1 = ax1.scatter(V, I, c=x, s=10)
    ax1.grid(True, which="both", ls="--", alpha=0.35)
    ax1.set_xlabel(f"{vcol} (V)")
    ax1.set_ylabel("I (A)")
    ax1.set_title("I–V (symlog), colored by x")
    cbar1 = fig.colorbar(sc1, ax=ax1)
    cbar1.set_label("x (state)")

    sc2 = ax2.scatter(V, x, c=x, s=10)
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.set_xlabel(f"{vcol} (V)")
    ax2.set_ylabel("x (state)")
    ax2.set_title("x–V, colored by x")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label("x (state)")

    fig.suptitle(title, y=1.02)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def make_all_case_plots(df: pd.DataFrame, plot_dir: Path, tag: str):
    plot_vcmd_vs_time(df, plot_dir / "Vcmd_vs_time.png", f"Input voltage vs time - {tag}")
    plot_x_vs_time(df, plot_dir / "x_vs_time.png", f"x vs time - {tag}")
    plot_xh_vs_time(df, plot_dir / "xh_vs_time.png", f"xh vs time - {tag}")
    plot_x_vs_phase(df, plot_dir / "x_vs_phase.png", f"x vs normalized phase - {tag}")
    plot_iv_symlog(df, plot_dir / "IV_symlog.png", f"I–V (symlog) - {tag}", use_vp=True)
    plot_logabsI_vs_V(df, plot_dir / "logabsI_vs_Vp.png", f"log(|I|) vs Vp - {tag}", use_vp=True)
    plot_x_vs_V(df, plot_dir / "x_vs_Vp.png", f"x vs Vp - {tag}", use_vp=True, connect=True)
    plot_x_vs_I(df, plot_dir / "x_vs_logabsI.png", f"x vs log|I| - {tag}", log_absI=True, connect=True)
    plot_IV_colored_by_x(df, plot_dir / "IV_colored_by_x.png", f"I–V colored by x - {tag}", use_vp=True, symlog_y=True)
    plot_butterfly_with_x(df, plot_dir / "butterfly_with_x.png", f"Butterfly & x relationship - {tag}", use_vp=True)


def plot_overlay_abs(sim_dict: dict, out_png: Path, title: str, use_vp=True):
    import matplotlib.pyplot as plt
    plt.figure()
    vcol = "Vp" if use_vp else "Vcmd"
    for k, df in sim_dict.items():
        V = df[vcol].to_numpy(float)
        I = df["I"].to_numpy(float)
        plt.semilogy(V, np.abs(I) + I_FLOOR_ABS, ".", label=_case_tag_to_label(k))
    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.xlabel(f"{vcol} (V)")
    plt.ylabel("|I| (A)")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_overlay_symlog(sim_dict: dict, out_png: Path, title: str, use_vp=True):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.yscale("symlog", linthresh=SYMLINTHRESH)
    vcol = "Vp" if use_vp else "Vcmd"
    for k, df in sim_dict.items():
        V = df[vcol].to_numpy(float)
        I = df["I"].to_numpy(float)
        plt.plot(V, I, ".", label=_case_tag_to_label(k))
    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.xlabel(f"{vcol} (V)")
    plt.ylabel("I (A)")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_overlay_x(sim_dict: dict, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure()
    for k, df in sim_dict.items():
        V = df["Vp"].to_numpy(float)
        x = df["x"].to_numpy(float)
        plt.plot(V, x, ".", label=_case_tag_to_label(k))
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel("Vp (V)")
    plt.ylabel("x (state)")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_overlay_vcmd_time(sim_dict: dict, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure()
    for k, df in sim_dict.items():
        t = df["time"].to_numpy(float)
        v = df["Vcmd"].to_numpy(float)
        plt.plot(t, v, label=_case_tag_to_label(k))
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel("time (s)")
    plt.ylabel("Vcmd (V)")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_overlay_x_time(sim_dict: dict, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure()
    for k, df in sim_dict.items():
        t = df["time"].to_numpy(float)
        x = df["x"].to_numpy(float)
        plt.plot(t, x, label=_case_tag_to_label(k))
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel("time (s)")
    plt.ylabel("x (state)")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_overlay_x_phase(sim_dict: dict, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure()
    for k, df in sim_dict.items():
        ph = df["phase"].to_numpy(float)
        x = df["x"].to_numpy(float)
        plt.plot(ph, x, label=_case_tag_to_label(k))
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel("normalized phase (0~1)")
    plt.ylabel("x (state)")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_compare_meas_vs_sim_abs(V_meas, I_meas, sim_df: pd.DataFrame, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    Vsim = sim_df["Vp"].to_numpy(float)
    Isim = sim_df["I"].to_numpy(float)

    plt.figure()
    plt.semilogy(V_meas, np.abs(I_meas) + I_FLOOR_ABS, ".", label="meas")
    plt.semilogy(Vsim, np.abs(Isim) + I_FLOOR_ABS, ".", label="sim")
    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.xlabel("V (V)")
    plt.ylabel("|I| (A)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_metric_vs_frequency(df: pd.DataFrame, ycol: str, out_png: Path, title: str, ylabel: str):
    import matplotlib.pyplot as plt

    df = df.copy()
    df = df[np.isfinite(df["frequency_Hz"]) & np.isfinite(df[ycol])]
    if len(df) == 0:
        return

    df = df.sort_values("frequency_Hz")

    plt.figure()
    plt.plot(df["frequency_Hz"], df[ycol], "o-")
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel("frequency (Hz)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# ============================================================
# Case summary
# ============================================================

def summarize_case(df: pd.DataFrame, tstop: float, V_cmd: np.ndarray, path_nodes: list):
    V = df["Vp"].to_numpy(float)
    I = df["I"].to_numpy(float)
    x = df["x"].to_numpy(float)
    xh = df["xh"].to_numpy(float)

    out = {
        "N": int(len(df)),
        "Vp_min": float(np.min(V)),
        "Vp_max": float(np.max(V)),
        "Imax_abs": float(np.max(np.abs(I))),
    }

    out.update(calc_cmd_metrics(V_cmd=V_cmd, tstop=tstop, num_segments=max(len(path_nodes) - 1, 1)))
    out.update(calc_sweep_rates(path_nodes=path_nodes, tstop=tstop))
    out.update(compute_return_metrics(df))
    out.update(calc_loop_metrics(V=V, I=I))
    out.update(compute_branch_metrics(df))

    # convenience fields often cited in thesis
    out["x_final_over_initial_ratio"] = float(x[-1] / x[0]) if abs(x[0]) > 1e-30 else np.nan
    out["xh_final_over_initial_ratio"] = float(xh[-1] / xh[0]) if abs(xh[0]) > 1e-30 else np.nan

    return out


# ============================================================
# Case builders
# ============================================================

def build_continuous_cases(V_meas: np.ndarray, pts_per_ramp: int, hold_pts: int):
    """
    Continuous linear-ramp cases.
    """
    cases = []

    if ENABLE_BASELINE:
        cases.append({
            "case_name": "measV",
            "V_cmd": V_meas,
            "path_nodes": [float(V_meas[0]), float(V_meas[-1])],
        })

        nodes = [0.0, FIXED_BASELINE_VPOS, 0.0, FIXED_BASELINE_VNEG, 0.0]
        V_cmd = make_voltage_sequence(nodes, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
        cases.append({
            "case_name": f"fixed_vpos{FIXED_BASELINE_VPOS:g}_vneg{FIXED_BASELINE_VNEG:g}_return",
            "V_cmd": V_cmd,
            "path_nodes": nodes,
        })

    if ENABLE_AMPLITUDE_SWEEP:
        vneg = float(FIXED_VNEG_FOR_VPOS_SWEEP)
        for vpos in VPOS_LIST:
            nodes = [0.0, float(vpos), 0.0, vneg, 0.0]
            V_cmd = make_voltage_sequence(nodes, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
            cases.append({
                "case_name": f"amp_vpos{vpos:g}_vneg{vneg:g}_return",
                "V_cmd": V_cmd,
                "path_nodes": nodes,
            })

        vpos = float(FIXED_VPOS_FOR_VNEG_SWEEP)
        for vneg in VNEG_LIST:
            nodes = [0.0, vpos, 0.0, float(vneg), 0.0]
            V_cmd = make_voltage_sequence(nodes, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
            cases.append({
                "case_name": f"amp_vpos{vpos:g}_vneg{vneg:g}_return",
                "V_cmd": V_cmd,
                "path_nodes": nodes,
            })

        if ENABLE_FULL_VPOSxVNEG_GRID:
            for vpos in VPOS_LIST:
                for vneg in VNEG_LIST:
                    nodes = [0.0, float(vpos), 0.0, float(vneg), 0.0]
                    V_cmd = make_voltage_sequence(nodes, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
                    cases.append({
                        "case_name": f"grid_vpos{vpos:g}_vneg{vneg:g}_return",
                        "V_cmd": V_cmd,
                        "path_nodes": nodes,
                    })

    if ENABLE_MULTI_NEG_SWEEP:
        for vpos in VPOS_LIST:
            for seq in MULTI_NEG_SEQ_LIST:
                nodes = [0.0, float(vpos), 0.0]
                for vneg in seq:
                    nodes += [float(vneg), 0.0]
                V_cmd = make_voltage_sequence(nodes, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
                seq_tag = "_".join([f"{v:g}" for v in seq])
                cases.append({
                    "case_name": f"multi_vpos{vpos:g}_seq_{seq_tag}",
                    "V_cmd": V_cmd,
                    "path_nodes": nodes,
                })

    if ENABLE_REPEAT_NEG_SWEEP:
        for vpos in REPEAT_VPOS_LIST:
            for vneg in REPEAT_NEG_LIST:
                for repN in REPEAT_COUNT_LIST:
                    nodes = [0.0, float(vpos), 0.0]
                    for _ in range(int(repN)):
                        nodes += [float(vneg), 0.0]
                    V_cmd = make_voltage_sequence(nodes, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
                    cases.append({
                        "case_name": f"repeat_vpos{vpos:g}_vneg{vneg:g}_x{repN}",
                        "V_cmd": V_cmd,
                        "path_nodes": nodes,
                    })

    return cases


def build_discrete_cases(levels_per_ramp: int, hold_pts_per_level: int):
    """
    Discretized-input / staircase cases.
    """
    cases = []
    if ENABLE_DISCRETE_LEVEL_SWEEP:
        nodes = [0.0, FIXED_BASELINE_VPOS, 0.0, FIXED_BASELINE_VNEG, 0.0]
        V_cmd = make_staircase_sequence(
            nodes,
            levels_per_ramp=levels_per_ramp,
            hold_pts_per_level=hold_pts_per_level,
        )
        cases.append({
            "case_name": f"stairs_vpos{FIXED_BASELINE_VPOS:g}_vneg{FIXED_BASELINE_VNEG:g}_return",
            "V_cmd": V_cmd,
            "path_nodes": nodes,
        })
    return cases


def should_keep_in_baseline_frequency_summary(row: dict):
    case_name = str(row.get("case", ""))
    input_style = str(row.get("input_style", ""))
    if input_style not in BASELINE_SUMMARY_INPUT_STYLES:
        return False
    return any(case_name.startswith(prefix) for prefix in BASELINE_SUMMARY_CASE_PREFIXES)


def make_baseline_frequency_outputs(summary_df: pd.DataFrame):
    if len(summary_df) == 0:
        return

    mask = summary_df.apply(lambda r: should_keep_in_baseline_frequency_summary(r.to_dict()), axis=1)
    base_df = summary_df[mask].copy()
    if len(base_df) == 0:
        return

    base_df = base_df.sort_values(["mode", "case", "frequency_Hz", "time_scale"])
    base_df.to_csv(OUT_DIR / "summary_baseline_frequency.csv", index=False)

    # representative subsets
    for mode in sorted(base_df["mode"].dropna().unique()):
        dmode = base_df[base_df["mode"] == mode].copy()
        if len(dmode) == 0:
            continue

        dmode.to_csv(OUT_DIR / f"summary_baseline_frequency_{mode}.csv", index=False)

        plot_metric_vs_frequency(
            dmode, "x_recovery_abs",
            METRIC_PLOT_DIR / f"{mode}_x_recovery_abs_vs_freq.png",
            f"x recovery error vs frequency ({mode})",
            "|x_end - x_start|",
        )
        plot_metric_vs_frequency(
            dmode, "x_span",
            METRIC_PLOT_DIR / f"{mode}_x_span_vs_freq.png",
            f"x span vs frequency ({mode})",
            "x_max - x_min",
        )
        plot_metric_vs_frequency(
            dmode, "loop_area_abs_path_A_V",
            METRIC_PLOT_DIR / f"{mode}_loop_area_abs_vs_freq.png",
            f"Hysteresis area vs frequency ({mode})",
            "Σ|I_mid·dV| (A·V)",
        )
        if "branch_sep_I_at_V0p0" in dmode.columns:
            plot_metric_vs_frequency(
                dmode, "branch_sep_I_at_V0p0",
                METRIC_PLOT_DIR / f"{mode}_branch_sep_at_0V_vs_freq.png",
                f"Branch separation at 0 V vs frequency ({mode})",
                "|ΔI| at 0 V (A)",
            )


# ============================================================
# Main
# ============================================================

def main():
    ensure_dirs()

    if not CSV_MEAS.exists():
        raise FileNotFoundError(f"Missing {CSV_MEAS}")
    if not NGSPICE.exists():
        raise FileNotFoundError(f"Missing {NGSPICE}")
    if not TEMPLATE_SRC.exists():
        raise FileNotFoundError(f"Missing {TEMPLATE_SRC}")
    if not THETA_PATH.exists():
        raise FileNotFoundError(f"Missing {THETA_PATH}")

    V_meas, I_meas = read_meas_csv(CSV_MEAS)
    theta = read_theta_best(THETA_PATH)

    N_ref = len(V_meas)
    ic_est = estimate_icomp_pos(V_meas, I_meas)

    print("[INFO] template:", TEMPLATE_SRC)
    print("[INFO] theta_best:", THETA_PATH)
    print(f"[INFO] meas points={N_ref} Vmin={V_meas.min():.4g} Vmax={V_meas.max():.4g}")
    print(f"[INFO] icomp_est≈{ic_est:.4g} A (for LIMIT mode baseline)")
    print(f"[INFO] TSTOP_BASE={TSTOP_BASE} DTMAX_SIM={DTMAX_SIM} tstep_print={pick_tstep_print(N_ref, TSTOP_BASE):.3g}")

    LIMIT_BASE = dict(
        icomp_pos=ic_est,
        vcomp=VCOMP_FIXED,
        rlo=RLO_FIXED,
        rhi=RHI_FIXED,
        vslope_pos=VSLOPE_POS_FIX,
        islope_rel=ISLOPE_REL,
    )

    NOLIMIT = dict(
        icomp_pos=1e30,
        vcomp=VCOMP_FIXED,
        rlo=1e-3,
        rhi=1e-3,
        vslope_pos=VSLOPE_POS_FIX,
        islope_rel=ISLOPE_REL,
    )

    overlay_bucket = {
        "LIMIT": {},
        "NOLIMIT": {},
    }
    overlay_bucket_freq = {
        "LIMIT": {},
        "NOLIMIT": {},
    }

    summary_rows = []

    if ENABLE_RATE_SWEEP:
        time_scale_list = TIME_SCALE_LIST
        ppr_list = PTS_PER_RAMP_LIST
        hold_list = HOLD_PTS_LIST
    else:
        time_scale_list = [1.0]
        ppr_list = [max(160, int(N_ref / 6))]
        hold_list = [0]

    if ENABLE_COMPLIANCE_SWEEP:
        icomp_scales = ICOMP_SCALE_LIST
        vcomp_list = VCOMP_LIST
        rhi_list = RHI_LIST
    else:
        icomp_scales = [1.0]
        vcomp_list = [VCOMP_FIXED]
        rhi_list = [RHI_FIXED]

    primary_time_scale = time_scale_list[0] if time_scale_list else 1.0
    primary_ppr = ppr_list[0] if ppr_list else np.nan
    primary_hold = hold_list[0] if hold_list else np.nan

    runs_per_case = 0
    if RUN_NOLIMIT_MODE:
        runs_per_case += 1
    if RUN_LIMIT_MODE:
        runs_per_case += len(icomp_scales) * len(vcomp_list) * len(rhi_list)

    total_planned = 0

    for pts_per_ramp in ppr_list:
        for hold_pts in hold_list:
            total_planned += len(build_continuous_cases(V_meas, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts))

    if ENABLE_DISCRETE_LEVEL_SWEEP:
        for levels_per_ramp in LEVELS_PER_RAMP_LIST:
            total_planned += len(build_discrete_cases(levels_per_ramp, HOLD_PTS_PER_LEVEL))

    total_planned *= len(time_scale_list) * runs_per_case
    print(f"[INFO] planned simulation runs ≈ {total_planned}")

    sim_count = 0
    fail_count = 0
    overlay_count = 0
    used_case_tags = {}

    def unique_case_tag(tag: str) -> str:
        n = used_case_tags.get(tag, 0) + 1
        used_case_tags[tag] = n
        if n == 1:
            return tag
        return safe_tag(f"{tag}_dup{n}")

    def maybe_keep_frequency_overlay(mode: str, case_name: str, df: pd.DataFrame, cfg_tag: str, input_style: str):
        keep = (
            input_style == "linear" and
            case_name.startswith(f"fixed_vpos{FIXED_BASELINE_VPOS:g}_vneg{FIXED_BASELINE_VNEG:g}")
        )
        if keep:
            overlay_bucket_freq[mode][safe_tag(f"{case_name}_{cfg_tag}")] = df

    def run_case_modes(case_name, V_cmd, path_nodes, cfg_tag, time_scale, input_style,
                       pts_per_ramp=np.nan, hold_pts=np.nan, levels_per_ramp=np.nan, hold_pts_per_level=np.nan):
        nonlocal sim_count, fail_count, overlay_count

        tstop = TSTOP_BASE * float(time_scale)

        # ---------- NOLIMIT ----------
        if RUN_NOLIMIT_MODE:
            raw_tag = safe_tag(f"{case_name}_NOLIMIT_{cfg_tag}")
            case_tag = unique_case_tag(raw_tag)

            retN = simulate_from_fitdeck(
                theta, V_cmd, case_tag=case_tag, tstop=tstop, **NOLIMIT
            )

            if retN is None:
                fail_count += 1
            else:
                sim_count += 1
                print(f"[PROGRESS] sim#{sim_count}/{total_planned} OK: {case_tag}")

                dfN, plot_dirN = retN
                make_all_case_plots(dfN, plot_dirN, case_tag)

                s = summarize_case(dfN, tstop=tstop, V_cmd=V_cmd, path_nodes=path_nodes)
                s.update({
                    "mode": "NOLIMIT",
                    "case": case_name,
                    "case_tag": case_tag,
                    "input_style": input_style,
                    "pts_per_ramp": pts_per_ramp,
                    "hold_pts": hold_pts,
                    "levels_per_ramp": levels_per_ramp,
                    "hold_pts_per_level": hold_pts_per_level,
                    "time_scale": float(time_scale),
                    "tstop": float(tstop),
                    "icomp_scale": np.nan,
                    "vcomp": NOLIMIT["vcomp"],
                    "rhi": NOLIMIT["rhi"],
                })
                summary_rows.append(s)
                save_case_text_summary(plot_dirN, s)

                keep_overlay_n = False
                if case_name == "measV":
                    keep_overlay_n = True
                if (
                    case_name.startswith("amp_")
                    and time_scale == primary_time_scale
                    and pts_per_ramp == primary_ppr
                    and hold_pts == primary_hold
                ):
                    keep_overlay_n = True
                if case_name.startswith("stairs_") and time_scale == primary_time_scale:
                    keep_overlay_n = True

                if keep_overlay_n:
                    overlay_bucket["NOLIMIT"][safe_tag(f"{case_name}_{cfg_tag}")] = dfN

                maybe_keep_frequency_overlay("NOLIMIT", case_name, dfN, cfg_tag, input_style)

        # ---------- LIMIT ----------
        if RUN_LIMIT_MODE:
            for icomp_scale in icomp_scales:
                for vcomp in vcomp_list:
                    for rhi in rhi_list:
                        LIMIT = dict(LIMIT_BASE)
                        LIMIT["icomp_pos"] = float(ic_est) * float(icomp_scale)
                        LIMIT["vcomp"] = float(vcomp)
                        LIMIT["rhi"] = float(rhi)

                        raw_tag = safe_tag(f"{case_name}_LIMIT_{cfg_tag}")
                        case_tag = unique_case_tag(raw_tag)

                        retL = simulate_from_fitdeck(
                            theta, V_cmd, case_tag=case_tag, tstop=tstop, **LIMIT
                        )

                        if retL is None:
                            fail_count += 1
                            continue

                        sim_count += 1
                        print(f"[PROGRESS] sim#{sim_count}/{total_planned} OK: {case_tag}")

                        dfL, plot_dirL = retL
                        make_all_case_plots(dfL, plot_dirL, case_tag)

                        s = summarize_case(dfL, tstop=tstop, V_cmd=V_cmd, path_nodes=path_nodes)
                        s.update({
                            "mode": "LIMIT",
                            "case": case_name,
                            "case_tag": case_tag,
                            "input_style": input_style,
                            "pts_per_ramp": pts_per_ramp,
                            "hold_pts": hold_pts,
                            "levels_per_ramp": levels_per_ramp,
                            "hold_pts_per_level": hold_pts_per_level,
                            "time_scale": float(time_scale),
                            "tstop": float(tstop),
                            "icomp_scale": float(icomp_scale),
                            "vcomp": float(vcomp),
                            "rhi": float(rhi),
                        })
                        summary_rows.append(s)
                        save_case_text_summary(plot_dirL, s)

                        keep_overlay = False
                        is_baseline_limit = (
                            icomp_scale == 1.0
                            and vcomp == VCOMP_FIXED
                            and rhi == RHI_FIXED
                        )

                        if is_baseline_limit:
                            if (
                                case_name == "measV"
                                and time_scale == primary_time_scale
                                and pts_per_ramp == primary_ppr
                                and hold_pts == primary_hold
                            ):
                                keep_overlay = True

                            if (
                                case_name.startswith("amp_")
                                and time_scale == primary_time_scale
                                and pts_per_ramp == primary_ppr
                                and hold_pts == primary_hold
                            ):
                                keep_overlay = True

                            if (
                                case_name.startswith("stairs_")
                                and time_scale == primary_time_scale
                            ):
                                keep_overlay = True

                        if keep_overlay:
                            overlay_bucket["LIMIT"][safe_tag(f"{case_name}_{cfg_tag}")] = dfL
                            overlay_count += 1

                        maybe_keep_frequency_overlay("LIMIT", case_name, dfL, cfg_tag, input_style)

    # ------------------------------------------------------------
    # Run continuous linear-ramp cases
    # ------------------------------------------------------------
    for pts_per_ramp in ppr_list:
        for hold_pts in hold_list:
            cont_cases = build_continuous_cases(V_meas, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)

            for time_scale in time_scale_list:
                for case in cont_cases:
                    case_name = case["case_name"]
                    V_cmd = case["V_cmd"]
                    path_nodes = case["path_nodes"]

                    cfg_tag = safe_tag(f"ppr{pts_per_ramp}_hold{hold_pts}_ts{time_scale:g}")
                    run_case_modes(
                        case_name=case_name,
                        V_cmd=V_cmd,
                        path_nodes=path_nodes,
                        cfg_tag=cfg_tag,
                        time_scale=time_scale,
                        input_style="linear",
                        pts_per_ramp=pts_per_ramp,
                        hold_pts=hold_pts,
                    )

    # ------------------------------------------------------------
    # Run discretized-input staircase cases
    # ------------------------------------------------------------
    if ENABLE_DISCRETE_LEVEL_SWEEP:
        for levels_per_ramp in LEVELS_PER_RAMP_LIST:
            disc_cases = build_discrete_cases(levels_per_ramp, HOLD_PTS_PER_LEVEL)

            for time_scale in time_scale_list:
                for case in disc_cases:
                    case_name = case["case_name"]
                    V_cmd = case["V_cmd"]
                    path_nodes = case["path_nodes"]

                    cfg_tag = safe_tag(f"lpr{levels_per_ramp}_hpl{HOLD_PTS_PER_LEVEL}_ts{time_scale:g}")
                    run_case_modes(
                        case_name=case_name,
                        V_cmd=V_cmd,
                        path_nodes=path_nodes,
                        cfg_tag=cfg_tag,
                        time_scale=time_scale,
                        input_style="staircase",
                        levels_per_ramp=levels_per_ramp,
                        hold_pts_per_level=HOLD_PTS_PER_LEVEL,
                    )

    # ------------------------------------------------------------
    # Overlay plots
    # ------------------------------------------------------------
    if overlay_bucket["LIMIT"]:
        plot_overlay_abs(
            overlay_bucket["LIMIT"],
            OVERLAY_DIR / "overlay_LIMIT_logabsI.png",
            "Overlay |I| (LIMIT) - selected sweep cases",
            use_vp=True,
        )
        plot_overlay_symlog(
            overlay_bucket["LIMIT"],
            OVERLAY_DIR / "overlay_LIMIT_symlog.png",
            "Overlay I (symlog) (LIMIT) - selected sweep cases",
            use_vp=True,
        )
        plot_overlay_x(
            overlay_bucket["LIMIT"],
            OVERLAY_DIR / "overlay_LIMIT_x_vs_Vp.png",
            "Overlay x vs Vp (LIMIT) - selected sweep cases",
        )
        plot_overlay_vcmd_time(
            overlay_bucket["LIMIT"],
            OVERLAY_DIR / "overlay_LIMIT_Vcmd_vs_time.png",
            "Overlay input voltage vs time (LIMIT) - selected sweep cases",
        )

    if overlay_bucket["NOLIMIT"]:
        plot_overlay_abs(
            overlay_bucket["NOLIMIT"],
            OVERLAY_DIR / "overlay_NOLIMIT_logabsI.png",
            "Overlay |I| (NOLIMIT) - representative subset",
            use_vp=True,
        )
        plot_overlay_symlog(
            overlay_bucket["NOLIMIT"],
            OVERLAY_DIR / "overlay_NOLIMIT_symlog.png",
            "Overlay I (symlog) (NOLIMIT) - representative subset",
            use_vp=True,
        )
        plot_overlay_vcmd_time(
            overlay_bucket["NOLIMIT"],
            OVERLAY_DIR / "overlay_NOLIMIT_Vcmd_vs_time.png",
            "Overlay input voltage vs time (NOLIMIT) - representative subset",
        )

    # frequency-focus overlays for thesis 4.4
    for mode in ("LIMIT", "NOLIMIT"):
        if overlay_bucket_freq[mode]:
            plot_overlay_abs(
                overlay_bucket_freq[mode],
                OVERLAY_DIR / f"overlay_{mode}_freq_logabsI.png",
                f"Frequency sweep overlay |I| ({mode})",
                use_vp=True,
            )
            plot_overlay_symlog(
                overlay_bucket_freq[mode],
                OVERLAY_DIR / f"overlay_{mode}_freq_symlog.png",
                f"Frequency sweep overlay I–V ({mode})",
                use_vp=True,
            )
            plot_overlay_x(
                overlay_bucket_freq[mode],
                OVERLAY_DIR / f"overlay_{mode}_freq_x_vs_Vp.png",
                f"Frequency sweep overlay x–V ({mode})",
            )
            plot_overlay_x_time(
                overlay_bucket_freq[mode],
                OVERLAY_DIR / f"overlay_{mode}_freq_x_vs_time.png",
                f"Frequency sweep overlay x–time ({mode})",
            )
            plot_overlay_x_phase(
                overlay_bucket_freq[mode],
                OVERLAY_DIR / f"overlay_{mode}_freq_x_vs_phase.png",
                f"Frequency sweep overlay x–phase ({mode})",
            )
            plot_overlay_vcmd_time(
                overlay_bucket_freq[mode],
                OVERLAY_DIR / f"overlay_{mode}_freq_Vcmd_vs_time.png",
                f"Frequency sweep overlay input voltage vs time ({mode})",
            )

    # meas vs sim overlay (baseline measV) best-effort
    if RUN_LIMIT_MODE:
        key = None
        for k in overlay_bucket["LIMIT"].keys():
            if k.startswith("measV_"):
                key = k
                break
        if key is not None:
            plot_compare_meas_vs_sim_abs(
                V_meas,
                I_meas,
                overlay_bucket["LIMIT"][key],
                OVERLAY_DIR / "compare_measV_LIMIT_logabsI.png",
                "Measured vs Simulated (LIMIT) - baseline measV",
            )

    if RUN_NOLIMIT_MODE:
        key = None
        for k in overlay_bucket["NOLIMIT"].keys():
            if k.startswith("measV_"):
                key = k
                break
        if key is not None:
            plot_compare_meas_vs_sim_abs(
                V_meas,
                I_meas,
                overlay_bucket["NOLIMIT"][key],
                OVERLAY_DIR / "compare_measV_NOLIMIT_logabsI.png",
                "Measured vs Simulated (NOLIMIT) - baseline measV",
            )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(OUT_DIR / "summary_all_cases.csv", index=False)
        make_baseline_frequency_outputs(summary_df)

    print("\n[OK] Outputs written to:", OUT_DIR)
    print(" - cases/: each case_tag has decks/logs/sims/plots")
    print(" - overlay/: representative overlay plots + frequency overlays")
    print(" - metrics_vs_frequency/: chapter-4.4 metric plots")
    print(" - summary_all_cases.csv")
    print(" - summary_baseline_frequency.csv")
    print(f"\n[STATS] successful simulations = {sim_count}")
    print(f"[STATS] failed simulations     = {fail_count}")
    print(f"[STATS] overlay-kept cases     = {overlay_count}")
    print("\nRecommended for Chapter 4.4:")
    print(" - Use default TIME_SCALE_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]")
    print(" - Focus on overlay_*_freq_*.png and summary_baseline_frequency.csv")
    print(" - x_vs_time / x_vs_phase are the key plots for state recovery discussion")

if __name__ == "__main__":
    main()
