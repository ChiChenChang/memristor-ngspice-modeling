# analyze_0307_thesis45.py
# ============================================================
# Thesis-oriented sweep analyzer for memdiode fitdeck template
# (NGSpice + Python)
#
# Purpose:
#   - Support thesis section 4.5: comparison under different input-voltage conditions
#   - Keep the original sweep workflow, but make outputs more "thesis ready"
#
# Main additions relative to the earlier analyze script:
#   1) Amplitude sweep enabled by default
#   2) Overlay plots use Vp consistently (same x-axis meaning as single-case IV plots)
#   3) Quantitative thesis metrics added to summary CSV
#   4) Separate CSV tables for case settings / thesis metrics / branch-split metrics
#   5) Automatic grouped overlays for amplitude-vneg / amplitude-vpos / staircase cases
#   6) Automatic baseline-vs-case comparison plots for amplitude cases
#
# Outputs:
#   analyze_result_sweep_thesis45/
#     fitdeck_embedded.cir
#     summary_all_cases.csv
#     thesis_metrics_table.csv
#     case_settings_table.csv
#     branch_metrics_table.csv
#     overlay/*.png
#     cases/<case_tag>/{decks,logs,sims,plots}/...
#
# Run:
#   py analyze_0307_thesis45.py
# ============================================================

import math
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

OUT_DIR = BASE_DIR / "results" / "analysis_voltage"
TEMPLATE_SNAPSHOT = OUT_DIR / "fitdeck_embedded.cir"
CASES_DIR = OUT_DIR / "cases"
OVERLAY_DIR = OUT_DIR / "overlay"

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
# Plot settings / metrics
# =========================
I_FLOOR_ABS = 3e-10
SYMLINTHRESH = 1e-9
BRANCH_SAMPLE_V_LIST = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
NEAR_ZERO_V_WINDOW = 0.12
MAX_FIND_DV = 0.25

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
ENABLE_RATE_SWEEP = False
ENABLE_AMPLITUDE_SWEEP = True
ENABLE_MULTI_NEG_SWEEP = True
ENABLE_REPEAT_NEG_SWEEP = True
ENABLE_COMPLIANCE_SWEEP = False
ENABLE_DISCRETE_LEVEL_SWEEP = False

# ----- Modes -----
RUN_LIMIT_MODE = False
RUN_NOLIMIT_MODE = True

# ----- Shared fixed baseline waveform -----
FIXED_BASELINE_VPOS = 10.0
FIXED_BASELINE_VNEG = -6.0

# ----- Rate / time / continuous linear-ramp sweeps -----
TIME_SCALE_LIST = [1.0]

# Continuous linear ramp sampling
PTS_PER_RAMP_LIST = [160]
HOLD_PTS_LIST = [0]

# ----- Staircase / discretized-input sweeps -----
LEVELS_PER_RAMP_LIST = [3, 17, 2560]
HOLD_PTS_PER_LEVEL = 1

# ----- Amplitude sweeps -----
# 正向振幅掃描：固定 VNEG = -4 V，改變 VPOS
AMP_VPOS_LIST = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# 負向振幅掃描：固定 VPOS = +10 V，改變 VNEG
VNEG_LIST = [-2.0, -3.0, -4.0, -5.0, -6.0, -8.0, -10.0]

FIXED_VNEG_FOR_VPOS_SWEEP = -6.0
FIXED_VPOS_FOR_VNEG_SWEEP = 10.0
ENABLE_FULL_VPOSxVNEG_GRID = False

# ----- Multi negative sequences -----
MULTI_NEG_VPOS_LIST = [6.0, 8.0, 10.0]

MULTI_NEG_SEQ_LIST = [
    [-2.0, -4.0, -6.0, -8.0, -10.0],
    [-10.0, -8.0, -6.0, -4.0, -2.0],
]

# ----- Repeat negative cycles -----
# 論文 4.5 若要寫 +10/-10 V 重複循環，這裡必須這樣設
REPEAT_NEG_LIST = [-10.0]
REPEAT_COUNT_LIST = [1, 2, 3, 5]
REPEAT_VPOS_LIST = [10.0]

# ----- Compliance sweeps (LIMIT only) -----
ICOMP_SCALE_LIST = [0.5, 1.0, 2.0]
VCOMP_LIST = [0.0]
RHI_LIST = [2e8]

# ----- Robustness / fair-time settings -----
NORMALIZE_TSTOP_BY_RAMP_COUNT = True
BASELINE_RAMP_COUNT = 4

INCOMPLETE_TIME_FRAC = 0.995
RETRY_ON_INCOMPLETE = True
RETRY_DTMAX_FACTORS = [0.5, 0.25]
RETRY_TIMEOUTS = [240, 360]
SAVE_FAILED_CASES = True

# ============================================================
# Utilities
# ============================================================

def ensure_dirs():
    OUT_DIR.mkdir(exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

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


def fmt_num(x):
    if x is None:
        return "na"
    if isinstance(x, float):
        if math.isnan(x):
            return "na"
        if abs(x - round(x)) < 1e-12:
            return str(int(round(x)))
    return f"{x:g}"


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


def is_simulation_complete(df: pd.DataFrame, tstop: float):
    if df is None or len(df) == 0:
        return False
    try:
        time_end = float(df["time"].iloc[-1])
    except Exception:
        return False
    return time_end >= float(INCOMPLETE_TIME_FRAC) * float(tstop)


def compute_case_tstop(case_meta: dict, time_scale: float):
    if case_meta.get("input_style") == "meas_trace":
        return TSTOP_BASE * float(time_scale)
    if NORMALIZE_TSTOP_BY_RAMP_COUNT:
        num_ramps = case_meta.get("num_ramps", BASELINE_RAMP_COUNT)
        if isinstance(num_ramps, float) and np.isnan(num_ramps):
            num_ramps = BASELINE_RAMP_COUNT
        num_ramps = max(1, int(num_ramps))
        return TSTOP_BASE * float(time_scale) * (num_ramps / BASELINE_RAMP_COUNT)
    return TSTOP_BASE * float(time_scale)

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
    Staircase waveform.
    Example: 0 -> 10 -> 0 -> -10 -> 0 with levels_per_ramp=3 becomes
             0 -> 5 -> 10 -> 5 -> 0 -> -5 -> -10 -> -5 -> 0
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
    dtmax_sim: float = DTMAX_SIM,
    timeout_s: int = 180,
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
        "@DTMAX@": f"{float(dtmax_sim):.12g}",
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

    rc = run_ngspice(deck_path, log_path, cwd=case_dir, timeout_s=timeout_s)
    if rc != 0 or (not sim_path.exists()):
        print(f"[SIM] ngspice failed: {case_tag}, rc={rc}")
        print("[log tail]\n" + tail_text(log_path))
        return None

    t, vcmd, vp, idev, vx, vxh = load_wrdata(sim_path)
    df = pd.DataFrame({"time": t, "Vcmd": vcmd, "Vp": vp, "I": idev, "x": vx, "xh": vxh})
    df.to_csv(sim_dir / f"{case_tag}_sim.csv", index=False)
    return df, plot_dir, sim_dir, log_path

# ============================================================
# Metrics helpers
# ============================================================

def fill_zero_sign(sign_arr: np.ndarray):
    s = sign_arr.astype(float).copy()
    n = len(s)
    if n == 0:
        return s

    # forward fill
    for i in range(1, n):
        if s[i] == 0:
            s[i] = s[i - 1]
    # backward fill for leading zeros
    for i in range(n - 2, -1, -1):
        if s[i] == 0:
            s[i] = s[i + 1]
    return s


def get_sweep_direction(v: np.ndarray):
    if len(v) <= 1:
        return np.zeros_like(v)
    dv = np.diff(v)
    sign = np.sign(dv)
    sign = np.r_[sign, sign[-1]]
    sign = fill_zero_sign(sign)
    return sign


def nearest_index(mask: np.ndarray, target_v: float, v_arr: np.ndarray):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    local = idx[np.argmin(np.abs(v_arr[idx] - target_v))]
    if abs(v_arr[local] - target_v) > MAX_FIND_DV:
        return None
    return int(local)


def safe_logabs(x):
    return float(np.log10(abs(float(x)) + I_FLOOR_ABS))


def trapezoid_area(y, x):
    """NumPy-version-safe trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    # Fallback for older/minimal NumPy builds
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2 or x.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


def compute_branch_metrics(df: pd.DataFrame, sample_v_list=None):
    if sample_v_list is None:
        sample_v_list = BRANCH_SAMPLE_V_LIST

    V = df["Vp"].to_numpy(float)
    Vcmd = df["Vcmd"].to_numpy(float)
    I = df["I"].to_numpy(float)
    x = df["x"].to_numpy(float)
    xh = df["xh"].to_numpy(float)
    t = df["time"].to_numpy(float)

    dir_sign = get_sweep_direction(Vcmd)

    pos_up = (V >= 0) & (dir_sign > 0)
    pos_down = (V >= 0) & (dir_sign < 0)
    neg_down = (V <= 0) & (dir_sign < 0)
    neg_up = (V <= 0) & (dir_sign > 0)

    out = {
        "N": int(len(df)),
        "time_start_s": float(t[0]),
        "time_end_s": float(t[-1]),
        "Vcmd_min": float(np.min(Vcmd)),
        "Vcmd_max": float(np.max(Vcmd)),
        "Vp_min": float(np.min(V)),
        "Vp_max": float(np.max(V)),
        "Imax_abs": float(np.max(np.abs(I))),
        "I_pos_max": float(np.max(I)) if len(I) else np.nan,
        "I_neg_min": float(np.min(I)) if len(I) else np.nan,
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "x_range": float(np.max(x) - np.min(x)),
        "xh_min": float(np.min(xh)),
        "xh_max": float(np.max(xh)),
        "xh_range": float(np.max(xh) - np.min(xh)),
        "xh_end": float(xh[-1]),
        "loop_area_idv": float(trapezoid_area(I, V)),
        "loop_area_abs_idv": float(abs(trapezoid_area(I, V))),
    }

    # Near-zero metrics
    zero_mask = np.abs(V) <= NEAR_ZERO_V_WINDOW
    if np.any(zero_mask):
        Iz = I[zero_mask]
        out["I_zero_abs_max"] = float(np.max(np.abs(Iz)))
        out["I_zero_mean"] = float(np.mean(Iz))
        out["I_zero_abs_mean"] = float(np.mean(np.abs(Iz)))
    else:
        out["I_zero_abs_max"] = np.nan
        out["I_zero_mean"] = np.nan
        out["I_zero_abs_mean"] = np.nan

    # Branch-based samples
    for vtar in sample_v_list:
        tag = str(vtar).replace(".", "p")

        idx_pu = nearest_index(pos_up, +vtar, V)
        idx_pd = nearest_index(pos_down, +vtar, V)
        idx_nd = nearest_index(neg_down, -vtar, V)
        idx_nu = nearest_index(neg_up, -vtar, V)

        out[f"I_pos_up_at_{tag}V"] = float(I[idx_pu]) if idx_pu is not None else np.nan
        out[f"I_pos_down_at_{tag}V"] = float(I[idx_pd]) if idx_pd is not None else np.nan
        out[f"I_neg_down_at_{tag}V"] = float(I[idx_nd]) if idx_nd is not None else np.nan
        out[f"I_neg_up_at_{tag}V"] = float(I[idx_nu]) if idx_nu is not None else np.nan

        if idx_pu is not None and idx_pd is not None:
            out[f"branch_sep_pos_absI_at_{tag}V"] = float(abs(I[idx_pu] - I[idx_pd]))
            out[f"branch_sep_pos_logdec_at_{tag}V"] = float(abs(safe_logabs(I[idx_pu]) - safe_logabs(I[idx_pd])))
        else:
            out[f"branch_sep_pos_absI_at_{tag}V"] = np.nan
            out[f"branch_sep_pos_logdec_at_{tag}V"] = np.nan

        if idx_nd is not None and idx_nu is not None:
            out[f"branch_sep_neg_absI_at_{tag}V"] = float(abs(I[idx_nd] - I[idx_nu]))
            out[f"branch_sep_neg_logdec_at_{tag}V"] = float(abs(safe_logabs(I[idx_nd]) - safe_logabs(I[idx_nu])))
        else:
            out[f"branch_sep_neg_absI_at_{tag}V"] = np.nan
            out[f"branch_sep_neg_logdec_at_{tag}V"] = np.nan

    return out


def flatten_branch_metrics_for_table(case_tag: str, case_name: str, mode: str, group: str, df: pd.DataFrame):
    V = df["Vp"].to_numpy(float)
    Vcmd = df["Vcmd"].to_numpy(float)
    I = df["I"].to_numpy(float)
    dir_sign = get_sweep_direction(Vcmd)

    rows = []
    for vtar in BRANCH_SAMPLE_V_LIST:
        tag = str(vtar).replace(".", "p")
        pos_up = (V >= 0) & (dir_sign > 0)
        pos_down = (V >= 0) & (dir_sign < 0)
        neg_down = (V <= 0) & (dir_sign < 0)
        neg_up = (V <= 0) & (dir_sign > 0)

        idx_pu = nearest_index(pos_up, +vtar, V)
        idx_pd = nearest_index(pos_down, +vtar, V)
        idx_nd = nearest_index(neg_down, -vtar, V)
        idx_nu = nearest_index(neg_up, -vtar, V)

        rows.append({
            "case_tag": case_tag,
            "case": case_name,
            "mode": mode,
            "group": group,
            "sample_absV": float(vtar),
            "I_pos_up": float(I[idx_pu]) if idx_pu is not None else np.nan,
            "I_pos_down": float(I[idx_pd]) if idx_pd is not None else np.nan,
            "pos_abs_sep": float(abs(I[idx_pu] - I[idx_pd])) if idx_pu is not None and idx_pd is not None else np.nan,
            "pos_logdec_sep": float(abs(safe_logabs(I[idx_pu]) - safe_logabs(I[idx_pd]))) if idx_pu is not None and idx_pd is not None else np.nan,
            "I_neg_down": float(I[idx_nd]) if idx_nd is not None else np.nan,
            "I_neg_up": float(I[idx_nu]) if idx_nu is not None else np.nan,
            "neg_abs_sep": float(abs(I[idx_nd] - I[idx_nu])) if idx_nd is not None and idx_nu is not None else np.nan,
            "neg_logdec_sep": float(abs(safe_logabs(I[idx_nd]) - safe_logabs(I[idx_nu]))) if idx_nd is not None and idx_nu is not None else np.nan,
        })
    return rows



def _segment_boundaries_from_vcmd(vcmd: np.ndarray):
    vcmd = np.asarray(vcmd, dtype=float)
    n = len(vcmd)
    if n <= 1:
        return [(0, max(0, n - 1))]

    dv = np.diff(vcmd)
    sign = fill_zero_sign(np.sign(dv))
    starts = [0]
    for i in range(1, len(sign)):
        if sign[i] != sign[i - 1]:
            starts.append(i)
    starts.append(n - 1)

    segs = []
    for a, b in zip(starts[:-1], starts[1:]):
        lo = int(a)
        hi = int(max(a + 1, b))
        if hi > lo:
            segs.append((lo, hi))
    if not segs:
        segs = [(0, n - 1)]
    return segs


def flatten_segment_metrics_for_table(case_tag: str, case_name: str, mode: str, group: str, df: pd.DataFrame):
    Vcmd = df["Vcmd"].to_numpy(float)
    Vp = df["Vp"].to_numpy(float)
    I = df["I"].to_numpy(float)
    x = df["x"].to_numpy(float)
    t = df["time"].to_numpy(float)

    rows = []
    segs = _segment_boundaries_from_vcmd(Vcmd)
    for k, (lo, hi) in enumerate(segs, start=1):
        sl = slice(lo, hi + 1)
        vv = Vcmd[sl]
        vp = Vp[sl]
        ii = I[sl]
        xx = x[sl]
        rows.append({
            "case_tag": case_tag,
            "case": case_name,
            "mode": mode,
            "group": group,
            "segment_index": k,
            "segment_type": "up" if vv[-1] > vv[0] else "down",
            "time_start_s": float(t[lo]),
            "time_end_s": float(t[hi]),
            "Vcmd_start": float(vv[0]),
            "Vcmd_end": float(vv[-1]),
            "Vp_start": float(vp[0]),
            "Vp_end": float(vp[-1]),
            "Vp_min": float(np.min(vp)),
            "Vp_max": float(np.max(vp)),
            "I_min": float(np.min(ii)),
            "I_max": float(np.max(ii)),
            "Imax_abs": float(np.max(np.abs(ii))),
            "x_start": float(xx[0]),
            "x_end": float(xx[-1]),
            "x_min": float(np.min(xx)),
            "x_max": float(np.max(xx)),
            "x_range": float(np.max(xx) - np.min(xx)),
        })
    return rows



# ============================================================
# Thesis-clean legend labels
# ============================================================

def _clean_float_token(s):
    """Format numeric tokens from case tags for concise plot legends."""
    try:
        x = float(str(s).replace('m', '-'))
    except Exception:
        return str(s)
    if abs(x - round(x)) < 1e-10:
        return str(int(round(x)))
    return f"{x:g}"


def _strip_ts_suffix(label: str):
    return re.sub(r"_ts[-+0-9.]+$", "", str(label))


def clean_legend_label(label: str):
    """
    Convert internal case tags into concise thesis-ready legend labels.
    """
    raw = str(label)
    s = _strip_ts_suffix(raw)

    if s == "measV" or s.startswith("measV_"):
        return "measured waveform"

    m = re.search(r"fixed_vpos([-+0-9.]+)_vneg([-+0-9.]+)_return", s)
    if m:
        vp = _clean_float_token(m.group(1))
        vn = _clean_float_token(m.group(2))
        return f"baseline (+{vp}/{vn} V)"

    m = re.search(r"amp_vpos([-+0-9.]+)_vneg([-+0-9.]+)_return", s)
    if m:
        vp = _clean_float_token(m.group(1))
        vn = _clean_float_token(m.group(2))
        return f"Vpos = {vp} V, Vneg = {vn} V"

    m = re.search(r"grid_vpos([-+0-9.]+)_vneg([-+0-9.]+)_return", s)
    if m:
        vp = _clean_float_token(m.group(1))
        vn = _clean_float_token(m.group(2))
        return f"Vpos = {vp} V, Vneg = {vn} V"

    m = re.search(r"repeat_vpos([-+0-9.]+)_vneg([-+0-9.]+)_x(\d+)", s)
    if m:
        vn = _clean_float_token(m.group(2))
        rep = m.group(3)
        return f"Vneg = {vn} V, repeat = {rep}"

    m = re.search(r"multi_vpos([-+0-9.]+)_seq_(.+)", s)
    if m:
        seq = m.group(2)
        vals = [_clean_float_token(v) for v in seq.split("_") if v != ""]
        return "seq: " + "→".join(vals) + " V"

    m = re.search(r"stairs_vpos([-+0-9.]+)_vneg([-+0-9.]+)_return", s)
    if m:
        vp = _clean_float_token(m.group(1))
        vn = _clean_float_token(m.group(2))
        return f"staircase (+{vp}/{vn} V)"

    m = re.search(r"lpr(\d+)_hpl(\d+)", s)
    if m:
        return f"{m.group(1)} levels/ramp"

    s = s.replace("_", " ")
    s = s.replace("NOLIMIT", "")
    s = s.replace("LIMIT", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# Plots
# ============================================================

def _downsample_df(df: pd.DataFrame, cols, max_points=20000):
    n = len(df)
    if n <= max_points:
        return [df[c].to_numpy(float) for c in cols]
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return [df[c].to_numpy(float)[idx] for c in cols]


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
    plot_iv_symlog(df, plot_dir / "IV_symlog.png", f"I–V (symlog) - {tag}", use_vp=True)
    plot_logabsI_vs_V(df, plot_dir / "logabsI_vs_Vp.png", f"log(|I|) vs Vp - {tag}", use_vp=True)
    plot_x_vs_V(df, plot_dir / "x_vs_Vp.png", f"x vs Vp - {tag}", use_vp=True, connect=True)
    plot_x_vs_I(df, plot_dir / "x_vs_logabsI.png", f"x vs log|I| - {tag}", log_absI=True, connect=True)
    plot_IV_colored_by_x(df, plot_dir / "IV_colored_by_x.png", f"I–V colored by x - {tag}", use_vp=True, symlog_y=True)
    plot_butterfly_with_x(df, plot_dir / "butterfly_with_x.png", f"Butterfly & x relationship - {tag}", use_vp=True)

def apply_overlay_legend(plt_module, n_series: int, plot_kind: str = "generic"):
    """Keep legends inside the axes and choose a corner that usually interferes least."""
    if plot_kind == "vcmd_time":
        loc = "upper right"
    elif plot_kind == "x_vs_v":
        loc = "lower right" if n_series > 4 else "best"
    elif plot_kind in {"abs_iv", "symlog_iv"}:
        loc = "upper left" if n_series > 5 else "best"
    else:
        loc = "best"

    fontsize = 6 if n_series > 6 else 7
    plt_module.legend(
        fontsize=fontsize,
        loc=loc,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="0.70",
        labelspacing=0.35,
        borderpad=0.35,
        handletextpad=0.5,
    )
    plt_module.tight_layout()



def plot_overlay_abs(sim_dict: dict, out_png: Path, title: str, use_vp=True):
    import matplotlib.pyplot as plt
    vcol = "Vp" if use_vp else "Vcmd"
    plt.figure()
    for k, df in sim_dict.items():
        V = df[vcol].to_numpy(float)
        I = df["I"].to_numpy(float)
        plt.semilogy(V, np.abs(I) + I_FLOOR_ABS, ".", label=clean_legend_label(k))
    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.xlabel(f"{vcol} (V)")
    plt.ylabel("|I| (A)")
    plt.title(title)
    apply_overlay_legend(plt, len(sim_dict), plot_kind="abs_iv")
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_overlay_symlog(sim_dict: dict, out_png: Path, title: str, use_vp=True):
    import matplotlib.pyplot as plt
    vcol = "Vp" if use_vp else "Vcmd"
    plt.figure()
    plt.yscale("symlog", linthresh=SYMLINTHRESH)
    for k, df in sim_dict.items():
        V = df[vcol].to_numpy(float)
        I = df["I"].to_numpy(float)
        plt.plot(V, I, ".", label=clean_legend_label(k))
    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.xlabel(f"{vcol} (V)")
    plt.ylabel("I (A)")
    plt.title(title)
    apply_overlay_legend(plt, len(sim_dict), plot_kind="symlog_iv")
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_overlay_x(sim_dict: dict, out_png: Path, title: str, use_vp=True):
    import matplotlib.pyplot as plt
    vcol = "Vp" if use_vp else "Vcmd"
    plt.figure()
    for k, df in sim_dict.items():
        V = df[vcol].to_numpy(float)
        x = df["x"].to_numpy(float)
        plt.plot(V, x, ".", label=clean_legend_label(k))
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel(f"{vcol} (V)")
    plt.ylabel("x (state)")
    plt.title(title)
    apply_overlay_legend(plt, len(sim_dict), plot_kind="x_vs_v")
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_overlay_vcmd_time(sim_dict: dict, out_png: Path, title: str):
    import matplotlib.pyplot as plt
    plt.figure()
    for k, df in sim_dict.items():
        t = df["time"].to_numpy(float)
        v = df["Vcmd"].to_numpy(float)
        plt.plot(t, v, label=clean_legend_label(k))
    plt.grid(True, ls="--", alpha=0.35)
    plt.xlabel("time (s)")
    plt.ylabel("Vcmd (V)")
    plt.title(title)
    apply_overlay_legend(plt, len(sim_dict), plot_kind="vcmd_time")
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_compare_meas_vs_sim_abs(V_meas, I_meas, sim_df: pd.DataFrame, out_png: Path, title: str, use_vp=True):
    import matplotlib.pyplot as plt
    vcol = "Vp" if use_vp else "Vcmd"
    Vsim = sim_df[vcol].to_numpy(float)
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


def plot_compare_two_sim_abs(df_ref: pd.DataFrame, df_other: pd.DataFrame, out_png: Path, title: str, label_ref: str, label_other: str, use_vp=True):
    import matplotlib.pyplot as plt
    vcol = "Vp" if use_vp else "Vcmd"
    plt.figure()
    plt.semilogy(df_ref[vcol].to_numpy(float), np.abs(df_ref["I"].to_numpy(float)) + I_FLOOR_ABS, ".", label=clean_legend_label(label_ref))
    plt.semilogy(df_other[vcol].to_numpy(float), np.abs(df_other["I"].to_numpy(float)) + I_FLOOR_ABS, ".", label=clean_legend_label(label_other))
    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.xlabel(f"{vcol} (V)")
    plt.ylabel("|I| (A)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def get_named_overlay_cases(sim_registry_mode: dict, target_names):
    out = {}
    for target in target_names:
        for rec in sim_registry_mode.values():
            if rec["case_meta"]["name"] == target:
                out[target] = rec["df"]
                break
    return out


def get_filtered_overlay_cases(sim_registry_mode: dict, predicate, sort_key=None):
    pairs = []
    for rec in sim_registry_mode.values():
        meta = rec["case_meta"]
        if predicate(meta):
            pairs.append((meta["name"], rec["df"], meta))
    if sort_key is not None:
        pairs.sort(key=lambda x: sort_key(x[2]))
    out = {}
    for name, df, _ in pairs:
        out[name] = df
    return out


def add_overlay_bundle(sim_dict: dict, mode: str, stem: str, title_suffix: str, manifest_rows: list):
    if len(sim_dict) < 2:
        return
    p1 = OVERLAY_DIR / f"overlay_{mode}_{stem}_Vcmd_vs_time.png"
    plot_overlay_vcmd_time(sim_dict, p1, f"Overlay input voltage vs time ({mode}) - {title_suffix}")
    manifest_rows.append({"mode": mode, "stem": stem, "kind": "Vcmd_vs_time", "path": str(p1)})

    p2 = OVERLAY_DIR / f"overlay_{mode}_{stem}_logabsI_vs_Vp.png"
    plot_overlay_abs(sim_dict, p2, f"Overlay |I| ({mode}) - {title_suffix}", use_vp=True)
    manifest_rows.append({"mode": mode, "stem": stem, "kind": "logabsI_vs_Vp", "path": str(p2)})

    p3 = OVERLAY_DIR / f"overlay_{mode}_{stem}_symlogI_vs_Vp.png"
    plot_overlay_symlog(sim_dict, p3, f"Overlay I ({mode}) - {title_suffix}", use_vp=True)
    manifest_rows.append({"mode": mode, "stem": stem, "kind": "symlogI_vs_Vp", "path": str(p3)})

    p4 = OVERLAY_DIR / f"overlay_{mode}_{stem}_x_vs_Vp.png"
    plot_overlay_x(sim_dict, p4, f"Overlay x vs Vp ({mode}) - {title_suffix}", use_vp=True)
    manifest_rows.append({"mode": mode, "stem": stem, "kind": "x_vs_Vp", "path": str(p4)})

# ============================================================
# Case builders
# ============================================================

def make_case_dict(name, V_cmd, group, input_style, **meta):
    d = {
        "name": name,
        "V_cmd": np.asarray(V_cmd, dtype=float),
        "group": group,
        "input_style": input_style,
        "vcmd_min": float(np.min(V_cmd)),
        "vcmd_max": float(np.max(V_cmd)),
    }
    d.update(meta)
    return d


def build_continuous_cases(V_meas: np.ndarray, pts_per_ramp: int, hold_pts: int):
    cases = []

    if ENABLE_BASELINE:
        cases.append(make_case_dict(
            name="measV",
            V_cmd=V_meas,
            group="baseline_meas",
            input_style="meas_trace",
            vpos=float(np.max(V_meas)),
            vneg=float(np.min(V_meas)),
            pts_per_ramp=np.nan,
            hold_pts=np.nan,
            num_ramps=np.nan,
        ))

        nodes = [0.0, FIXED_BASELINE_VPOS, 0.0, FIXED_BASELINE_VNEG, 0.0]
        V_cmd = make_voltage_sequence(nodes, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
        cases.append(make_case_dict(
            name=f"fixed_vpos{fmt_num(FIXED_BASELINE_VPOS)}_vneg{fmt_num(FIXED_BASELINE_VNEG)}_return",
            V_cmd=V_cmd,
            group="baseline_fixed",
            input_style="linear",
            vpos=float(FIXED_BASELINE_VPOS),
            vneg=float(FIXED_BASELINE_VNEG),
            pts_per_ramp=int(pts_per_ramp),
            hold_pts=int(hold_pts),
            num_ramps=len(nodes) - 1,
        ))

    if ENABLE_AMPLITUDE_SWEEP:
        vneg = float(FIXED_VNEG_FOR_VPOS_SWEEP)
        for vpos in AMP_VPOS_LIST:
            if ENABLE_BASELINE and float(vpos) == float(FIXED_BASELINE_VPOS) and float(vneg) == float(FIXED_BASELINE_VNEG):
                continue
            nodes = [0.0, float(vpos), 0.0, vneg, 0.0]
            V_cmd = make_voltage_sequence(nodes, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
            cases.append(make_case_dict(
                name=f"amp_vpos{fmt_num(vpos)}_vneg{fmt_num(vneg)}_return",
                V_cmd=V_cmd,
                group="amp_vpos",
                input_style="linear",
                vpos=float(vpos),
                vneg=float(vneg),
                pts_per_ramp=int(pts_per_ramp),
                hold_pts=int(hold_pts),
            ))

        vpos = float(FIXED_VPOS_FOR_VNEG_SWEEP)
        for vneg in VNEG_LIST:
            if ENABLE_BASELINE and float(vpos) == float(FIXED_BASELINE_VPOS) and float(vneg) == float(FIXED_BASELINE_VNEG):
                continue
            nodes = [0.0, vpos, 0.0, float(vneg), 0.0]
            V_cmd = make_voltage_sequence(nodes, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
            cases.append(make_case_dict(
                name=f"amp_vpos{fmt_num(vpos)}_vneg{fmt_num(vneg)}_return",
                V_cmd=V_cmd,
                group="amp_vneg",
                input_style="linear",
                vpos=float(vpos),
                vneg=float(vneg),
                pts_per_ramp=int(pts_per_ramp),
                hold_pts=int(hold_pts),
            ))

        if ENABLE_FULL_VPOSxVNEG_GRID:
            for vpos in AMP_VPOS_LIST:
                for vneg in VNEG_LIST:
                    nodes = [0.0, float(vpos), 0.0, float(vneg), 0.0]
                    V_cmd = make_voltage_sequence(nodes, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
                    cases.append(make_case_dict(
                        name=f"grid_vpos{fmt_num(vpos)}_vneg{fmt_num(vneg)}_return",
                        V_cmd=V_cmd,
                        group="amp_grid",
                        input_style="linear",
                        vpos=float(vpos),
                        vneg=float(vneg),
                        pts_per_ramp=int(pts_per_ramp),
                        hold_pts=int(hold_pts),
                    ))

    if ENABLE_MULTI_NEG_SWEEP:
        for vpos in MULTI_NEG_VPOS_LIST:
            for seq in MULTI_NEG_SEQ_LIST:
                nodes = [0.0, float(vpos), 0.0]
                for vneg in seq:
                    nodes += [float(vneg), 0.0]
                V_cmd = make_voltage_sequence(nodes, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
                seq_tag = "_".join([fmt_num(v) for v in seq])
                cases.append(make_case_dict(
                    name=f"multi_vpos{fmt_num(vpos)}_seq_{seq_tag}",
                    V_cmd=V_cmd,
                    group="multi_neg",
                    input_style="linear",
                    vpos=float(vpos),
                    vneg=float(min(seq)),
                    pts_per_ramp=int(pts_per_ramp),
                    hold_pts=int(hold_pts),
                    num_ramps=len(nodes) - 1,
                ))

    if ENABLE_REPEAT_NEG_SWEEP:
        for vpos in REPEAT_VPOS_LIST:
            for vneg in REPEAT_NEG_LIST:
                for repN in REPEAT_COUNT_LIST:
                    nodes = [0.0, float(vpos), 0.0]
                    for _ in range(int(repN)):
                        nodes += [float(vneg), 0.0]
                    V_cmd = make_voltage_sequence(nodes, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
                    cases.append(make_case_dict(
                        name=f"repeat_vpos{fmt_num(vpos)}_vneg{fmt_num(vneg)}_x{repN}",
                        V_cmd=V_cmd,
                        group="repeat_neg",
                        input_style="linear",
                        vpos=float(vpos),
                        vneg=float(vneg),
                        repeat_count=int(repN),
                        pts_per_ramp=int(pts_per_ramp),
                        hold_pts=int(hold_pts),
                        num_ramps=len(nodes) - 1,
                    ))

    return cases


def build_discrete_cases(levels_per_ramp: int, hold_pts_per_level: int):
    cases = []
    if ENABLE_DISCRETE_LEVEL_SWEEP:
        nodes = [0.0, FIXED_BASELINE_VPOS, 0.0, FIXED_BASELINE_VNEG, 0.0]
        V_cmd = make_staircase_sequence(
            nodes,
            levels_per_ramp=levels_per_ramp,
            hold_pts_per_level=hold_pts_per_level,
        )
        cases.append(make_case_dict(
            name=f"stairs_vpos{fmt_num(FIXED_BASELINE_VPOS)}_vneg{fmt_num(FIXED_BASELINE_VNEG)}_return",
            V_cmd=V_cmd,
            group="stairs",
            input_style="staircase",
            vpos=float(FIXED_BASELINE_VPOS),
            vneg=float(FIXED_BASELINE_VNEG),
            levels_per_ramp=int(levels_per_ramp),
            hold_pts_per_level=int(hold_pts_per_level),
            num_ramps=len(nodes) - 1,
        ))
    return cases

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

    overlay_bucket = {"LIMIT": {}, "NOLIMIT": {}}
    group_overlay_bucket = {"LIMIT": {}, "NOLIMIT": {}}
    sim_registry = {"LIMIT": {}, "NOLIMIT": {}}

    summary_rows = []
    branch_table_rows = []
    segment_table_rows = []
    case_meta_rows = []
    failed_rows = []
    overlay_manifest_rows = []

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
    used_case_tags = {}

    def unique_case_tag(tag: str) -> str:
        n = used_case_tags.get(tag, 0) + 1
        used_case_tags[tag] = n
        if n == 1:
            return tag
        return safe_tag(f"{tag}_dup{n}")

    def keep_for_overlay(case_meta, time_scale, pts_per_ramp, hold_pts):
        keep = False
        overlay_groups = {"baseline_meas", "baseline_fixed", "amp_vpos", "amp_vneg", "amp_grid", "multi_neg", "repeat_neg", "stairs"}
        if case_meta["group"] in overlay_groups:
            if time_scale == primary_time_scale:
                if case_meta["input_style"] == "staircase":
                    keep = True
                elif pts_per_ramp == primary_ppr and hold_pts == primary_hold:
                    keep = True
        return keep

    def record_case_outputs(mode, case_tag, case_meta, df, plot_dir, sim_dir, tstop, icomp_scale, vcomp, rhi):
        make_all_case_plots(df, plot_dir, case_tag)

        metrics = compute_branch_metrics(df)
        row = dict(metrics)
        row.update({
            "mode": mode,
            "case": case_meta["name"],
            "case_tag": case_tag,
            "group": case_meta["group"],
            "input_style": case_meta["input_style"],
            "vpos": case_meta.get("vpos", np.nan),
            "vneg": case_meta.get("vneg", np.nan),
            "repeat_count": case_meta.get("repeat_count", np.nan),
            "num_ramps": case_meta.get("num_ramps", np.nan),
            "pts_per_ramp": case_meta.get("pts_per_ramp", np.nan),
            "hold_pts": case_meta.get("hold_pts", np.nan),
            "levels_per_ramp": case_meta.get("levels_per_ramp", np.nan),
            "hold_pts_per_level": case_meta.get("hold_pts_per_level", np.nan),
            "time_scale": float(time_scale),
            "tstop": float(tstop),
            "icomp_scale": icomp_scale,
            "vcomp": vcomp,
            "rhi": rhi,
            "sim_dir": str(sim_dir),
            "plot_dir": str(plot_dir),
        })
        summary_rows.append(row)

        case_meta_rows.append({
            "case_tag": case_tag,
            "case": case_meta["name"],
            "mode": mode,
            "group": case_meta["group"],
            "input_style": case_meta["input_style"],
            "vpos": case_meta.get("vpos", np.nan),
            "vneg": case_meta.get("vneg", np.nan),
            "repeat_count": case_meta.get("repeat_count", np.nan),
            "num_ramps": case_meta.get("num_ramps", np.nan),
            "pts_per_ramp": case_meta.get("pts_per_ramp", np.nan),
            "hold_pts": case_meta.get("hold_pts", np.nan),
            "levels_per_ramp": case_meta.get("levels_per_ramp", np.nan),
            "hold_pts_per_level": case_meta.get("hold_pts_per_level", np.nan),
            "time_scale": float(time_scale),
            "tstop": float(tstop),
            "vcmd_min": case_meta.get("vcmd_min", np.nan),
            "vcmd_max": case_meta.get("vcmd_max", np.nan),
        })

        branch_table_rows.extend(
            flatten_branch_metrics_for_table(
                case_tag=case_tag,
                case_name=case_meta["name"],
                mode=mode,
                group=case_meta["group"],
                df=df,
            )
        )
        segment_table_rows.extend(
            flatten_segment_metrics_for_table(
                case_tag=case_tag,
                case_name=case_meta["name"],
                mode=mode,
                group=case_meta["group"],
                df=df,
            )
        )

        sim_registry[mode][case_tag] = {
            "df": df,
            "case_meta": case_meta,
            "plot_dir": plot_dir,
            "sim_dir": sim_dir,
        }

        if keep_for_overlay(case_meta, time_scale, case_meta.get("pts_per_ramp", np.nan), case_meta.get("hold_pts", np.nan)):
            label = safe_tag(f"{case_meta['name']}_ts{fmt_num(time_scale)}")
            overlay_bucket[mode][label] = df
            group_overlay_bucket[mode].setdefault(case_meta["group"], {})[label] = df


    def run_case_once(case_tag, V_cmd, tstop, mode_kwargs, dtmax_try, timeout_try):
        return simulate_from_fitdeck(
            theta,
            V_cmd,
            case_tag=case_tag,
            tstop=tstop,
            dtmax_sim=dtmax_try,
            timeout_s=timeout_try,
            **mode_kwargs,
        )

    def finalize_failed_case(mode, case_tag, case_meta, tstop, reason, attempt_idx, log_path=None, time_end_s=np.nan):
        failed_rows.append({
            "mode": mode,
            "case": case_meta["name"],
            "case_tag": case_tag,
            "group": case_meta["group"],
            "input_style": case_meta["input_style"],
            "vpos": case_meta.get("vpos", np.nan),
            "vneg": case_meta.get("vneg", np.nan),
            "repeat_count": case_meta.get("repeat_count", np.nan),
            "num_ramps": case_meta.get("num_ramps", np.nan),
            "time_scale": float(time_scale),
            "expected_tstop_s": float(tstop),
            "time_end_s": time_end_s,
            "reason": reason,
            "attempt_idx": int(attempt_idx),
            "log_path": str(log_path) if log_path is not None else "",
            "log_tail": tail_text(log_path, n_lines=80) if log_path is not None else "",
        })

    def run_case_modes(case_meta, cfg_tag, time_scale):
        nonlocal sim_count, fail_count

        case_name = case_meta["name"]
        V_cmd = case_meta["V_cmd"]
        tstop = compute_case_tstop(case_meta, time_scale)

        def execute_with_retries(mode_name, case_tag, mode_kwargs):
            attempts = [(DTMAX_SIM, 180)]
            if RETRY_ON_INCOMPLETE:
                for dt_factor, timeout_try in zip(RETRY_DTMAX_FACTORS, RETRY_TIMEOUTS):
                    attempts.append((DTMAX_SIM * float(dt_factor), int(timeout_try)))

            last_incomplete = None
            for attempt_idx, (dtmax_try, timeout_try) in enumerate(attempts, start=1):
                ret = run_case_once(case_tag, V_cmd, tstop, mode_kwargs, dtmax_try, timeout_try)
                if ret is None:
                    if attempt_idx == len(attempts):
                        finalize_failed_case(
                            mode=mode_name,
                            case_tag=case_tag,
                            case_meta=case_meta,
                            tstop=tstop,
                            reason="ngspice_failed",
                            attempt_idx=attempt_idx,
                            log_path=None,
                            time_end_s=np.nan,
                        )
                    continue

                df_try, plot_dir_try, sim_dir_try, log_path_try = ret
                time_end_s = float(df_try["time"].iloc[-1]) if len(df_try) else np.nan
                if is_simulation_complete(df_try, tstop):
                    if attempt_idx > 1:
                        print(f"[RECOVERED] {case_tag} completed on retry #{attempt_idx} with dtmax={dtmax_try:g}")
                    return df_try, plot_dir_try, sim_dir_try

                last_incomplete = (attempt_idx, log_path_try, time_end_s)
                print(
                    f"[WARN] incomplete simulation: {case_tag} "
                    f"(attempt {attempt_idx}, time_end={time_end_s:.4g}s < tstop={tstop:.4g}s)"
                )

            if last_incomplete is not None:
                attempt_idx, log_path_try, time_end_s = last_incomplete
                finalize_failed_case(
                    mode=mode_name,
                    case_tag=case_tag,
                    case_meta=case_meta,
                    tstop=tstop,
                    reason="truncated_before_tstop",
                    attempt_idx=attempt_idx,
                    log_path=log_path_try,
                    time_end_s=time_end_s,
                )
            return None

        # ---------- NOLIMIT ----------
        if RUN_NOLIMIT_MODE:
            raw_tag = safe_tag(f"{case_name}_NOLIMIT_{cfg_tag}")
            case_tag = unique_case_tag(raw_tag)
            retN = execute_with_retries("NOLIMIT", case_tag, NOLIMIT)
            if retN is None:
                fail_count += 1
            else:
                sim_count += 1
                print(f"[PROGRESS] sim#{sim_count}/{total_planned} OK: {case_tag}")
                dfN, plot_dirN, sim_dirN = retN
                record_case_outputs(
                    mode="NOLIMIT",
                    case_tag=case_tag,
                    case_meta=case_meta,
                    df=dfN,
                    plot_dir=plot_dirN,
                    sim_dir=sim_dirN,
                    tstop=tstop,
                    icomp_scale=np.nan,
                    vcomp=NOLIMIT["vcomp"],
                    rhi=NOLIMIT["rhi"],
                )

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
                        retL = execute_with_retries("LIMIT", case_tag, LIMIT)
                        if retL is None:
                            fail_count += 1
                            continue

                        sim_count += 1
                        print(f"[PROGRESS] sim#{sim_count}/{total_planned} OK: {case_tag}")
                        dfL, plot_dirL, sim_dirL = retL
                        record_case_outputs(
                            mode="LIMIT",
                            case_tag=case_tag,
                            case_meta=case_meta,
                            df=dfL,
                            plot_dir=plot_dirL,
                            sim_dir=sim_dirL,
                            tstop=tstop,
                            icomp_scale=float(icomp_scale),
                            vcomp=float(vcomp),
                            rhi=float(rhi),
                        )

    # ------------------------------------------------------------
    # Run continuous linear-ramp cases
    # ------------------------------------------------------------
    for pts_per_ramp in ppr_list:
        for hold_pts in hold_list:
            cont_cases = build_continuous_cases(V_meas, pts_per_ramp=pts_per_ramp, hold_pts=hold_pts)
            for time_scale in time_scale_list:
                for case_meta in cont_cases:
                    cfg_tag = safe_tag(f"ppr{fmt_num(pts_per_ramp)}_hold{fmt_num(hold_pts)}_ts{fmt_num(time_scale)}")
                    run_case_modes(case_meta=case_meta, cfg_tag=cfg_tag, time_scale=time_scale)

    # ------------------------------------------------------------
    # Run discretized-input staircase cases
    # ------------------------------------------------------------
    if ENABLE_DISCRETE_LEVEL_SWEEP:
        for levels_per_ramp in LEVELS_PER_RAMP_LIST:
            disc_cases = build_discrete_cases(levels_per_ramp, HOLD_PTS_PER_LEVEL)
            for time_scale in time_scale_list:
                for case_meta in disc_cases:
                    cfg_tag = safe_tag(f"lpr{fmt_num(levels_per_ramp)}_hpl{fmt_num(HOLD_PTS_PER_LEVEL)}_ts{fmt_num(time_scale)}")
                    run_case_modes(case_meta=case_meta, cfg_tag=cfg_tag, time_scale=time_scale)

    # ------------------------------------------------------------
    # Overlay plots (overall / grouped / subgroup-specific)
    # ------------------------------------------------------------
    for mode in ["LIMIT", "NOLIMIT"]:
        if overlay_bucket[mode]:
            add_overlay_bundle(
                overlay_bucket[mode],
                mode=mode,
                stem="all_selected",
                title_suffix="selected cases",
                manifest_rows=overlay_manifest_rows,
            )

    # Base group overlays
    for mode, group_dict in group_overlay_bucket.items():
        for group_name, sim_dict in group_dict.items():
            add_overlay_bundle(
                sim_dict,
                mode=mode,
                stem=group_name,
                title_suffix=group_name,
                manifest_rows=overlay_manifest_rows,
            )

    # Specialized subgroup overlays so every analysis view is available
    for mode in ["LIMIT", "NOLIMIT"]:
        reg = sim_registry[mode]

        # Multi-neg: compare the two sequence orders for each vpos
        for vpos in MULTI_NEG_VPOS_LIST:
            sim_dict = get_filtered_overlay_cases(
                reg,
                lambda m, v=float(vpos): m.get("group") == "multi_neg" and float(m.get("vpos", np.nan)) == v,
                sort_key=lambda m: m.get("name", ""),
            )
            add_overlay_bundle(
                sim_dict,
                mode=mode,
                stem=f"multi_neg_vpos{fmt_num(vpos)}_seq_compare",
                title_suffix=f"multi_neg, vpos={fmt_num(vpos)} V, sequence-order comparison",
                manifest_rows=overlay_manifest_rows,
            )

        # Multi-neg: compare vpos = 6/8/10 for each sequence order separately
        for seq in MULTI_NEG_SEQ_LIST:
            seq_tag = "_".join(fmt_num(v) for v in seq)
            sim_dict = get_filtered_overlay_cases(
                reg,
                lambda m, st=seq_tag: m.get("group") == "multi_neg" and m.get("name", "").endswith(st),
                sort_key=lambda m: m.get("vpos", np.nan),
            )
            add_overlay_bundle(
                sim_dict,
                mode=mode,
                stem=f"multi_neg_seq_{seq_tag}_vpos_compare",
                title_suffix=f"multi_neg, seq={seq_tag}, vpos comparison",
                manifest_rows=overlay_manifest_rows,
            )

        # Multi-neg: thesis representative pair at vpos = 10 V
        seq_cases = get_named_overlay_cases(
            reg,
            [
                "multi_vpos10_seq_-2_-4_-6_-8_-10",
                "multi_vpos10_seq_-10_-8_-6_-4_-2",
            ],
        )
        add_overlay_bundle(
            seq_cases,
            mode=mode,
            stem="multi_neg_vpos10_seq_only",
            title_suffix="multi_neg representative cases (vpos=10 V)",
            manifest_rows=overlay_manifest_rows,
        )

        # Repeat-neg: all repeats already grouped, but also create x1/x2/x3/x5 waveform-specific compare (same as group)
        repeat_cases = get_filtered_overlay_cases(
            reg,
            lambda m: m.get("group") == "repeat_neg",
            sort_key=lambda m: m.get("repeat_count", np.nan),
        )
        add_overlay_bundle(
            repeat_cases,
            mode=mode,
            stem="repeat_neg_all_repeats",
            title_suffix="repeat_neg, all repeat counts",
            manifest_rows=overlay_manifest_rows,
        )

        # Amplitude groups: explicit stems for thesis convenience
        amp_vpos_cases = get_filtered_overlay_cases(
            reg,
            lambda m: (
                m.get("group") == "amp_vpos"
                or (
                    m.get("group") == "baseline_fixed"
                    and float(m.get("vneg", np.nan)) == float(FIXED_VNEG_FOR_VPOS_SWEEP)
                )
            ),
            sort_key=lambda m: m.get("vpos", np.nan),
        )
        add_overlay_bundle(
            amp_vpos_cases,
            mode=mode,
            stem="amp_vpos_all",
            title_suffix="amp_vpos, all cases",
            manifest_rows=overlay_manifest_rows,
        )

        amp_vneg_cases = get_filtered_overlay_cases(
            reg,
            lambda m: (
                m.get("group") == "amp_vneg"
                or (
                    m.get("group") == "baseline_fixed"
                    and float(m.get("vpos", np.nan)) == float(FIXED_VPOS_FOR_VNEG_SWEEP)
                )
            ),
            sort_key=lambda m: m.get("vneg", np.nan),
        )
        add_overlay_bundle(
            amp_vneg_cases,
            mode=mode,
            stem="amp_vneg_all",
            title_suffix="amp_vneg, all cases",
            manifest_rows=overlay_manifest_rows,
        )
    # ------------------------------------------------------------
    # Measured vs simulated baseline compare
    # ------------------------------------------------------------
    for mode in ["LIMIT", "NOLIMIT"]:
        baseline_key = None
        for case_tag, rec in sim_registry[mode].items():
            if rec["case_meta"]["group"] == "baseline_meas":
                baseline_key = case_tag
                break
        if baseline_key is not None:
            rec = sim_registry[mode][baseline_key]
            plot_compare_meas_vs_sim_abs(
                V_meas,
                I_meas,
                rec["df"],
                OVERLAY_DIR / f"compare_meas_trace_vs_sim_{mode}.png",
                f"Measured vs simulated ({mode}) - measurement-driven waveform",
                use_vp=True,
            )

    # ------------------------------------------------------------
    # Baseline fixed vs amplitude-case compare (very useful for thesis 4.5)
    # ------------------------------------------------------------
    for mode in ["LIMIT", "NOLIMIT"]:
        baseline_fixed_tag = None
        for case_tag, rec in sim_registry[mode].items():
            if rec["case_meta"]["group"] == "baseline_fixed":
                baseline_fixed_tag = case_tag
                break

        if baseline_fixed_tag is None:
            continue

        base_df = sim_registry[mode][baseline_fixed_tag]["df"]
        base_label = sim_registry[mode][baseline_fixed_tag]["case_meta"]["name"]

        for case_tag, rec in sim_registry[mode].items():
            group = rec["case_meta"]["group"]
            if group not in {"amp_vpos", "amp_vneg", "amp_grid", "repeat_neg", "multi_neg"}:
                continue
            out_png = OVERLAY_DIR / safe_tag(f"compare_{mode}_{base_label}_vs_{rec['case_meta']['name']}.png")
            plot_compare_two_sim_abs(
                df_ref=base_df,
                df_other=rec["df"],
                out_png=out_png,
                title=f"Baseline vs amplitude case ({mode})",
                label_ref=base_label,
                label_other=rec["case_meta"]["name"],
                use_vp=True,
            )

    # ------------------------------------------------------------
    # Save CSV tables
    # ------------------------------------------------------------
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        settings_df = pd.DataFrame(case_meta_rows)
        branch_df = pd.DataFrame(branch_table_rows)
        segment_df = pd.DataFrame(segment_table_rows)

        summary_df.to_csv(OUT_DIR / "summary_all_cases.csv", index=False)
        settings_df.drop_duplicates().to_csv(OUT_DIR / "case_settings_table.csv", index=False)
        branch_df.to_csv(OUT_DIR / "branch_metrics_table.csv", index=False)
        segment_df.to_csv(OUT_DIR / "segment_metrics_table.csv", index=False)
        if SAVE_FAILED_CASES and failed_rows:
            pd.DataFrame(failed_rows).to_csv(OUT_DIR / "failed_cases.csv", index=False)
        if overlay_manifest_rows:
            pd.DataFrame(overlay_manifest_rows).to_csv(OUT_DIR / "overlay_manifest.csv", index=False)

        thesis_cols = [
            "case_tag", "case", "mode", "group", "input_style",
            "vpos", "vneg", "time_scale", "tstop",
            "Vp_min", "Vp_max", "Imax_abs", "I_pos_max", "I_neg_min",
            "I_zero_abs_max", "I_zero_abs_mean",
            "x_min", "x_max", "x_range", "xh_min", "xh_max", "xh_range", "xh_end",
            "loop_area_abs_idv",
        ]
        for vtar in BRANCH_SAMPLE_V_LIST:
            tag = str(vtar).replace(".", "p")
            thesis_cols.extend([
                f"branch_sep_pos_absI_at_{tag}V",
                f"branch_sep_pos_logdec_at_{tag}V",
                f"branch_sep_neg_absI_at_{tag}V",
                f"branch_sep_neg_logdec_at_{tag}V",
            ])
        thesis_cols = [c for c in thesis_cols if c in summary_df.columns]
        summary_df[thesis_cols].to_csv(OUT_DIR / "thesis_metrics_table.csv", index=False)

    print("\n[OK] Outputs written to:", OUT_DIR)
    print(" - cases/: each case_tag has decks/logs/sims/plots")
    print(" - overlay/: overall overlays + grouped overlays + subgroup-specific overlays")
    print(" - summary_all_cases.csv : all metrics")
    print(" - thesis_metrics_table.csv : direct-use thesis metrics")
    print(" - case_settings_table.csv : case configuration table")
    print(" - branch_metrics_table.csv : branch separation table")
    print(" - segment_metrics_table.csv : per-ramp metrics (useful for repeated negative sweeps)")
    if SAVE_FAILED_CASES:
        print(" - failed_cases.csv : failed / truncated runs (excluded from overlays)")
    print(" - overlay_manifest.csv : list of every generated overlay figure")
    print(f"\n[STATS] successful simulations = {sim_count}")
    print(f"[STATS] failed simulations     = {fail_count}")
    print("\nDefault thesis-oriented choices:")
    print(" - ENABLE_AMPLITUDE_SWEEP = True")
    print(" - Overlay x-axis uses Vp consistently")
    print(" - Baseline fixed case = 0 -> +10 -> 0 -> -4 -> 0 V")
    print(" - VPOS sweep = [2, 3, 4, 5, 6, 8, 10] with VNEG fixed at -4 V")
    print(" - VNEG sweep = [-2, -3, -4, -5, -6, -8, -10] with VPOS fixed at +10 V")
    print(" - Multi-negative sequences enabled (progressive and reverse)")
    print(" - Repeated negative cycles enabled: vneg=-10 V, repeat=[1,2,3,5]")
    print(f" - tstop normalization by ramp count = {NORMALIZE_TSTOP_BY_RAMP_COUNT} (baseline ramp count = {BASELINE_RAMP_COUNT})")
    print(f" - incomplete-run check = time_end >= {INCOMPLETE_TIME_FRAC:.3f} * tstop")
    print(" - Want staircase comparison too? set ENABLE_DISCRETE_LEVEL_SWEEP = True")


if __name__ == "__main__":
    main()
