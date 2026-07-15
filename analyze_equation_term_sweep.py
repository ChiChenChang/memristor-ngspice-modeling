"""
analyze_equation_term_sweep.py
================================
Equation-term sensitivity analysis for the fitted NGSpice memristive model.

This script keeps the external voltage waveform fixed and changes selected
parameter groups that control eight equation-term families:

1. I0_TERM
2. ACOEF_TERM
3. IDIFF_TERM
4. SET_TAU_TERM
5. RESET_TAU_TERM
6. RH_TERM
7. MEMORY_TERM
8. CURRENT_SCALE_LEAK_TERM

Expected project layout
-----------------------
project_root/
├── analyze_equation_term_sweep.py
├── data/
│   └── DC-IV.csv                         # optional in NOLIMIT mode
└── results/
    └── fit/
        ├── fitdeck_embedded.cir
        └── theta_best.csv

The SPICE template must use the same placeholders as the fitting workflow,
including @PWL_INLINE@, @SIMOUT@, @TSTEP@, @DTMAX@, @TSTOP@, model knobs,
compliance knobs, and fitted parameter names.

Main outputs
------------
results/equation_term_sweep/
├── fitdeck_embedded.cir
├── cases/<case_tag>/{decks,logs,sims,plots}/...
├── overlays/*.png
└── reports/
    ├── term_family_map.csv
    ├── case_manifest.csv
    ├── result_metrics.csv
    ├── term_trace_metrics.csv
    ├── family_summary.csv
    └── failed_cases.csv                  # only when failures occur
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_MEAS = BASE_DIR / "data" / "DC-IV.csv"
TEMPLATE_SRC = BASE_DIR / "results" / "fit" / "fitdeck_embedded.cir"
THETA_PATH = BASE_DIR / "results" / "fit" / "theta_best.csv"

_ngspice_on_path = shutil.which("ngspice")
NGSPICE = Path(_ngspice_on_path) if _ngspice_on_path else BASE_DIR / "ngspice.exe"

OUT_DIR = BASE_DIR / "results" / "equation_term_sweep"
TEMPLATE_SNAPSHOT = OUT_DIR / "fitdeck_embedded.cir"
CASES_DIR = OUT_DIR / "cases"
OVERLAY_DIR = OUT_DIR / "overlays"
REPORT_DIR = OUT_DIR / "reports"


# =============================================================================
# Simulation and numerical settings
# =============================================================================

TSTOP_BASE = 10.0
TIME_SCALE = 1.0
DTMAX_SIM = 2e-4

PRINT_DIV = 6
PRINT_MIN = 2e-5
PRINT_MAX = 4e-3

NGSPICE_TIMEOUT_S = 180
RETRY_ON_FAILURE = True
RETRY_DTMAX_FACTORS = [0.5, 0.25]
RETRY_TIMEOUTS_S = [240, 360]
INCOMPLETE_TIME_FRAC = 0.995

I_FLOOR_ABS = 3e-10
SYMLINTHRESH = 1e-9
MAX_PLOT_POINTS = 24000
ZERO_V_WINDOW_MIN = 0.05


# =============================================================================
# Fixed voltage waveform
# =============================================================================

FIXED_BASELINE_VPOS = 10.0
FIXED_BASELINE_VNEG = -4.0
PTS_PER_RAMP = 240
HOLD_PTS = 0


# =============================================================================
# Fixed model knobs
# =============================================================================

KSW_FIXED = 3.0
RH0_FIXED = 1e3
RH_MIN_FIXED = 1.0
RH_MAX_FIXED = 1e7
VSLOPE_FIXED = 0.5


# =============================================================================
# Compliance settings
# =============================================================================

RUN_NOLIMIT_MODE = True
RUN_LIMIT_MODE = False

RLO_FIXED = 1.0
RHI_FIXED = 2e8
VCOMP_FIXED = 0.0
VSLOPE_POS_FIXED = 0.02
ISLOPE_REL = 0.02


# =============================================================================
# Parameters expected in theta_best.csv
# =============================================================================

REQUIRED_PARAMS = {
    "IMAX",
    "IMIN",
    "ALPHA_MAX",
    "ALPHA_MIN",
    "BETAA",
    "VSET",
    "VRES",
    "ETA_SET",
    "ETA_RES",
    "CH0",
    "ISCALE",
    "H0",
    "EI",
    "ROFF",
}

ALIASES = {"BETA": "BETAA"}


# =============================================================================
# Equation-term sweep configuration
# =============================================================================

TERM_SWEEP_CONFIG: dict[str, list[dict[str, Any]]] = {
    "I0_TERM": [
        {"mode": "member_IMAX_scale", "values": [0.5, 0.75, 1.0, 1.5, 2.0]},
        {"mode": "member_IMIN_scale", "values": [0.5, 0.75, 1.0, 1.5, 2.0]},
        {"mode": "common_scale", "values": [0.5, 0.75, 1.0, 1.5, 2.0]},
        {"mode": "contrast_scale", "values": [0.0, 0.5, 1.0, 1.5, 2.0]},
    ],
    "ACOEF_TERM": [
        {"mode": "member_ALPHA_MAX_scale", "values": [0.5, 0.75, 1.0, 1.5, 2.0]},
        {"mode": "member_ALPHA_MIN_scale", "values": [0.5, 0.75, 1.0, 1.5, 2.0]},
        {"mode": "common_scale", "values": [0.5, 0.75, 1.0, 1.5, 2.0]},
        {"mode": "contrast_scale", "values": [0.0, 0.5, 1.0, 1.5, 2.0]},
    ],
    "IDIFF_TERM": [
        {"mode": "beta_shift", "values": [-0.15, -0.08, 0.0, 0.08, 0.15]},
    ],
    "SET_TAU_TERM": [
        {"mode": "vset_shift", "values": [-0.8, -0.4, 0.0, 0.4, 0.8]},
        {"mode": "eta_set_scale", "values": [0.5, 0.75, 1.0, 1.5, 2.0]},
    ],
    "RESET_TAU_TERM": [
        {"mode": "vres_shift", "values": [-0.6, -0.3, 0.0, 0.3, 0.6]},
        {"mode": "eta_res_scale", "values": [0.5, 0.75, 1.0, 1.5, 2.0]},
    ],
    "RH_TERM": [
        {"mode": "rh0_scale", "values": [0.25, 0.5, 1.0, 2.0, 4.0]},
        {"mode": "vslope_scale", "values": [0.5, 1.0, 2.0]},
    ],
    "MEMORY_TERM": [
        {"mode": "ch0_scale", "values": [0.25, 0.5, 1.0, 2.0, 4.0]},
        {"mode": "h0_shift", "values": [-0.20, -0.10, 0.0, 0.10, 0.20]},
    ],
    "CURRENT_SCALE_LEAK_TERM": [
        {"mode": "iscale_scale", "values": [0.25, 0.5, 1.0, 2.0, 4.0]},
        {"mode": "ei_scale", "values": [0.25, 0.5, 1.0, 2.0, 4.0]},
        {"mode": "roff_scale", "values": [0.25, 0.5, 1.0, 2.0, 4.0]},
    ],
}

TERM_FAMILY_INFO: dict[str, dict[str, Any]] = {
    "I0_TERM": {
        "equation": "I0(h) = IMAX*h + IMIN*(1-h)",
        "member_params": ["IMAX", "IMIN"],
        "term_cols": ["I0_eff"],
        "result_focus": ["I", "xh", "x"],
    },
    "ACOEF_TERM": {
        "equation": "Acoef(h) = ALPHA_MAX*h + ALPHA_MIN*(1-h)",
        "member_params": ["ALPHA_MAX", "ALPHA_MIN"],
        "term_cols": ["Acoef_eff", "Idiff_eff"],
        "result_focus": ["I", "xh", "x"],
    },
    "IDIFF_TERM": {
        "equation": "Idiff(v,h) = exp(BETA*A*v) - exp(-(1-BETA)*A*v)",
        "member_params": ["BETAA"],
        "term_cols": ["Idiff_eff", "Idiff_pos_exp", "Idiff_neg_exp"],
        "result_focus": ["I", "xh", "x"],
    },
    "SET_TAU_TERM": {
        "equation": "Stau(v) = RH0*exp(-ETA_SET*(v-VSET))",
        "member_params": ["ETA_SET", "VSET"],
        "term_cols": ["Stau_eff", "Rh_eff", "tau_x_eff"],
        "result_focus": ["x", "xh", "I"],
    },
    "RESET_TAU_TERM": {
        "equation": "Rtau(v) = RH0*exp(ETA_RES*(v+VRES))",
        "member_params": ["ETA_RES", "VRES"],
        "term_cols": ["Rtau_eff", "Rh_eff", "tau_x_eff"],
        "result_focus": ["x", "xh", "I"],
    },
    "RH_TERM": {
        "equation": "Rh(v) = clip(A(v)*Stau(v) + (1-A(v))*Rtau(v))",
        "member_params": ["RH0", "VSLOPE"],
        "term_cols": ["A_switch", "Rh_eff", "tau_x_eff"],
        "result_focus": ["x", "xh", "I"],
    },
    "MEMORY_TERM": {
        "equation": "tau_x(v) = Rh(v)*CH0; initial state set by H0",
        "member_params": ["CH0", "H0"],
        "term_cols": ["Rh_eff", "tau_x_eff", "x", "xh"],
        "result_focus": ["x", "xh", "I"],
    },
    "CURRENT_SCALE_LEAK_TERM": {
        "equation": "I = ISCALE*(I0*Idiff + EI*v) + v/ROFF",
        "member_params": ["ISCALE", "EI", "ROFF"],
        "term_cols": ["I_main_est", "I_ei_est", "I_roff_est", "I_model_est"],
        "result_focus": ["I", "xh", "x"],
    },
}

DISPLAY_KEY_MAP: dict[tuple[str, str], list[tuple[str, str]]] = {
    ("I0_TERM", "member_IMAX_scale"): [("theta", "IMAX")],
    ("I0_TERM", "member_IMIN_scale"): [("theta", "IMIN")],
    ("I0_TERM", "common_scale"): [("theta", "IMAX"), ("theta", "IMIN")],
    ("I0_TERM", "contrast_scale"): [("theta", "IMAX"), ("theta", "IMIN")],
    ("ACOEF_TERM", "member_ALPHA_MAX_scale"): [("theta", "ALPHA_MAX")],
    ("ACOEF_TERM", "member_ALPHA_MIN_scale"): [("theta", "ALPHA_MIN")],
    ("ACOEF_TERM", "common_scale"): [("theta", "ALPHA_MAX"), ("theta", "ALPHA_MIN")],
    ("ACOEF_TERM", "contrast_scale"): [("theta", "ALPHA_MAX"), ("theta", "ALPHA_MIN")],
    ("IDIFF_TERM", "beta_shift"): [("theta", "BETAA")],
    ("SET_TAU_TERM", "vset_shift"): [("theta", "VSET")],
    ("SET_TAU_TERM", "eta_set_scale"): [("theta", "ETA_SET")],
    ("RESET_TAU_TERM", "vres_shift"): [("theta", "VRES")],
    ("RESET_TAU_TERM", "eta_res_scale"): [("theta", "ETA_RES")],
    ("RH_TERM", "rh0_scale"): [("knob", "RH0")],
    ("RH_TERM", "vslope_scale"): [("knob", "VSLOPE")],
    ("MEMORY_TERM", "ch0_scale"): [("theta", "CH0")],
    ("MEMORY_TERM", "h0_shift"): [("theta", "H0")],
    ("CURRENT_SCALE_LEAK_TERM", "iscale_scale"): [("theta", "ISCALE")],
    ("CURRENT_SCALE_LEAK_TERM", "ei_scale"): [("theta", "EI")],
    ("CURRENT_SCALE_LEAK_TERM", "roff_scale"): [("theta", "ROFF")],
}


# =============================================================================
# General utilities
# =============================================================================

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE_SRC.exists():
        raise FileNotFoundError(f"Missing SPICE template: {TEMPLATE_SRC}")

    TEMPLATE_SNAPSHOT.write_text(
        TEMPLATE_SRC.read_text(encoding="utf-8", errors="ignore"),
        encoding="utf-8",
    )


def safe_tag(text: str) -> str:
    text = str(text).strip().replace(" ", "_")
    text = text.replace("/", "_").replace("\\", "_").replace(":", "_")
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def make_case_dirs(case_tag: str):
    case_dir = CASES_DIR / case_tag
    deck_dir = case_dir / "decks"
    log_dir = case_dir / "logs"
    sim_dir = case_dir / "sims"
    plot_dir = case_dir / "plots"

    for directory in (deck_dir, log_dir, sim_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return case_dir, deck_dir, log_dir, sim_dir, plot_dir


def read_meas_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, engine="python")
    if df.shape[1] < 2:
        raise ValueError("DC-IV.csv must contain at least two columns: V and I")

    v = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    i = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    valid = v.notna() & i.notna()

    v_out = v[valid].to_numpy(float)
    i_out = i[valid].to_numpy(float)

    if len(v_out) < 10:
        raise ValueError("Too few valid numeric rows in DC-IV.csv")

    return v_out, i_out


def read_theta_best(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if not {"param", "value"}.issubset(df.columns):
        raise ValueError("theta_best.csv must contain columns named param and value")

    theta: dict[str, float] = {}
    for param, value in zip(df["param"], df["value"]):
        key = str(param).strip()
        key = ALIASES.get(key, key)
        theta[key] = float(value)

    missing = sorted(REQUIRED_PARAMS.difference(theta))
    if missing:
        raise ValueError(f"theta_best.csv is missing required parameters: {missing}")

    return theta


def estimate_icomp_pos(v: np.ndarray, i: np.ndarray) -> float:
    mask = np.asarray(v, float) > 0.5
    if not np.any(mask):
        return 1e-3
    estimate = float(np.quantile(np.abs(np.asarray(i, float)[mask]), 0.98))
    return float(np.clip(estimate, 1e-6, 5e-2))


def make_time_vector(n_points: int, tstop: float) -> np.ndarray:
    if n_points < 2:
        return np.array([0.0])
    return np.linspace(0.0, float(tstop), int(n_points))


def pick_tstep_print(n_points: int, tstop: float) -> float:
    if n_points <= 1:
        return 1e-3
    dt_cmd = float(tstop) / float(n_points - 1)
    return float(np.clip(dt_cmd / PRINT_DIV, PRINT_MIN, PRINT_MAX))


def pwl_inline_from_tv(t: np.ndarray, v: np.ndarray, pairs_per_line: int = 8) -> str:
    pairs = [f"{ti:.12g} {vi:.12g}" for ti, vi in zip(t, v)]
    return "\n".join(
        "+ " + " ".join(pairs[index : index + pairs_per_line])
        for index in range(0, len(pairs), pairs_per_line)
    )


def run_ngspice(deck_path: Path, log_path: Path, cwd: Path, timeout_s: int) -> int:
    command = [str(NGSPICE), "-b", "-o", str(log_path), str(deck_path)]
    try:
        completed = subprocess.run(command, cwd=str(cwd), timeout=int(timeout_s))
        return int(completed.returncode)
    except subprocess.TimeoutExpired:
        return 124


def tail_text(path: Path, n_lines: int = 100) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except OSError:
        return ""


def load_wrdata(path: Path) -> tuple[np.ndarray, ...]:
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise ValueError(f"Expected at least six wrdata columns, got {data.shape[1]}: {path}")
    return tuple(data[:, index] for index in range(6))


def is_simulation_complete(df: pd.DataFrame, tstop: float) -> bool:
    if df.empty:
        return False
    return float(df["time"].iloc[-1]) >= float(INCOMPLETE_TIME_FRAC) * float(tstop)


def downsample_arrays(df: pd.DataFrame, columns: list[str]) -> list[np.ndarray]:
    n_rows = len(df)
    if n_rows <= MAX_PLOT_POINTS:
        return [df[column].to_numpy(float) for column in columns]
    indices = np.linspace(0, n_rows - 1, MAX_PLOT_POINTS).astype(int)
    return [df[column].to_numpy(float)[indices] for column in columns]


def safe_exp_np(value: Any, clip: float = 80.0) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    return np.exp(np.clip(array, -float(clip), float(clip)))


def sstep_np(v: Any, vslope: float) -> np.ndarray:
    slope = max(abs(float(vslope)), 1e-12)
    return 0.5 * (1.0 + np.tanh(np.asarray(v, dtype=float) / slope))


def clip_positive(value: float, minimum: float = 1e-30) -> float:
    return max(float(minimum), float(value))


def clip_unit(value: float) -> float:
    return float(np.clip(float(value), 1e-6, 0.999999))


def fmt_number(value: float, significant_digits: int = 5) -> str:
    value = float(value)
    magnitude = abs(value)
    if magnitude >= 1e4 or (0 < magnitude < 1e-3):
        return f"{value:.{significant_digits}e}"
    return f"{value:.{significant_digits}g}"


# =============================================================================
# Waveform and NGSpice simulation
# =============================================================================

def ramp(v0: float, v1: float, n_points: int) -> np.ndarray:
    return np.linspace(float(v0), float(v1), int(max(2, n_points)))


def make_voltage_sequence(
    nodes: Iterable[float],
    pts_per_ramp: int = PTS_PER_RAMP,
    hold_pts: int = HOLD_PTS,
) -> np.ndarray:
    node_list = [float(value) for value in nodes]
    if len(node_list) < 2:
        raise ValueError("nodes must contain at least two voltage values")

    repeat_count = max(1, int(hold_pts) + 1)
    segments: list[np.ndarray] = []

    for index, (start, stop) in enumerate(zip(node_list[:-1], node_list[1:])):
        segment = ramp(start, stop, pts_per_ramp)
        if index > 0:
            segment = segment[1:]
        segments.append(np.repeat(segment, repeat_count))

    return np.concatenate(segments)


def build_baseline_waveform() -> np.ndarray:
    nodes = [0.0, FIXED_BASELINE_VPOS, 0.0, FIXED_BASELINE_VNEG, 0.0]
    return make_voltage_sequence(nodes)


def baseline_model_knobs() -> dict[str, float]:
    return {
        "KSW": KSW_FIXED,
        "RH0": RH0_FIXED,
        "RH_MIN": RH_MIN_FIXED,
        "RH_MAX": RH_MAX_FIXED,
        "VSLOPE": VSLOPE_FIXED,
    }


def simulate_once(
    theta: dict[str, float],
    model_knobs: dict[str, float],
    v_cmd: np.ndarray,
    case_tag: str,
    tstop: float,
    run_settings: dict[str, float],
    dtmax: float,
    timeout_s: int,
):
    case_dir, deck_dir, log_dir, sim_dir, plot_dir = make_case_dirs(case_tag)

    time = make_time_vector(len(v_cmd), tstop)
    tstep_print = pick_tstep_print(len(v_cmd), tstop)

    deck_path = deck_dir / f"{case_tag}.cir"
    log_path = log_dir / f"{case_tag}.log"
    sim_path = sim_dir / f"{case_tag}.dat"

    template = TEMPLATE_SNAPSHOT.read_text(encoding="utf-8", errors="ignore")
    icomp_pos = float(run_settings["icomp_pos"])
    islope = max(1e-12, float(run_settings["islope_rel"]) * icomp_pos)

    replacements: dict[str, str] = {
        "@PWL_INLINE@": pwl_inline_from_tv(time, v_cmd),
        "@SIMOUT@": str(sim_path),
        "@TSTEP@": f"{tstep_print:.12g}",
        "@DTMAX@": f"{float(dtmax):.12g}",
        "@TSTOP@": f"{float(tstop):.12g}",
        "@KSW@": f"{float(model_knobs['KSW']):.12g}",
        "@RH0@": f"{float(model_knobs['RH0']):.12g}",
        "@RH_MIN@": f"{float(model_knobs['RH_MIN']):.12g}",
        "@RH_MAX@": f"{float(model_knobs['RH_MAX']):.12g}",
        "@VSLOPE@": f"{float(model_knobs['VSLOPE']):.12g}",
        "@ICOMP_POS@": f"{icomp_pos:.12g}",
        "@VCOMP@": f"{float(run_settings['vcomp']):.12g}",
        "@RLO@": f"{float(run_settings['rlo']):.12g}",
        "@RHI@": f"{float(run_settings['rhi']):.12g}",
        "@ISLOPE@": f"{islope:.12g}",
        "@VSLOPE_POS@": f"{float(run_settings['vslope_pos']):.12g}",
    }

    for key, value in theta.items():
        replacements[f"@{key}@"] = f"{float(value):.12g}"

    for token, replacement in replacements.items():
        template = template.replace(token, replacement)

    leftovers = sorted(set(re.findall(r"@[A-Za-z0-9_]+@", template)))
    if leftovers:
        raise RuntimeError(f"Unreplaced SPICE placeholders: {leftovers[:30]}")

    deck_path.write_text(template, encoding="utf-8")
    return_code = run_ngspice(deck_path, log_path, case_dir, timeout_s)

    if return_code != 0 or not sim_path.exists():
        return None, plot_dir, sim_dir, log_path, return_code

    time_out, vcmd, vp, current, x, xh = load_wrdata(sim_path)
    df = pd.DataFrame(
        {
            "time": time_out,
            "Vcmd": vcmd,
            "Vp": vp,
            "I": current,
            "x": x,
            "xh": xh,
        }
    )
    df["phase"] = df["time"] / max(float(df["time"].iloc[-1]), 1e-30)
    df.to_csv(sim_dir / f"{case_tag}_sim.csv", index=False)

    return df, plot_dir, sim_dir, log_path, return_code


def simulate_case(
    theta: dict[str, float],
    model_knobs: dict[str, float],
    v_cmd: np.ndarray,
    case_tag: str,
    tstop: float,
    run_settings: dict[str, float],
):
    attempts = [(DTMAX_SIM, NGSPICE_TIMEOUT_S)]
    if RETRY_ON_FAILURE:
        attempts.extend(
            (DTMAX_SIM * factor, timeout)
            for factor, timeout in zip(RETRY_DTMAX_FACTORS, RETRY_TIMEOUTS_S)
        )

    last_result = None
    for attempt_index, (dtmax, timeout_s) in enumerate(attempts, start=1):
        result = simulate_once(
            theta=theta,
            model_knobs=model_knobs,
            v_cmd=v_cmd,
            case_tag=case_tag,
            tstop=tstop,
            run_settings=run_settings,
            dtmax=dtmax,
            timeout_s=timeout_s,
        )
        last_result = result
        df = result[0]

        if df is not None and is_simulation_complete(df, tstop):
            return df, result[1], result[2]

        print(
            f"[retry] {case_tag}: attempt={attempt_index}, "
            f"dtmax={dtmax:.3g}, timeout={timeout_s}s"
        )

    assert last_result is not None
    df, _, _, log_path, return_code = last_result
    reason = "incomplete transient" if df is not None else f"ngspice return code {return_code}"
    raise RuntimeError(f"Simulation failed for {case_tag}: {reason}\n{tail_text(log_path)}")


# =============================================================================
# Equation-term calculations
# =============================================================================

def calc_term_traces(
    df: pd.DataFrame,
    theta: dict[str, float],
    model_knobs: dict[str, float],
) -> pd.DataFrame:
    v = df["Vp"].to_numpy(float)
    x = df["x"].to_numpy(float)
    h = np.clip(df["xh"].to_numpy(float), 0.0, 1.0)

    imax = float(theta["IMAX"])
    imin = float(theta["IMIN"])
    alpha_max = float(theta["ALPHA_MAX"])
    alpha_min = float(theta["ALPHA_MIN"])
    beta = float(theta["BETAA"])

    vset = float(theta["VSET"])
    vres = float(theta["VRES"])
    eta_set = float(theta["ETA_SET"])
    eta_res = float(theta["ETA_RES"])
    ch0 = float(theta["CH0"])

    iscale = float(theta["ISCALE"])
    ei = float(theta["EI"])
    roff = float(theta["ROFF"])

    rh0 = float(model_knobs["RH0"])
    rh_min = float(model_knobs["RH_MIN"])
    rh_max = float(model_knobs["RH_MAX"])
    vslope = float(model_knobs["VSLOPE"])

    i0_eff = imax * h + imin * (1.0 - h)
    acoef_eff = alpha_max * h + alpha_min * (1.0 - h)

    idiff_pos_exp = safe_exp_np(beta * acoef_eff * v)
    idiff_neg_exp = safe_exp_np(-(1.0 - beta) * acoef_eff * v)
    idiff_eff = idiff_pos_exp - idiff_neg_exp

    stau_eff = rh0 * safe_exp_np(-eta_set * (v - vset), clip=25.0)
    rtau_eff = rh0 * safe_exp_np(eta_res * (v + vres), clip=25.0)
    a_switch = sstep_np(v, vslope)

    rh_eff = a_switch * stau_eff + (1.0 - a_switch) * rtau_eff
    rh_eff = np.clip(rh_eff, rh_min, rh_max)
    tau_x_eff = rh_eff * ch0

    i_main_est = iscale * i0_eff * idiff_eff
    i_ei_est = iscale * ei * v
    i_roff_est = v / roff if abs(roff) > 1e-30 else np.full_like(v, np.nan)
    i_model_est = i_main_est + i_ei_est + i_roff_est

    return pd.DataFrame(
        {
            "time": df["time"].to_numpy(float),
            "phase": df["phase"].to_numpy(float),
            "Vcmd": df["Vcmd"].to_numpy(float),
            "Vp": v,
            "I": df["I"].to_numpy(float),
            "x": x,
            "xh": h,
            "A_switch": a_switch,
            "I0_eff": i0_eff,
            "Acoef_eff": acoef_eff,
            "Idiff_eff": idiff_eff,
            "Idiff_pos_exp": idiff_pos_exp,
            "Idiff_neg_exp": idiff_neg_exp,
            "Stau_eff": stau_eff,
            "Rtau_eff": rtau_eff,
            "Rh_eff": rh_eff,
            "tau_x_eff": tau_x_eff,
            "I_main_est": i_main_est,
            "I_ei_est": i_ei_est,
            "I_roff_est": i_roff_est,
            "I_model_est": i_model_est,
        }
    )


def calc_static_term_curves(
    theta: dict[str, float],
    model_knobs: dict[str, float],
    vmin: float,
    vmax: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h_grid = np.linspace(0.0, 1.0, 201)
    v_grid = np.linspace(float(vmin), float(vmax), 401)

    i0_of_h = float(theta["IMAX"]) * h_grid + float(theta["IMIN"]) * (1.0 - h_grid)
    acoef_of_h = (
        float(theta["ALPHA_MAX"]) * h_grid
        + float(theta["ALPHA_MIN"]) * (1.0 - h_grid)
    )

    rh0 = float(model_knobs["RH0"])
    rh_min = float(model_knobs["RH_MIN"])
    rh_max = float(model_knobs["RH_MAX"])
    vslope = float(model_knobs["VSLOPE"])

    a_switch = sstep_np(v_grid, vslope)
    stau = rh0 * safe_exp_np(
        -float(theta["ETA_SET"]) * (v_grid - float(theta["VSET"])),
        clip=25.0,
    )
    rtau = rh0 * safe_exp_np(
        float(theta["ETA_RES"]) * (v_grid + float(theta["VRES"])),
        clip=25.0,
    )
    rh = np.clip(a_switch * stau + (1.0 - a_switch) * rtau, rh_min, rh_max)
    tau_x = rh * float(theta["CH0"])

    h_df = pd.DataFrame(
        {
            "h": h_grid,
            "I0_of_h": i0_of_h,
            "Acoef_of_h": acoef_of_h,
        }
    )

    v_df = pd.DataFrame(
        {
            "V": v_grid,
            "A_switch_of_v": a_switch,
            "Stau_of_v": stau,
            "Rtau_of_v": rtau,
            "Rh_of_v": rh,
            "tau_x_of_v": tau_x,
        }
    )

    idiff_rows: list[dict[str, float]] = []
    beta = float(theta["BETAA"])
    for h_value in (0.0, 0.25, 0.5, 0.75, 1.0):
        acoef = (
            float(theta["ALPHA_MAX"]) * h_value
            + float(theta["ALPHA_MIN"]) * (1.0 - h_value)
        )
        idiff = safe_exp_np(beta * acoef * v_grid) - safe_exp_np(
            -(1.0 - beta) * acoef * v_grid
        )
        for voltage, value in zip(v_grid, idiff):
            idiff_rows.append({"h": h_value, "V": voltage, "Idiff": value})

    return h_df, v_df, pd.DataFrame(idiff_rows)


# =============================================================================
# Sweep application and labels
# =============================================================================

def apply_term_sweep(
    theta_base: dict[str, float],
    knobs_base: dict[str, float],
    family: str,
    mode: str,
    value: float,
) -> tuple[dict[str, float], dict[str, float]]:
    theta = dict(theta_base)
    knobs = dict(knobs_base)
    value = float(value)

    if family == "I0_TERM":
        if mode == "member_IMAX_scale":
            theta["IMAX"] = clip_positive(theta_base["IMAX"] * value)
        elif mode == "member_IMIN_scale":
            theta["IMIN"] = clip_positive(theta_base["IMIN"] * value)
        elif mode == "common_scale":
            theta["IMAX"] = clip_positive(theta_base["IMAX"] * value)
            theta["IMIN"] = clip_positive(theta_base["IMIN"] * value)
        elif mode == "contrast_scale":
            center = 0.5 * (theta_base["IMAX"] + theta_base["IMIN"])
            half_difference = 0.5 * (theta_base["IMAX"] - theta_base["IMIN"])
            theta["IMAX"] = clip_positive(center + value * half_difference)
            theta["IMIN"] = clip_positive(center - value * half_difference)
        else:
            raise ValueError(f"Unsupported mode: {family}/{mode}")

    elif family == "ACOEF_TERM":
        if mode == "member_ALPHA_MAX_scale":
            theta["ALPHA_MAX"] = clip_positive(theta_base["ALPHA_MAX"] * value)
        elif mode == "member_ALPHA_MIN_scale":
            theta["ALPHA_MIN"] = clip_positive(theta_base["ALPHA_MIN"] * value)
        elif mode == "common_scale":
            theta["ALPHA_MAX"] = clip_positive(theta_base["ALPHA_MAX"] * value)
            theta["ALPHA_MIN"] = clip_positive(theta_base["ALPHA_MIN"] * value)
        elif mode == "contrast_scale":
            center = 0.5 * (theta_base["ALPHA_MAX"] + theta_base["ALPHA_MIN"])
            half_difference = 0.5 * (
                theta_base["ALPHA_MAX"] - theta_base["ALPHA_MIN"]
            )
            theta["ALPHA_MAX"] = clip_positive(center + value * half_difference)
            theta["ALPHA_MIN"] = clip_positive(center - value * half_difference)
        else:
            raise ValueError(f"Unsupported mode: {family}/{mode}")

    elif family == "IDIFF_TERM":
        if mode != "beta_shift":
            raise ValueError(f"Unsupported mode: {family}/{mode}")
        theta["BETAA"] = clip_unit(theta_base["BETAA"] + value)

    elif family == "SET_TAU_TERM":
        if mode == "vset_shift":
            theta["VSET"] = theta_base["VSET"] + value
        elif mode == "eta_set_scale":
            theta["ETA_SET"] = clip_positive(theta_base["ETA_SET"] * value)
        else:
            raise ValueError(f"Unsupported mode: {family}/{mode}")

    elif family == "RESET_TAU_TERM":
        if mode == "vres_shift":
            theta["VRES"] = theta_base["VRES"] + value
        elif mode == "eta_res_scale":
            theta["ETA_RES"] = clip_positive(theta_base["ETA_RES"] * value)
        else:
            raise ValueError(f"Unsupported mode: {family}/{mode}")

    elif family == "RH_TERM":
        if mode == "rh0_scale":
            knobs["RH0"] = clip_positive(knobs_base["RH0"] * value)
        elif mode == "vslope_scale":
            knobs["VSLOPE"] = clip_positive(knobs_base["VSLOPE"] * value)
        else:
            raise ValueError(f"Unsupported mode: {family}/{mode}")

    elif family == "MEMORY_TERM":
        if mode == "ch0_scale":
            theta["CH0"] = clip_positive(theta_base["CH0"] * value)
        elif mode == "h0_shift":
            theta["H0"] = clip_unit(theta_base["H0"] + value)
        else:
            raise ValueError(f"Unsupported mode: {family}/{mode}")

    elif family == "CURRENT_SCALE_LEAK_TERM":
        if mode == "iscale_scale":
            theta["ISCALE"] = clip_positive(theta_base["ISCALE"] * value)
        elif mode == "ei_scale":
            theta["EI"] = theta_base["EI"] * value
        elif mode == "roff_scale":
            theta["ROFF"] = clip_positive(theta_base["ROFF"] * value)
        else:
            raise ValueError(f"Unsupported mode: {family}/{mode}")

    else:
        raise ValueError(f"Unknown term family: {family}")

    return theta, knobs


def build_case_label(
    family: str,
    mode: str,
    value: float,
    theta: dict[str, float],
    knobs: dict[str, float],
) -> str:
    prefix = f"{mode}={fmt_number(value)}"
    display_keys = DISPLAY_KEY_MAP.get((family, mode), [])
    if not display_keys:
        return prefix

    parts: list[str] = []
    for source, key in display_keys:
        actual_value = theta[key] if source == "theta" else knobs[key]
        parts.append(f"{key}={fmt_number(actual_value)}")

    return prefix + " | " + ", ".join(parts)


# =============================================================================
# Metrics
# =============================================================================

def trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def calc_result_metrics(df: pd.DataFrame) -> dict[str, float]:
    v = df["Vp"].to_numpy(float)
    current = df["I"].to_numpy(float)
    x = df["x"].to_numpy(float)
    xh = np.clip(df["xh"].to_numpy(float), 0.0, 1.0)

    zero_window = max(ZERO_V_WINDOW_MIN, 0.02 * float(np.max(np.abs(v))))
    zero_mask = np.abs(v) <= zero_window
    positive_mask = v > 0
    negative_mask = v < 0

    return {
        "N": int(len(df)),
        "Vp_min": float(np.min(v)),
        "Vp_max": float(np.max(v)),
        "I_abs_max": float(np.max(np.abs(current))),
        "I_pos_peak": float(np.max(current[positive_mask])) if np.any(positive_mask) else np.nan,
        "I_neg_peak_abs": float(np.max(np.abs(current[negative_mask]))) if np.any(negative_mask) else np.nan,
        "I_zero_abs_median": float(np.median(np.abs(current[zero_mask]))) if np.any(zero_mask) else np.nan,
        "loop_area_signed_A_V": trapezoid_area(current, v),
        "loop_area_abs_path_A_V": float(
            np.sum(np.abs(0.5 * (current[:-1] + current[1:]) * np.diff(v)))
        ),
        "x_start": float(x[0]),
        "x_end": float(x[-1]),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "x_span": float(np.max(x) - np.min(x)),
        "x_recovery_abs": float(abs(x[-1] - x[0])),
        "xh_start": float(xh[0]),
        "xh_end": float(xh[-1]),
        "xh_min": float(np.min(xh)),
        "xh_max": float(np.max(xh)),
        "xh_span": float(np.max(xh) - np.min(xh)),
        "xh_recovery_abs": float(abs(xh[-1] - xh[0])),
    }


def calc_term_trace_metrics(term_df: pd.DataFrame) -> dict[str, float]:
    metric_columns = [
        "A_switch",
        "I0_eff",
        "Acoef_eff",
        "Idiff_eff",
        "Stau_eff",
        "Rtau_eff",
        "Rh_eff",
        "tau_x_eff",
        "I_main_est",
        "I_ei_est",
        "I_roff_est",
        "I_model_est",
    ]

    metrics: dict[str, float] = {}
    for column in metric_columns:
        values = term_df[column].to_numpy(float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            metrics[f"{column}_min"] = np.nan
            metrics[f"{column}_max"] = np.nan
            metrics[f"{column}_span"] = np.nan
            metrics[f"{column}_abs_max"] = np.nan
            continue
        metrics[f"{column}_min"] = float(np.min(finite))
        metrics[f"{column}_max"] = float(np.max(finite))
        metrics[f"{column}_span"] = float(np.max(finite) - np.min(finite))
        metrics[f"{column}_abs_max"] = float(np.max(np.abs(finite)))

    model_error = term_df["I"].to_numpy(float) - term_df["I_model_est"].to_numpy(float)
    finite_error = model_error[np.isfinite(model_error)]
    metrics["I_model_est_mae"] = (
        float(np.mean(np.abs(finite_error))) if len(finite_error) else np.nan
    )
    return metrics


# =============================================================================
# Plotting
# =============================================================================

def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_result_suite(df: pd.DataFrame, plot_dir: Path, title_prefix: str) -> None:
    import matplotlib.pyplot as plt

    v, current = downsample_arrays(df, ["Vp", "I"])
    fig, ax = plt.subplots()
    ax.set_yscale("symlog", linthresh=SYMLINTHRESH)
    ax.plot(v, current)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_xlabel("Vp (V)")
    ax.set_ylabel("I (A)")
    ax.set_title(f"{title_prefix}: I-V")
    save_figure(fig, plot_dir / "iv_symlog.png")

    fig, ax = plt.subplots()
    ax.semilogy(v, np.abs(current) + I_FLOOR_ABS)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_xlabel("Vp (V)")
    ax.set_ylabel("|I| (A)")
    ax.set_title(f"{title_prefix}: |I|-V")
    save_figure(fig, plot_dir / "iv_logabs.png")

    v, x, xh = downsample_arrays(df, ["Vp", "x", "xh"])
    fig, ax = plt.subplots()
    ax.plot(v, x, label="x")
    ax.plot(v, xh, label="xh")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("Vp (V)")
    ax.set_ylabel("State")
    ax.set_title(f"{title_prefix}: state versus voltage")
    ax.legend()
    save_figure(fig, plot_dir / "state_vs_voltage.png")

    time, x, xh = downsample_arrays(df, ["time", "x", "xh"])
    fig, ax = plt.subplots()
    ax.plot(time, x, label="x")
    ax.plot(time, xh, label="xh")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("State")
    ax.set_title(f"{title_prefix}: state versus time")
    ax.legend()
    save_figure(fig, plot_dir / "state_vs_time.png")


def plot_static_term_curves(
    h_df: pd.DataFrame,
    v_df: pd.DataFrame,
    idiff_df: pd.DataFrame,
    plot_dir: Path,
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(h_df["h"], h_df["I0_of_h"], label="I0(h)")
    ax.plot(h_df["h"], h_df["Acoef_of_h"], label="Acoef(h)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("h")
    ax.set_ylabel("Term value")
    ax.set_title(f"{title_prefix}: state-dependent terms")
    ax.legend()
    save_figure(fig, plot_dir / "static_terms_vs_h.png")

    fig, ax = plt.subplots()
    ax.semilogy(v_df["V"], np.abs(v_df["Stau_of_v"]) + I_FLOOR_ABS, label="Stau")
    ax.semilogy(v_df["V"], np.abs(v_df["Rtau_of_v"]) + I_FLOOR_ABS, label="Rtau")
    ax.semilogy(v_df["V"], np.abs(v_df["Rh_of_v"]) + I_FLOOR_ABS, label="Rh")
    ax.semilogy(v_df["V"], np.abs(v_df["tau_x_of_v"]) + I_FLOOR_ABS, label="tau_x")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_xlabel("V (V)")
    ax.set_ylabel("Time/resistance term")
    ax.set_title(f"{title_prefix}: voltage-dependent time terms")
    ax.legend()
    save_figure(fig, plot_dir / "static_time_terms_vs_voltage.png")

    fig, ax = plt.subplots()
    ax.set_yscale("symlog", linthresh=SYMLINTHRESH)
    for h_value, group in idiff_df.groupby("h"):
        ax.plot(group["V"], group["Idiff"], label=f"h={h_value:g}")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_xlabel("V (V)")
    ax.set_ylabel("Idiff")
    ax.set_title(f"{title_prefix}: static Idiff slices")
    ax.legend()
    save_figure(fig, plot_dir / "static_idiff_slices.png")


def plot_term_trace_suite(
    term_df: pd.DataFrame,
    family: str,
    plot_dir: Path,
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    for column in TERM_FAMILY_INFO[family]["term_cols"]:
        v, y = downsample_arrays(term_df, ["Vp", column])
        fig, ax = plt.subplots()

        if column in {"Stau_eff", "Rtau_eff", "Rh_eff", "tau_x_eff"}:
            ax.semilogy(v, np.abs(y) + I_FLOOR_ABS)
        elif column in {
            "Idiff_eff",
            "Idiff_pos_exp",
            "Idiff_neg_exp",
            "I_main_est",
            "I_ei_est",
            "I_roff_est",
            "I_model_est",
        }:
            ax.set_yscale("symlog", linthresh=SYMLINTHRESH)
            ax.plot(v, y)
        else:
            ax.plot(v, y)

        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.set_xlabel("Vp (V)")
        ax.set_ylabel(column)
        ax.set_title(f"{title_prefix}: {column}")
        save_figure(fig, plot_dir / f"term_{safe_tag(column)}_vs_voltage.png")


def plot_overlay(
    case_dict: dict[str, pd.DataFrame],
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    yscale: str = "linear",
    absolute_value: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for label, df in case_dict.items():
        x, y = downsample_arrays(df, [x_column, y_column])
        if absolute_value:
            y = np.abs(y)

        if yscale == "log":
            ax.semilogy(x, np.abs(y) + I_FLOOR_ABS, label=label)
        elif yscale == "symlog":
            ax.set_yscale("symlog", linthresh=SYMLINTHRESH)
            ax.plot(x, y, label=label)
        else:
            ax.plot(x, y, label=label)

    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(title)
    ax.legend(fontsize=7)
    save_figure(fig, output_path)


def plot_term_to_result_panel(
    case_dict: dict[str, pd.DataFrame],
    family: str,
    mode: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    term_columns = list(TERM_FAMILY_INFO[family]["term_cols"])
    first_term = term_columns[0]
    second_term = term_columns[1] if len(term_columns) > 1 else first_term

    panel_defs = [
        ("Vp", first_term, "primary term", "auto"),
        ("Vp", second_term, "secondary term", "auto"),
        ("Vp", "I", "I-V", "symlog"),
        ("Vp", "I", "|I|-V", "logabs"),
        ("Vp", "x", "x-V", "linear"),
        ("Vp", "xh", "xh-V", "linear"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    for axis, (x_column, y_column, panel_title, scale) in zip(axes.ravel(), panel_defs):
        for label, df in case_dict.items():
            x, y = downsample_arrays(df, [x_column, y_column])

            selected_scale = scale
            if selected_scale == "auto":
                if y_column in {"Stau_eff", "Rtau_eff", "Rh_eff", "tau_x_eff"}:
                    selected_scale = "logabs"
                elif y_column in {
                    "Idiff_eff",
                    "Idiff_pos_exp",
                    "Idiff_neg_exp",
                    "I_main_est",
                    "I_ei_est",
                    "I_roff_est",
                    "I_model_est",
                }:
                    selected_scale = "symlog"
                else:
                    selected_scale = "linear"

            if selected_scale == "logabs":
                axis.semilogy(x, np.abs(y) + I_FLOOR_ABS, label=label)
            elif selected_scale == "symlog":
                axis.set_yscale("symlog", linthresh=SYMLINTHRESH)
                axis.plot(x, y, label=label)
            else:
                axis.plot(x, y, label=label)

        axis.grid(True, which="both", linestyle="--", alpha=0.35)
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
        axis.set_title(panel_title)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", fontsize=7)
        fig.subplots_adjust(right=0.83)

    fig.suptitle(f"{family} / {mode}: term level to result level")
    save_figure(fig, output_path)


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    ensure_dirs()

    if not THETA_PATH.exists():
        raise FileNotFoundError(f"Missing fitted parameter file: {THETA_PATH}")
    if not NGSPICE.exists():
        raise FileNotFoundError(
            f"NGSpice was not found. Install ngspice or place ngspice.exe at {NGSPICE}"
        )
    if RUN_LIMIT_MODE and RUN_NOLIMIT_MODE:
        raise ValueError("Enable only one of RUN_LIMIT_MODE and RUN_NOLIMIT_MODE")
    if not RUN_LIMIT_MODE and not RUN_NOLIMIT_MODE:
        raise ValueError("Enable one of RUN_LIMIT_MODE and RUN_NOLIMIT_MODE")

    theta_base = read_theta_best(THETA_PATH)
    knobs_base = baseline_model_knobs()
    v_wave = build_baseline_waveform()
    tstop = TSTOP_BASE * TIME_SCALE

    if RUN_LIMIT_MODE:
        if not CSV_MEAS.exists():
            raise FileNotFoundError(
                "LIMIT mode requires data/DC-IV.csv to estimate positive compliance"
            )
        v_meas, i_meas = read_meas_csv(CSV_MEAS)
        icomp_est = estimate_icomp_pos(v_meas, i_meas)
        run_mode = "LIMIT"
        run_settings = {
            "icomp_pos": icomp_est,
            "vcomp": VCOMP_FIXED,
            "rlo": RLO_FIXED,
            "rhi": RHI_FIXED,
            "vslope_pos": VSLOPE_POS_FIXED,
            "islope_rel": ISLOPE_REL,
        }
    else:
        run_mode = "NOLIMIT"
        run_settings = {
            "icomp_pos": 1e30,
            "vcomp": VCOMP_FIXED,
            "rlo": 1e-3,
            "rhi": 1e-3,
            "vslope_pos": VSLOPE_POS_FIXED,
            "islope_rel": ISLOPE_REL,
        }

    print("[info] template:", TEMPLATE_SRC)
    print("[info] theta_best:", THETA_PATH)
    print("[info] ngspice:", NGSPICE)
    print("[info] output:", OUT_DIR)
    print(f"[info] waveform: 0 -> {FIXED_BASELINE_VPOS:g} -> 0 -> {FIXED_BASELINE_VNEG:g} -> 0 V")
    print(f"[info] tstop={tstop:g} s, mode={run_mode}")

    family_map_rows = []
    for family, info in TERM_FAMILY_INFO.items():
        family_map_rows.append(
            {
                "term_family": family,
                "equation": info["equation"],
                "member_params": ", ".join(info["member_params"]),
                "term_cols": ", ".join(info["term_cols"]),
                "result_focus": ", ".join(info["result_focus"]),
            }
        )
    pd.DataFrame(family_map_rows).to_csv(
        REPORT_DIR / "term_family_map.csv", index=False
    )

    manifest_rows: list[dict[str, Any]] = []
    result_metric_rows: list[dict[str, Any]] = []
    term_metric_rows: list[dict[str, Any]] = []
    family_summary_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Baseline case
    # -------------------------------------------------------------------------
    baseline_tag = safe_tag(f"baseline_{run_mode}")
    print(f"[run] {baseline_tag}")

    baseline_df, baseline_plot_dir, baseline_sim_dir = simulate_case(
        theta=theta_base,
        model_knobs=knobs_base,
        v_cmd=v_wave,
        case_tag=baseline_tag,
        tstop=tstop,
        run_settings=run_settings,
    )
    baseline_term_df = calc_term_traces(baseline_df, theta_base, knobs_base)
    baseline_h_df, baseline_v_df, baseline_idiff_df = calc_static_term_curves(
        theta_base,
        knobs_base,
        vmin=float(baseline_df["Vp"].min()),
        vmax=float(baseline_df["Vp"].max()),
    )

    baseline_term_df.to_csv(
        baseline_sim_dir / f"{baseline_tag}_term_trace.csv", index=False
    )
    baseline_h_df.to_csv(
        baseline_sim_dir / f"{baseline_tag}_static_terms_h.csv", index=False
    )
    baseline_v_df.to_csv(
        baseline_sim_dir / f"{baseline_tag}_static_terms_v.csv", index=False
    )
    baseline_idiff_df.to_csv(
        baseline_sim_dir / f"{baseline_tag}_static_idiff_slices.csv", index=False
    )

    plot_result_suite(baseline_df, baseline_plot_dir, baseline_tag)
    plot_static_term_curves(
        baseline_h_df,
        baseline_v_df,
        baseline_idiff_df,
        baseline_plot_dir,
        baseline_tag,
    )

    manifest_rows.append(
        {
            "case_tag": baseline_tag,
            "family": "BASELINE",
            "mode": "BASELINE",
            "value": 1.0,
            "label": "baseline",
            "run_mode": run_mode,
        }
    )
    result_metric_rows.append(
        {
            "case_tag": baseline_tag,
            "family": "BASELINE",
            "mode": "BASELINE",
            "value": 1.0,
            **calc_result_metrics(baseline_df),
        }
    )
    term_metric_rows.append(
        {
            "case_tag": baseline_tag,
            "family": "BASELINE",
            "mode": "BASELINE",
            "value": 1.0,
            **calc_term_trace_metrics(baseline_term_df),
        }
    )

    # -------------------------------------------------------------------------
    # Eight equation-term families
    # -------------------------------------------------------------------------
    for family, mode_list in TERM_SWEEP_CONFIG.items():
        for mode_config in mode_list:
            mode = str(mode_config["mode"])
            values = [float(value) for value in mode_config["values"]]

            result_overlays: dict[str, pd.DataFrame] = {}
            term_overlays: dict[str, pd.DataFrame] = {}
            successful_case_tags: list[str] = []

            for value in values:
                theta_case, knobs_case = apply_term_sweep(
                    theta_base=theta_base,
                    knobs_base=knobs_base,
                    family=family,
                    mode=mode,
                    value=value,
                )
                label = build_case_label(
                    family=family,
                    mode=mode,
                    value=value,
                    theta=theta_case,
                    knobs=knobs_case,
                )
                case_tag = safe_tag(f"{family}_{mode}_{value:g}_{run_mode}")
                print(f"[run] {case_tag}")

                try:
                    sim_df, plot_dir, sim_dir = simulate_case(
                        theta=theta_case,
                        model_knobs=knobs_case,
                        v_cmd=v_wave,
                        case_tag=case_tag,
                        tstop=tstop,
                        run_settings=run_settings,
                    )

                    term_df = calc_term_traces(sim_df, theta_case, knobs_case)
                    h_df, v_df, idiff_df = calc_static_term_curves(
                        theta_case,
                        knobs_case,
                        vmin=float(sim_df["Vp"].min()),
                        vmax=float(sim_df["Vp"].max()),
                    )

                    term_df.to_csv(
                        sim_dir / f"{case_tag}_term_trace.csv", index=False
                    )
                    h_df.to_csv(
                        sim_dir / f"{case_tag}_static_terms_h.csv", index=False
                    )
                    v_df.to_csv(
                        sim_dir / f"{case_tag}_static_terms_v.csv", index=False
                    )
                    idiff_df.to_csv(
                        sim_dir / f"{case_tag}_static_idiff_slices.csv", index=False
                    )

                    plot_result_suite(sim_df, plot_dir, case_tag)
                    plot_static_term_curves(
                        h_df, v_df, idiff_df, plot_dir, case_tag
                    )
                    plot_term_trace_suite(term_df, family, plot_dir, case_tag)

                    manifest_rows.append(
                        {
                            "case_tag": case_tag,
                            "family": family,
                            "mode": mode,
                            "value": value,
                            "label": label,
                            "run_mode": run_mode,
                            **{
                                f"theta_{key}": float(theta_case[key])
                                for key in sorted(theta_case)
                            },
                            **{
                                f"knob_{key}": float(knobs_case[key])
                                for key in sorted(knobs_case)
                            },
                        }
                    )
                    result_metric_rows.append(
                        {
                            "case_tag": case_tag,
                            "family": family,
                            "mode": mode,
                            "value": value,
                            **calc_result_metrics(sim_df),
                        }
                    )
                    term_metric_rows.append(
                        {
                            "case_tag": case_tag,
                            "family": family,
                            "mode": mode,
                            "value": value,
                            **calc_term_trace_metrics(term_df),
                        }
                    )

                    result_overlays[label] = sim_df
                    term_overlays[label] = term_df
                    successful_case_tags.append(case_tag)

                except Exception as exc:
                    print(f"[failed] {case_tag}: {exc}")
                    failed_rows.append(
                        {
                            "case_tag": case_tag,
                            "family": family,
                            "mode": mode,
                            "value": value,
                            "error": str(exc),
                        }
                    )

            if term_overlays:
                stem = safe_tag(f"{family}_{mode}")
                primary_term = str(TERM_FAMILY_INFO[family]["term_cols"][0])

                if primary_term in {"Stau_eff", "Rtau_eff", "Rh_eff", "tau_x_eff"}:
                    primary_scale = "log"
                    primary_abs = True
                elif primary_term in {
                    "Idiff_eff",
                    "Idiff_pos_exp",
                    "Idiff_neg_exp",
                    "I_main_est",
                    "I_ei_est",
                    "I_roff_est",
                    "I_model_est",
                }:
                    primary_scale = "symlog"
                    primary_abs = False
                else:
                    primary_scale = "linear"
                    primary_abs = False

                plot_term_to_result_panel(
                    term_overlays,
                    family,
                    mode,
                    OVERLAY_DIR / f"{stem}_term_to_result_panel.png",
                )
                plot_overlay(
                    term_overlays,
                    "Vp",
                    primary_term,
                    OVERLAY_DIR / f"{stem}_primary_term_overlay.png",
                    f"{family}/{mode}: {primary_term}",
                    yscale=primary_scale,
                    absolute_value=primary_abs,
                )
                plot_overlay(
                    result_overlays,
                    "Vp",
                    "I",
                    OVERLAY_DIR / f"{stem}_iv_symlog_overlay.png",
                    f"{family}/{mode}: I-V",
                    yscale="symlog",
                )
                plot_overlay(
                    result_overlays,
                    "Vp",
                    "I",
                    OVERLAY_DIR / f"{stem}_logabsI_overlay.png",
                    f"{family}/{mode}: |I|-V",
                    yscale="log",
                    absolute_value=True,
                )
                plot_overlay(
                    result_overlays,
                    "Vp",
                    "x",
                    OVERLAY_DIR / f"{stem}_x_overlay.png",
                    f"{family}/{mode}: x-V",
                )
                plot_overlay(
                    result_overlays,
                    "Vp",
                    "xh",
                    OVERLAY_DIR / f"{stem}_xh_overlay.png",
                    f"{family}/{mode}: xh-V",
                )

            family_summary_rows.append(
                {
                    "family": family,
                    "mode": mode,
                    "requested_case_count": len(values),
                    "successful_case_count": len(successful_case_tags),
                    "failed_case_count": len(values) - len(successful_case_tags),
                    "primary_term_col": TERM_FAMILY_INFO[family]["term_cols"][0],
                    "equation": TERM_FAMILY_INFO[family]["equation"],
                }
            )

    pd.DataFrame(manifest_rows).to_csv(
        REPORT_DIR / "case_manifest.csv", index=False
    )
    pd.DataFrame(result_metric_rows).to_csv(
        REPORT_DIR / "result_metrics.csv", index=False
    )
    pd.DataFrame(term_metric_rows).to_csv(
        REPORT_DIR / "term_trace_metrics.csv", index=False
    )
    pd.DataFrame(family_summary_rows).to_csv(
        REPORT_DIR / "family_summary.csv", index=False
    )

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(
            REPORT_DIR / "failed_cases.csv", index=False
        )

    print("[done] equation-term sweep completed")
    print("[saved]", OUT_DIR)


if __name__ == "__main__":
    main()
