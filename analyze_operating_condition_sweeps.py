"""
analyze_operating_condition_sweeps.py
=====================================
Unified NGSpice analysis workflow for a fitted memristive-device model.

This script combines the original frequency-sweep and voltage-sweep analyzers
into one operating-condition study. Model parameters remain fixed while the
external stimulus is changed.

Supported analyses
------------------
1. Frequency / sweep-rate sweep
2. Positive-voltage amplitude sweep
3. Negative-voltage amplitude sweep
4. Multi-step negative-voltage sequences
5. Repeated negative-voltage cycles
6. Optional staircase-input comparison
7. Optional measurement-trace replay

"""

from __future__ import annotations

import math
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

OUT_DIR = BASE_DIR / "results" / "operating_condition_sweeps"
TEMPLATE_SNAPSHOT = OUT_DIR / "fitdeck_embedded.cir"
CASES_DIR = OUT_DIR / "cases"
OVERLAY_DIR = OUT_DIR / "overlays"
FREQUENCY_METRIC_DIR = OUT_DIR / "metrics_vs_frequency"


# =============================================================================
# Simulation and numerical settings
# =============================================================================

TSTOP_BASE = 10.0
DTMAX_SIM = 2e-4

PRINT_DIV = 6
PRINT_MIN = 2e-5
PRINT_MAX = 4e-3

NGSPICE_TIMEOUT_S = 180
RETRY_ON_FAILURE_OR_TRUNCATION = True
RETRY_DTMAX_FACTORS = [0.5, 0.25]
RETRY_TIMEOUTS_S = [240, 360]
INCOMPLETE_TIME_FRAC = 0.995

# Preserve approximately the same time per voltage ramp when a waveform has
# more than the four ramps in 0 -> +V -> 0 -> -V -> 0.
NORMALIZE_TSTOP_BY_RAMP_COUNT = True
BASELINE_RAMP_COUNT = 4


# =============================================================================
# Fixed model knobs used by the fitted SPICE template
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
# Analysis selection
# =============================================================================

ENABLE_MEASUREMENT_TRACE = True
ENABLE_FREQUENCY_SWEEP = True
ENABLE_VOLTAGE_SWEEP = True
ENABLE_MULTI_NEGATIVE_SWEEP = True
ENABLE_REPEAT_CYCLE_SWEEP = True
ENABLE_STAIRCASE_SWEEP = False


# =============================================================================
# Input-waveform settings
# =============================================================================

# Common waveform resolution for continuous ramps.
PTS_PER_RAMP = 160
HOLD_PTS = 0

# ----- Frequency sweep -----
# frequency_Hz = 1 / (TSTOP_BASE * time_scale)
# With TSTOP_BASE = 10 s:
#   0.25 -> 0.4 Hz
#   0.50 -> 0.2 Hz
#   1.00 -> 0.1 Hz
#   2.00 -> 0.05 Hz
#   4.00 -> 0.025 Hz
FREQUENCY_TIME_SCALE_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]
FREQUENCY_VPOS = 10.0
FREQUENCY_VNEG = -4.0

# ----- Voltage-amplitude sweep -----
VOLTAGE_SWEEP_TIME_SCALE = 1.0

VPOS_LIST = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
FIXED_VNEG_FOR_VPOS_SWEEP = -4.0

VNEG_LIST = [-2.0, -3.0, -4.0, -5.0, -6.0, -8.0, -10.0]
FIXED_VPOS_FOR_VNEG_SWEEP = 10.0

# ----- Multi-step negative-voltage sequences -----
MULTI_NEGATIVE_VPOS_LIST = [6.0, 8.0, 10.0]
MULTI_NEGATIVE_SEQUENCE_LIST = [
    [-2.0, -4.0, -6.0, -8.0, -10.0],
    [-10.0, -8.0, -6.0, -4.0, -2.0],
]
MULTI_NEGATIVE_TIME_SCALE = 1.0

# ----- Repeated negative cycles -----
REPEAT_VPOS_LIST = [10.0]
REPEAT_VNEG_LIST = [-10.0]
REPEAT_COUNT_LIST = [1, 2, 3, 5]
REPEAT_TIME_SCALE = 1.0

# ----- Optional staircase comparison -----
STAIRCASE_VPOS = 10.0
STAIRCASE_VNEG = -4.0
LEVELS_PER_RAMP_LIST = [3, 17, 256]
HOLD_PTS_PER_LEVEL = 4
STAIRCASE_TIME_SCALE = 1.0


# =============================================================================
# Plot and metric settings
# =============================================================================

I_FLOOR_ABS = 3e-10
SYMLINTHRESH = 1e-9
MAX_PLOT_POINTS = 24000
BRANCH_SAMPLE_V_LIST = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
MAX_BRANCH_V_ERROR = 0.25


# =============================================================================
# Fitted parameters expected in theta_best.csv
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
# Basic utilities
# =============================================================================


def ensure_dirs() -> None:
    for directory in (OUT_DIR, CASES_DIR, OVERLAY_DIR, FREQUENCY_METRIC_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE_SRC.exists():
        raise FileNotFoundError(f"Missing SPICE template: {TEMPLATE_SRC}")

    TEMPLATE_SNAPSHOT.write_text(
        TEMPLATE_SRC.read_text(encoding="utf-8", errors="ignore"),
        encoding="utf-8",
    )


def safe_tag(value: str) -> str:
    value = str(value).strip().replace(" ", "_")
    value = value.replace("/", "_").replace("\\", "_").replace(":", "_")
    value = re.sub(r"[^A-Za-z0-9_.+\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def fmt_num(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(number):
        return "na"
    if abs(number - round(number)) < 1e-12:
        return str(int(round(number)))
    return f"{number:g}"


def make_case_dirs(case_tag: str) -> dict[str, Path]:
    case_dir = CASES_DIR / case_tag
    dirs = {
        "case": case_dir,
        "deck": case_dir / "decks",
        "log": case_dir / "logs",
        "sim": case_dir / "sims",
        "plot": case_dir / "plots",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def read_measurement_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, engine="python")
    if df.shape[1] < 2:
        raise ValueError("DC-IV.csv must contain at least two columns: voltage and current.")

    voltage = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    current = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    valid = voltage.notna() & current.notna()

    v = voltage[valid].to_numpy(dtype=float)
    i = current[valid].to_numpy(dtype=float)
    if len(v) < 10:
        raise ValueError("Too few numeric rows remain after cleaning DC-IV.csv.")
    return v, i


def read_theta_best(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if not {"param", "value"}.issubset(df.columns):
        raise ValueError("theta_best.csv must contain columns named 'param' and 'value'.")

    theta: dict[str, float] = {}
    for raw_name, raw_value in zip(df["param"], df["value"]):
        name = str(raw_name).strip()
        name = ALIASES.get(name, name)
        theta[name] = float(raw_value)

    missing = sorted(REQUIRED_PARAMS.difference(theta))
    if missing:
        raise ValueError(
            f"theta_best.csv is missing required parameters: {missing}\n"
            f"Loaded parameters: {sorted(theta)}"
        )
    return theta


def estimate_positive_compliance(voltage: np.ndarray, current: np.ndarray) -> float:
    mask = voltage > 0.5
    if not np.any(mask):
        return 1e-3
    estimate = float(np.quantile(np.abs(current[mask]), 0.98))
    return float(np.clip(estimate, 1e-6, 5e-2))


def pick_tstep_print(num_points: int, tstop: float) -> float:
    if num_points <= 1:
        return 1e-3
    dt_command = float(tstop) / max(num_points - 1, 1)
    return float(np.clip(dt_command / PRINT_DIV, PRINT_MIN, PRINT_MAX))


def make_time_vector(num_points: int, tstop: float) -> np.ndarray:
    if num_points < 2:
        return np.array([0.0])
    return np.linspace(0.0, float(tstop), int(num_points))


def pwl_inline_from_tv(time: np.ndarray, voltage: np.ndarray, pairs_per_line: int = 8) -> str:
    pairs = [f"{t:.12g} {v:.12g}" for t, v in zip(time, voltage)]
    return "\n".join(
        "+ " + " ".join(pairs[start : start + pairs_per_line])
        for start in range(0, len(pairs), pairs_per_line)
    )


def tail_text(path: Path | None, n_lines: int = 100) -> str:
    if path is None or not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except OSError:
        return ""


def downsample_arrays(
    df: pd.DataFrame,
    columns: Iterable[str],
    max_points: int = MAX_PLOT_POINTS,
) -> list[np.ndarray]:
    columns = list(columns)
    if len(df) <= max_points:
        return [df[column].to_numpy(dtype=float) for column in columns]
    indices = np.linspace(0, len(df) - 1, max_points).astype(int)
    return [df[column].to_numpy(dtype=float)[indices] for column in columns]


# =============================================================================
# Waveform construction
# =============================================================================


def make_voltage_sequence(
    nodes: Iterable[float],
    pts_per_ramp: int = PTS_PER_RAMP,
    hold_pts: int = HOLD_PTS,
) -> np.ndarray:
    """Create a continuous, piecewise-linear command waveform."""
    node_list = [float(value) for value in nodes]
    if len(node_list) < 2:
        raise ValueError("At least two voltage nodes are required.")

    pts_per_ramp = max(2, int(pts_per_ramp))
    repeat_count = max(1, int(hold_pts) + 1)

    segments: list[np.ndarray] = []
    for index, (start, stop) in enumerate(zip(node_list[:-1], node_list[1:])):
        segment = np.linspace(start, stop, pts_per_ramp)
        if index > 0:
            segment = segment[1:]
        if repeat_count > 1:
            segment = np.repeat(segment, repeat_count)
        segments.append(segment)

    return np.concatenate(segments)


def make_staircase_sequence(
    nodes: Iterable[float],
    levels_per_ramp: int,
    hold_pts_per_level: int,
) -> np.ndarray:
    """Create a quantized staircase waveform with held voltage levels."""
    node_list = [float(value) for value in nodes]
    if len(node_list) < 2:
        raise ValueError("At least two voltage nodes are required.")

    levels_per_ramp = max(2, int(levels_per_ramp))
    hold_pts_per_level = max(1, int(hold_pts_per_level))

    segments: list[np.ndarray] = []
    for index, (start, stop) in enumerate(zip(node_list[:-1], node_list[1:])):
        levels = np.linspace(start, stop, levels_per_ramp)
        if index > 0:
            levels = levels[1:]
        segments.append(np.repeat(levels, hold_pts_per_level))

    return np.concatenate(segments)


def build_case(
    *,
    name: str,
    label: str,
    group: str,
    nodes: list[float] | None,
    voltage_command: np.ndarray,
    time_scale: float,
    input_style: str = "linear",
    **metadata: Any,
) -> dict[str, Any]:
    num_ramps = len(nodes) - 1 if nodes is not None else BASELINE_RAMP_COUNT
    case = {
        "name": name,
        "label": label,
        "group": group,
        "nodes": nodes,
        "V_cmd": np.asarray(voltage_command, dtype=float),
        "time_scale": float(time_scale),
        "input_style": input_style,
        "num_ramps": int(num_ramps),
        "pts_per_ramp": metadata.pop("pts_per_ramp", np.nan),
        "hold_pts": metadata.pop("hold_pts", np.nan),
        "levels_per_ramp": metadata.pop("levels_per_ramp", np.nan),
        "hold_pts_per_level": metadata.pop("hold_pts_per_level", np.nan),
    }
    case.update(metadata)
    return case


def build_operating_condition_cases(v_meas: np.ndarray) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    if ENABLE_MEASUREMENT_TRACE:
        cases.append(
            build_case(
                name="measurement_trace",
                label="measurement-driven waveform",
                group="measurement_trace",
                nodes=None,
                voltage_command=v_meas,
                time_scale=1.0,
                input_style="measurement",
                vpos=float(np.max(v_meas)),
                vneg=float(np.min(v_meas)),
            )
        )

    if ENABLE_FREQUENCY_SWEEP:
        nodes = [0.0, FREQUENCY_VPOS, 0.0, FREQUENCY_VNEG, 0.0]
        command = make_voltage_sequence(nodes, PTS_PER_RAMP, HOLD_PTS)
        for time_scale in FREQUENCY_TIME_SCALE_LIST:
            period = TSTOP_BASE * float(time_scale)
            frequency = 1.0 / period
            cases.append(
                build_case(
                    name=f"frequency_{fmt_num(frequency)}Hz",
                    label=f"f = {fmt_num(frequency)} Hz, T = {fmt_num(period)} s",
                    group="frequency",
                    nodes=nodes,
                    voltage_command=command,
                    time_scale=time_scale,
                    pts_per_ramp=PTS_PER_RAMP,
                    hold_pts=HOLD_PTS,
                    vpos=FREQUENCY_VPOS,
                    vneg=FREQUENCY_VNEG,
                    frequency_Hz=frequency,
                )
            )

    if ENABLE_VOLTAGE_SWEEP:
        for vpos in VPOS_LIST:
            nodes = [0.0, float(vpos), 0.0, FIXED_VNEG_FOR_VPOS_SWEEP, 0.0]
            cases.append(
                build_case(
                    name=f"amp_vpos_{fmt_num(vpos)}V",
                    label=f"Vpos = {fmt_num(vpos)} V",
                    group="amp_vpos",
                    nodes=nodes,
                    voltage_command=make_voltage_sequence(nodes, PTS_PER_RAMP, HOLD_PTS),
                    time_scale=VOLTAGE_SWEEP_TIME_SCALE,
                    pts_per_ramp=PTS_PER_RAMP,
                    hold_pts=HOLD_PTS,
                    vpos=float(vpos),
                    vneg=FIXED_VNEG_FOR_VPOS_SWEEP,
                )
            )

        for vneg in VNEG_LIST:
            nodes = [0.0, FIXED_VPOS_FOR_VNEG_SWEEP, 0.0, float(vneg), 0.0]
            cases.append(
                build_case(
                    name=f"amp_vneg_{fmt_num(vneg)}V",
                    label=f"Vneg = {fmt_num(vneg)} V",
                    group="amp_vneg",
                    nodes=nodes,
                    voltage_command=make_voltage_sequence(nodes, PTS_PER_RAMP, HOLD_PTS),
                    time_scale=VOLTAGE_SWEEP_TIME_SCALE,
                    pts_per_ramp=PTS_PER_RAMP,
                    hold_pts=HOLD_PTS,
                    vpos=FIXED_VPOS_FOR_VNEG_SWEEP,
                    vneg=float(vneg),
                )
            )

    if ENABLE_MULTI_NEGATIVE_SWEEP:
        for vpos in MULTI_NEGATIVE_VPOS_LIST:
            for sequence in MULTI_NEGATIVE_SEQUENCE_LIST:
                nodes = [0.0, float(vpos), 0.0]
                for vneg in sequence:
                    nodes.extend([float(vneg), 0.0])
                sequence_tag = "_".join(fmt_num(value) for value in sequence)
                cases.append(
                    build_case(
                        name=f"multi_vpos_{fmt_num(vpos)}_seq_{sequence_tag}",
                        label=(
                            f"Vpos = {fmt_num(vpos)} V, "
                            f"sequence = {' -> '.join(fmt_num(value) for value in sequence)} V"
                        ),
                        group="multi_negative",
                        nodes=nodes,
                        voltage_command=make_voltage_sequence(nodes, PTS_PER_RAMP, HOLD_PTS),
                        time_scale=MULTI_NEGATIVE_TIME_SCALE,
                        pts_per_ramp=PTS_PER_RAMP,
                        hold_pts=HOLD_PTS,
                        vpos=float(vpos),
                        vneg=float(np.min(sequence)),
                        sequence=sequence_tag,
                    )
                )

    if ENABLE_REPEAT_CYCLE_SWEEP:
        for vpos in REPEAT_VPOS_LIST:
            for vneg in REPEAT_VNEG_LIST:
                for repeat_count in REPEAT_COUNT_LIST:
                    nodes = [0.0, float(vpos), 0.0]
                    for _ in range(int(repeat_count)):
                        nodes.extend([float(vneg), 0.0])
                    cases.append(
                        build_case(
                            name=(
                                f"repeat_vpos_{fmt_num(vpos)}_"
                                f"vneg_{fmt_num(vneg)}_x{repeat_count}"
                            ),
                            label=(
                                f"Vneg = {fmt_num(vneg)} V, "
                                f"repeat = {repeat_count}"
                            ),
                            group="repeat_negative",
                            nodes=nodes,
                            voltage_command=make_voltage_sequence(nodes, PTS_PER_RAMP, HOLD_PTS),
                            time_scale=REPEAT_TIME_SCALE,
                            pts_per_ramp=PTS_PER_RAMP,
                            hold_pts=HOLD_PTS,
                            vpos=float(vpos),
                            vneg=float(vneg),
                            repeat_count=int(repeat_count),
                        )
                    )

    if ENABLE_STAIRCASE_SWEEP:
        nodes = [0.0, STAIRCASE_VPOS, 0.0, STAIRCASE_VNEG, 0.0]
        for levels_per_ramp in LEVELS_PER_RAMP_LIST:
            cases.append(
                build_case(
                    name=f"staircase_{levels_per_ramp}_levels_per_ramp",
                    label=f"{levels_per_ramp} levels/ramp",
                    group="staircase",
                    nodes=nodes,
                    voltage_command=make_staircase_sequence(
                        nodes,
                        levels_per_ramp=levels_per_ramp,
                        hold_pts_per_level=HOLD_PTS_PER_LEVEL,
                    ),
                    time_scale=STAIRCASE_TIME_SCALE,
                    input_style="staircase",
                    levels_per_ramp=levels_per_ramp,
                    hold_pts_per_level=HOLD_PTS_PER_LEVEL,
                    vpos=STAIRCASE_VPOS,
                    vneg=STAIRCASE_VNEG,
                )
            )

    return cases


def compute_case_tstop(case: dict[str, Any]) -> float:
    base_tstop = TSTOP_BASE * float(case["time_scale"])
    if case["group"] == "frequency":
        return base_tstop
    if case["input_style"] == "measurement":
        return base_tstop
    if NORMALIZE_TSTOP_BY_RAMP_COUNT:
        return base_tstop * float(case["num_ramps"]) / BASELINE_RAMP_COUNT
    return base_tstop


# =============================================================================
# NGSpice execution
# =============================================================================


def run_ngspice(deck_path: Path, log_path: Path, cwd: Path, timeout_s: int) -> int:
    command = [str(NGSPICE), "-b", "-o", str(log_path), str(deck_path)]
    try:
        completed = subprocess.run(command, cwd=str(cwd), timeout=int(timeout_s), check=False)
        return int(completed.returncode)
    except subprocess.TimeoutExpired:
        return 124
    except OSError as exc:
        log_path.write_text(f"Failed to start NGSpice: {exc}\n", encoding="utf-8")
        return 127


def load_wrdata(path: Path) -> pd.DataFrame:
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise ValueError(f"Expected at least 6 wrdata columns, found {data.shape[1]} in {path}")

    df = pd.DataFrame(
        {
            "time": data[:, 0],
            "Vcmd": data[:, 1],
            "Vp": data[:, 2],
            "I": data[:, 3],
            "x": data[:, 4],
            "xh": data[:, 5],
        }
    )
    final_time = max(float(df["time"].iloc[-1]), 1e-30)
    df["phase"] = df["time"] / final_time
    return df


def is_simulation_complete(df: pd.DataFrame, tstop: float) -> bool:
    if df.empty:
        return False
    return float(df["time"].iloc[-1]) >= INCOMPLETE_TIME_FRAC * float(tstop)


def build_deck_text(
    *,
    theta: dict[str, float],
    voltage_command: np.ndarray,
    simulation_output: Path,
    tstop: float,
    dtmax: float,
    mode_settings: dict[str, float],
) -> str:
    time = make_time_vector(len(voltage_command), tstop)
    pwl_inline = pwl_inline_from_tv(time, voltage_command)
    tstep_print = pick_tstep_print(len(voltage_command), tstop)

    icomp_pos = float(mode_settings["icomp_pos"])
    islope = max(1e-12, ISLOPE_REL * icomp_pos)

    replacements: dict[str, str] = {
        "@PWL_INLINE@": pwl_inline,
        "@SIMOUT@": str(simulation_output),
        "@TSTEP@": f"{tstep_print:.12g}",
        "@DTMAX@": f"{float(dtmax):.12g}",
        "@TSTOP@": f"{float(tstop):.12g}",
        "@KSW@": f"{KSW_FIXED:.12g}",
        "@RH0@": f"{RH0_FIXED:.12g}",
        "@RH_MIN@": f"{RH_MIN_FIXED:.12g}",
        "@RH_MAX@": f"{RH_MAX_FIXED:.12g}",
        "@VSLOPE@": f"{VSLOPE_FIXED:.12g}",
        "@ICOMP_POS@": f"{icomp_pos:.12g}",
        "@VCOMP@": f"{float(mode_settings['vcomp']):.12g}",
        "@RLO@": f"{float(mode_settings['rlo']):.12g}",
        "@RHI@": f"{float(mode_settings['rhi']):.12g}",
        "@ISLOPE@": f"{islope:.12g}",
        "@VSLOPE_POS@": f"{VSLOPE_POS_FIXED:.12g}",
    }

    for name, value in theta.items():
        replacements[f"@{name}@"] = f"{float(value):.12g}"

    text = TEMPLATE_SNAPSHOT.read_text(encoding="utf-8", errors="ignore")
    for token, replacement in replacements.items():
        text = text.replace(token, replacement)

    leftovers = sorted(set(re.findall(r"@[A-Za-z0-9_]+@", text)))
    if leftovers:
        raise RuntimeError(f"Unreplaced SPICE-template placeholders: {leftovers}")
    return text


def simulate_case(
    *,
    theta: dict[str, float],
    case: dict[str, Any],
    case_tag: str,
    tstop: float,
    mode_settings: dict[str, float],
    dtmax: float,
    timeout_s: int,
) -> tuple[pd.DataFrame | None, dict[str, Path], Path, str]:
    dirs = make_case_dirs(case_tag)
    deck_path = dirs["deck"] / f"{case_tag}.cir"
    log_path = dirs["log"] / f"{case_tag}.log"
    sim_path = dirs["sim"] / f"{case_tag}.dat"

    deck_text = build_deck_text(
        theta=theta,
        voltage_command=case["V_cmd"],
        simulation_output=sim_path,
        tstop=tstop,
        dtmax=dtmax,
        mode_settings=mode_settings,
    )
    deck_path.write_text(deck_text, encoding="utf-8")

    return_code = run_ngspice(deck_path, log_path, dirs["case"], timeout_s)
    if return_code != 0:
        return None, dirs, log_path, f"ngspice_return_code_{return_code}"
    if not sim_path.exists():
        return None, dirs, log_path, "missing_simulation_output"

    try:
        df = load_wrdata(sim_path)
    except (OSError, ValueError) as exc:
        return None, dirs, log_path, f"invalid_simulation_output: {exc}"

    df.to_csv(dirs["sim"] / f"{case_tag}_sim.csv", index=False)
    if not is_simulation_complete(df, tstop):
        return df, dirs, log_path, "truncated_simulation"
    return df, dirs, log_path, "ok"


# =============================================================================
# Metrics
# =============================================================================


def fill_zero_sign(values: np.ndarray) -> np.ndarray:
    signs = np.asarray(values, dtype=float).copy()
    if len(signs) == 0:
        return signs

    for index in range(1, len(signs)):
        if signs[index] == 0:
            signs[index] = signs[index - 1]
    for index in range(len(signs) - 2, -1, -1):
        if signs[index] == 0:
            signs[index] = signs[index + 1]
    return signs


def sweep_direction(voltage_command: np.ndarray) -> np.ndarray:
    voltage_command = np.asarray(voltage_command, dtype=float)
    if len(voltage_command) <= 1:
        return np.zeros_like(voltage_command)
    signs = fill_zero_sign(np.sign(np.diff(voltage_command)))
    return np.r_[signs, signs[-1]]


def nearest_masked_index(
    mask: np.ndarray,
    target_voltage: float,
    voltage: np.ndarray,
) -> int | None:
    indices = np.where(mask)[0]
    if indices.size == 0:
        return None
    best = int(indices[np.argmin(np.abs(voltage[indices] - target_voltage))])
    if abs(float(voltage[best]) - float(target_voltage)) > MAX_BRANCH_V_ERROR:
        return None
    return best


def safe_logabs(value: float) -> float:
    return float(np.log10(abs(float(value)) + I_FLOOR_ABS))


def loop_metrics(voltage: np.ndarray, current: np.ndarray) -> dict[str, float]:
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(voltage) < 2:
        return {"loop_area_signed_A_V": np.nan, "loop_area_abs_path_A_V": np.nan}

    delta_v = np.diff(voltage)
    current_mid = 0.5 * (current[:-1] + current[1:])
    return {
        "loop_area_signed_A_V": float(np.sum(current_mid * delta_v)),
        "loop_area_abs_path_A_V": float(np.sum(np.abs(current_mid * delta_v))),
    }


def compute_case_metrics(df: pd.DataFrame, case: dict[str, Any], tstop: float) -> dict[str, Any]:
    time = df["time"].to_numpy(dtype=float)
    vcmd = df["Vcmd"].to_numpy(dtype=float)
    vp = df["Vp"].to_numpy(dtype=float)
    current = df["I"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    xh = df["xh"].to_numpy(dtype=float)

    metrics: dict[str, Any] = {
        "N": int(len(df)),
        "time_end_s": float(time[-1]),
        "expected_tstop_s": float(tstop),
        "frequency_Hz": case.get("frequency_Hz", np.nan),
        "Vcmd_min": float(np.min(vcmd)),
        "Vcmd_max": float(np.max(vcmd)),
        "Vp_min": float(np.min(vp)),
        "Vp_max": float(np.max(vp)),
        "Imax_abs": float(np.max(np.abs(current))),
        "I_pos_max": float(np.max(current)),
        "I_neg_min": float(np.min(current)),
        "x_start": float(x[0]),
        "x_end": float(x[-1]),
        "x_recovery_signed": float(x[-1] - x[0]),
        "x_recovery_abs": float(abs(x[-1] - x[0])),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "x_span": float(np.max(x) - np.min(x)),
        "xh_start": float(xh[0]),
        "xh_end": float(xh[-1]),
        "xh_recovery_signed": float(xh[-1] - xh[0]),
        "xh_recovery_abs": float(abs(xh[-1] - xh[0])),
        "xh_min": float(np.min(xh)),
        "xh_max": float(np.max(xh)),
        "xh_span": float(np.max(xh) - np.min(xh)),
    }
    metrics.update(loop_metrics(vp, current))

    direction = sweep_direction(vcmd)
    positive_up = (vp >= 0) & (direction > 0)
    positive_down = (vp >= 0) & (direction < 0)
    negative_down = (vp <= 0) & (direction < 0)
    negative_up = (vp <= 0) & (direction > 0)

    for target in BRANCH_SAMPLE_V_LIST:
        tag = str(target).replace(".", "p")
        indices = {
            "pos_up": nearest_masked_index(positive_up, target, vp),
            "pos_down": nearest_masked_index(positive_down, target, vp),
            "neg_down": nearest_masked_index(negative_down, -target, vp),
            "neg_up": nearest_masked_index(negative_up, -target, vp),
        }

        for branch_name, index in indices.items():
            metrics[f"I_{branch_name}_at_{tag}V"] = (
                float(current[index]) if index is not None else np.nan
            )

        pos_up_idx = indices["pos_up"]
        pos_down_idx = indices["pos_down"]
        neg_down_idx = indices["neg_down"]
        neg_up_idx = indices["neg_up"]

        if pos_up_idx is not None and pos_down_idx is not None:
            metrics[f"branch_sep_pos_absI_at_{tag}V"] = float(
                abs(current[pos_up_idx] - current[pos_down_idx])
            )
            metrics[f"branch_sep_pos_logdec_at_{tag}V"] = float(
                abs(safe_logabs(current[pos_up_idx]) - safe_logabs(current[pos_down_idx]))
            )
        else:
            metrics[f"branch_sep_pos_absI_at_{tag}V"] = np.nan
            metrics[f"branch_sep_pos_logdec_at_{tag}V"] = np.nan

        if neg_down_idx is not None and neg_up_idx is not None:
            metrics[f"branch_sep_neg_absI_at_{tag}V"] = float(
                abs(current[neg_down_idx] - current[neg_up_idx])
            )
            metrics[f"branch_sep_neg_logdec_at_{tag}V"] = float(
                abs(safe_logabs(current[neg_down_idx]) - safe_logabs(current[neg_up_idx]))
            )
        else:
            metrics[f"branch_sep_neg_absI_at_{tag}V"] = np.nan
            metrics[f"branch_sep_neg_logdec_at_{tag}V"] = np.nan

    return metrics


def flatten_branch_metrics(
    *,
    case_tag: str,
    case: dict[str, Any],
    mode: str,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in BRANCH_SAMPLE_V_LIST:
        tag = str(target).replace(".", "p")
        rows.append(
            {
                "case_tag": case_tag,
                "case": case["name"],
                "mode": mode,
                "group": case["group"],
                "sample_absV": float(target),
                "I_pos_up": metrics.get(f"I_pos_up_at_{tag}V", np.nan),
                "I_pos_down": metrics.get(f"I_pos_down_at_{tag}V", np.nan),
                "pos_abs_sep": metrics.get(f"branch_sep_pos_absI_at_{tag}V", np.nan),
                "pos_logdec_sep": metrics.get(
                    f"branch_sep_pos_logdec_at_{tag}V", np.nan
                ),
                "I_neg_down": metrics.get(f"I_neg_down_at_{tag}V", np.nan),
                "I_neg_up": metrics.get(f"I_neg_up_at_{tag}V", np.nan),
                "neg_abs_sep": metrics.get(f"branch_sep_neg_absI_at_{tag}V", np.nan),
                "neg_logdec_sep": metrics.get(
                    f"branch_sep_neg_logdec_at_{tag}V", np.nan
                ),
            }
        )
    return rows


# =============================================================================
# Plotting
# =============================================================================


def save_current_figure(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_vcmd_vs_time(df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    time, voltage = downsample_arrays(df, ["time", "Vcmd"])
    plt.figure()
    plt.plot(time, voltage, linewidth=1.2)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xlabel("time (s)")
    plt.ylabel("Vcmd (V)")
    plt.title(title)
    save_current_figure(path)


def plot_iv_symlog(df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    voltage, current = downsample_arrays(df, ["Vp", "I"])
    plt.figure()
    plt.yscale("symlog", linthresh=SYMLINTHRESH)
    plt.plot(voltage, current, ".", markersize=2)
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.xlabel("Vp (V)")
    plt.ylabel("I (A)")
    plt.title(title)
    save_current_figure(path)


def plot_logabs_current(df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    voltage, current = downsample_arrays(df, ["Vp", "I"])
    plt.figure()
    plt.semilogy(voltage, np.abs(current) + I_FLOOR_ABS, ".", markersize=2)
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.xlabel("Vp (V)")
    plt.ylabel("|I| (A)")
    plt.title(title)
    save_current_figure(path)


def plot_state_vs_time(
    df: pd.DataFrame,
    state_column: str,
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    time, state = downsample_arrays(df, ["time", state_column])
    plt.figure()
    plt.plot(time, state, linewidth=1.2)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xlabel("time (s)")
    plt.ylabel(state_column)
    plt.title(title)
    save_current_figure(path)


def plot_state_vs_voltage(
    df: pd.DataFrame,
    state_column: str,
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    voltage, state = downsample_arrays(df, ["Vp", state_column])
    plt.figure()
    plt.plot(voltage, state, linewidth=1.0)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xlabel("Vp (V)")
    plt.ylabel(state_column)
    plt.title(title)
    save_current_figure(path)


def make_all_case_plots(df: pd.DataFrame, plot_dir: Path, title: str) -> None:
    plot_vcmd_vs_time(df, plot_dir / "input_voltage_vs_time.png", title)
    plot_iv_symlog(df, plot_dir / "iv_symlog.png", title)
    plot_logabs_current(df, plot_dir / "logabsI_vs_Vp.png", title)
    plot_state_vs_time(df, "x", plot_dir / "x_vs_time.png", title)
    plot_state_vs_time(df, "xh", plot_dir / "xh_vs_time.png", title)
    plot_state_vs_voltage(df, "x", plot_dir / "x_vs_Vp.png", title)
    plot_state_vs_voltage(df, "xh", plot_dir / "xh_vs_Vp.png", title)


def plot_overlay_iv(
    records: list[dict[str, Any]],
    path: Path,
    title: str,
    log_abs: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if not records:
        return

    plt.figure()
    if not log_abs:
        plt.yscale("symlog", linthresh=SYMLINTHRESH)

    for record in records:
        voltage, current = downsample_arrays(record["df"], ["Vp", "I"])
        if log_abs:
            plt.semilogy(voltage, np.abs(current) + I_FLOOR_ABS, linewidth=1.0, label=record["label"])
        else:
            plt.plot(voltage, current, linewidth=1.0, label=record["label"])

    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.xlabel("Vp (V)")
    plt.ylabel("|I| (A)" if log_abs else "I (A)")
    plt.title(title)
    plt.legend(fontsize=8)
    save_current_figure(path)


def plot_overlay_state_phase(
    records: list[dict[str, Any]],
    state_column: str,
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if not records:
        return

    plt.figure()
    for record in records:
        phase, state = downsample_arrays(record["df"], ["phase", state_column])
        plt.plot(phase, state, linewidth=1.0, label=record["label"])

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xlabel("normalized phase (0-1)")
    plt.ylabel(state_column)
    plt.title(title)
    plt.legend(fontsize=8)
    save_current_figure(path)


def generate_group_overlays(
    registry: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []

    for mode, groups in registry.items():
        for group, records in groups.items():
            if len(records) < 2:
                continue

            stem = safe_tag(f"{mode}_{group}")
            title = f"{group.replace('_', ' ')} ({mode})"
            outputs = [
                ("iv_symlog", OVERLAY_DIR / f"{stem}_iv_symlog.png"),
                ("logabsI", OVERLAY_DIR / f"{stem}_logabsI.png"),
                ("x_phase", OVERLAY_DIR / f"{stem}_x_vs_phase.png"),
                ("xh_phase", OVERLAY_DIR / f"{stem}_xh_vs_phase.png"),
            ]

            plot_overlay_iv(records, outputs[0][1], title, log_abs=False)
            plot_overlay_iv(records, outputs[1][1], title, log_abs=True)
            plot_overlay_state_phase(records, "x", outputs[2][1], title)
            plot_overlay_state_phase(records, "xh", outputs[3][1], title)

            for plot_type, output_path in outputs:
                manifest.append(
                    {
                        "mode": mode,
                        "group": group,
                        "plot_type": plot_type,
                        "path": str(output_path),
                    }
                )

    return manifest


def generate_frequency_metric_plots(summary_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if summary_df.empty or "frequency_Hz" not in summary_df.columns:
        return

    frequency_df = summary_df[
        (summary_df["group"] == "frequency")
        & pd.to_numeric(summary_df["frequency_Hz"], errors="coerce").notna()
    ].copy()
    if frequency_df.empty:
        return

    for mode in frequency_df["mode"].dropna().unique():
        mode_df = frequency_df[frequency_df["mode"] == mode].copy()
        mode_df["frequency_Hz"] = pd.to_numeric(mode_df["frequency_Hz"], errors="coerce")
        mode_df = mode_df.sort_values("frequency_Hz")

        metric_specs = [
            ("loop_area_abs_path_A_V", "absolute hysteresis path area (A V)"),
            ("x_recovery_abs", "|x_end - x_start|"),
            ("xh_recovery_abs", "|xh_end - xh_start|"),
            ("x_span", "x span"),
            ("Imax_abs", "maximum |I| (A)"),
        ]

        for metric, ylabel in metric_specs:
            if metric not in mode_df.columns:
                continue
            x_values = mode_df["frequency_Hz"].to_numpy(dtype=float)
            y_values = pd.to_numeric(mode_df[metric], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x_values) & np.isfinite(y_values)
            if not np.any(valid):
                continue

            plt.figure()
            plt.plot(x_values[valid], y_values[valid], marker="o")
            plt.xscale("log")
            plt.grid(True, which="both", linestyle="--", alpha=0.35)
            plt.xlabel("frequency (Hz)")
            plt.ylabel(ylabel)
            plt.title(f"{metric} vs frequency ({mode})")
            save_current_figure(
                FREQUENCY_METRIC_DIR / safe_tag(f"{mode}_{metric}_vs_frequency.png")
            )


def plot_measurement_vs_simulation(
    v_meas: np.ndarray,
    i_meas: np.ndarray,
    record: dict[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    vp, current = downsample_arrays(record["df"], ["Vp", "I"])
    plt.figure()
    plt.semilogy(v_meas, np.abs(i_meas) + I_FLOOR_ABS, ".", markersize=2, label="measurement")
    plt.semilogy(vp, np.abs(current) + I_FLOOR_ABS, linewidth=1.1, label="simulation")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.xlabel("voltage (V)")
    plt.ylabel("|I| (A)")
    plt.title(f"Measured vs simulated ({record['mode']})")
    plt.legend()
    save_current_figure(path)


# =============================================================================
# Main workflow
# =============================================================================


def main() -> None:
    ensure_dirs()

    for required_path, description in (
        (CSV_MEAS, "measurement CSV"),
        (TEMPLATE_SRC, "SPICE template"),
        (THETA_PATH, "fitted-parameter CSV"),
        (NGSPICE, "NGSpice executable"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing {description}: {required_path}")

    v_meas, i_meas = read_measurement_csv(CSV_MEAS)
    theta = read_theta_best(THETA_PATH)
    estimated_compliance = estimate_positive_compliance(v_meas, i_meas)

    mode_settings: dict[str, dict[str, float]] = {}
    if RUN_NOLIMIT_MODE:
        mode_settings["NOLIMIT"] = {
            "icomp_pos": 1e30,
            "vcomp": VCOMP_FIXED,
            "rlo": 1e-3,
            "rhi": 1e-3,
        }
    if RUN_LIMIT_MODE:
        mode_settings["LIMIT"] = {
            "icomp_pos": estimated_compliance,
            "vcomp": VCOMP_FIXED,
            "rlo": RLO_FIXED,
            "rhi": RHI_FIXED,
        }
    if not mode_settings:
        raise ValueError("Enable at least one of RUN_NOLIMIT_MODE or RUN_LIMIT_MODE.")

    cases = build_operating_condition_cases(v_meas)
    total_runs = len(cases) * len(mode_settings)

    print(f"[INFO] NGSpice: {NGSPICE}")
    print(f"[INFO] Template: {TEMPLATE_SRC}")
    print(f"[INFO] Parameters: {THETA_PATH}")
    print(f"[INFO] Measurement points: {len(v_meas)}")
    print(f"[INFO] Estimated positive compliance: {estimated_compliance:.6g} A")
    print(f"[INFO] Operating-condition cases: {len(cases)}")
    print(f"[INFO] Planned simulation runs: {total_runs}")

    summary_rows: list[dict[str, Any]] = []
    settings_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    registry: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {} for mode in mode_settings
    }
    measurement_records: list[dict[str, Any]] = []

    successful_runs = 0
    failed_runs = 0

    for case_index, case in enumerate(cases, start=1):
        tstop = compute_case_tstop(case)

        for mode, settings in mode_settings.items():
            case_tag = safe_tag(
                f"{case['group']}_{case['name']}_{mode}_ts{fmt_num(case['time_scale'])}"
            )

            attempts: list[tuple[float, int]] = [(DTMAX_SIM, NGSPICE_TIMEOUT_S)]
            if RETRY_ON_FAILURE_OR_TRUNCATION:
                attempts.extend(
                    (DTMAX_SIM * factor, timeout)
                    for factor, timeout in zip(RETRY_DTMAX_FACTORS, RETRY_TIMEOUTS_S)
                )

            final_df: pd.DataFrame | None = None
            final_dirs: dict[str, Path] | None = None
            final_log_path: Path | None = None
            final_reason = "not_run"
            final_attempt = 0

            for attempt_index, (dtmax, timeout_s) in enumerate(attempts, start=1):
                final_attempt = attempt_index
                df, dirs, log_path, reason = simulate_case(
                    theta=theta,
                    case=case,
                    case_tag=case_tag,
                    tstop=tstop,
                    mode_settings=settings,
                    dtmax=dtmax,
                    timeout_s=timeout_s,
                )
                final_df = df
                final_dirs = dirs
                final_log_path = log_path
                final_reason = reason

                if reason == "ok":
                    break

                print(
                    f"[WARN] {case_tag}: attempt {attempt_index}/{len(attempts)} "
                    f"ended with {reason}"
                )

            if final_reason != "ok" or final_df is None or final_dirs is None:
                failed_runs += 1
                failed_rows.append(
                    {
                        "case_tag": case_tag,
                        "case": case["name"],
                        "group": case["group"],
                        "mode": mode,
                        "reason": final_reason,
                        "attempts": final_attempt,
                        "expected_tstop_s": tstop,
                        "time_end_s": (
                            float(final_df["time"].iloc[-1])
                            if final_df is not None and not final_df.empty
                            else np.nan
                        ),
                        "log_path": str(final_log_path) if final_log_path else "",
                        "log_tail": tail_text(final_log_path, 80),
                    }
                )
                print(f"[FAIL] {case_tag}: {final_reason}")
                continue

            successful_runs += 1
            make_all_case_plots(final_df, final_dirs["plot"], case["label"])
            metrics = compute_case_metrics(final_df, case, tstop)

            row = dict(metrics)
            row.update(
                {
                    "case_tag": case_tag,
                    "case": case["name"],
                    "label": case["label"],
                    "mode": mode,
                    "group": case["group"],
                    "input_style": case["input_style"],
                    "time_scale": case["time_scale"],
                    "tstop": tstop,
                    "vpos": case.get("vpos", np.nan),
                    "vneg": case.get("vneg", np.nan),
                    "repeat_count": case.get("repeat_count", np.nan),
                    "sequence": case.get("sequence", ""),
                    "num_ramps": case["num_ramps"],
                    "pts_per_ramp": case.get("pts_per_ramp", np.nan),
                    "hold_pts": case.get("hold_pts", np.nan),
                    "levels_per_ramp": case.get("levels_per_ramp", np.nan),
                    "hold_pts_per_level": case.get("hold_pts_per_level", np.nan),
                    "sim_dir": str(final_dirs["sim"]),
                    "plot_dir": str(final_dirs["plot"]),
                }
            )
            summary_rows.append(row)

            settings_rows.append(
                {
                    key: row.get(key, np.nan)
                    for key in (
                        "case_tag",
                        "case",
                        "label",
                        "mode",
                        "group",
                        "input_style",
                        "time_scale",
                        "tstop",
                        "frequency_Hz",
                        "vpos",
                        "vneg",
                        "repeat_count",
                        "sequence",
                        "num_ramps",
                        "pts_per_ramp",
                        "hold_pts",
                        "levels_per_ramp",
                        "hold_pts_per_level",
                    )
                }
            )
            branch_rows.extend(
                flatten_branch_metrics(
                    case_tag=case_tag,
                    case=case,
                    mode=mode,
                    metrics=metrics,
                )
            )

            record = {
                "case_tag": case_tag,
                "case": case,
                "df": final_df,
                "label": case["label"],
                "mode": mode,
            }
            registry[mode].setdefault(case["group"], []).append(record)
            if case["group"] == "measurement_trace":
                measurement_records.append(record)

            print(
                f"[OK] {successful_runs + failed_runs}/{total_runs} "
                f"case-group {case_index}/{len(cases)}: {case_tag}"
            )

    summary_df = pd.DataFrame(summary_rows)
    settings_df = pd.DataFrame(settings_rows)
    branch_df = pd.DataFrame(branch_rows)

    summary_df.to_csv(OUT_DIR / "summary_all_cases.csv", index=False)
    settings_df.to_csv(OUT_DIR / "case_settings_table.csv", index=False)
    branch_df.to_csv(OUT_DIR / "branch_metrics_table.csv", index=False)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(OUT_DIR / "failed_cases.csv", index=False)

    overlay_manifest = generate_group_overlays(registry)
    if overlay_manifest:
        pd.DataFrame(overlay_manifest).to_csv(OUT_DIR / "overlay_manifest.csv", index=False)

    generate_frequency_metric_plots(summary_df)

    for record in measurement_records:
        plot_measurement_vs_simulation(
            v_meas,
            i_meas,
            record,
            OVERLAY_DIR / f"measurement_vs_simulation_{record['mode']}.png",
        )

    print("\n[COMPLETE]")
    print(f"Output directory: {OUT_DIR}")
    print(f"Successful simulations: {successful_runs}")
    print(f"Failed simulations: {failed_runs}")
    print("Generated tables:")
    print("  - summary_all_cases.csv")
    print("  - case_settings_table.csv")
    print("  - branch_metrics_table.csv")
    if failed_rows:
        print("  - failed_cases.csv")
    if overlay_manifest:
        print("  - overlay_manifest.csv")


if __name__ == "__main__":
    main()
檔案庫
/
analyze_operating_condition_sweeps.py


"""
analyze_operating_condition_sweeps.py
=====================================
Unified NGSpice analysis workflow for a fitted memristive-device model.

This script combines the original frequency-sweep and voltage-sweep analyzers
into one operating-condition study. Model parameters remain fixed while the
external stimulus is changed.

Supported analyses
------------------
1. Frequency / sweep-rate sweep
2. Positive-voltage amplitude sweep
3. Negative-voltage amplitude sweep
4. Multi-step negative-voltage sequences
5. Repeated negative-voltage cycles
6. Optional staircase-input comparison
7. Optional measurement-trace replay

Expected project layout
-----------------------
project_root/
├── analyze_operating_condition_sweeps.py
├── data/
│   └── DC-IV.csv
└── results/
    └── fit/
        ├── fitdeck_embedded.cir
        └── theta_best.csv

The SPICE template is expected to contain the same @PLACEHOLDER@ tokens used by
the fitting workflow, including @PWL_INLINE@, @SIMOUT@, @TSTEP@, @DTMAX@,
@TSTOP@, fixed model knobs, compliance knobs, and fitted parameter names.

Main outputs
------------
results/operating_condition_sweeps/
├── fitdeck_embedded.cir
├── cases/<case_tag>/{decks,logs,sims,plots}/...
├── overlays/*.png
├── metrics_vs_frequency/*.png
├── summary_all_cases.csv
├── case_settings_table.csv
├── branch_metrics_table.csv
├── failed_cases.csv                 # only when failures occur
└── overlay_manifest.csv
"""

from __future__ import annotations

import math
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

OUT_DIR = BASE_DIR / "results" / "operating_condition_sweeps"
TEMPLATE_SNAPSHOT = OUT_DIR / "fitdeck_embedded.cir"
CASES_DIR = OUT_DIR / "cases"
OVERLAY_DIR = OUT_DIR / "overlays"
FREQUENCY_METRIC_DIR = OUT_DIR / "metrics_vs_frequency"


# =============================================================================
# Simulation and numerical settings
# =============================================================================

TSTOP_BASE = 10.0
DTMAX_SIM = 2e-4

PRINT_DIV = 6
PRINT_MIN = 2e-5
PRINT_MAX = 4e-3

NGSPICE_TIMEOUT_S = 180
RETRY_ON_FAILURE_OR_TRUNCATION = True
RETRY_DTMAX_FACTORS = [0.5, 0.25]
RETRY_TIMEOUTS_S = [240, 360]
INCOMPLETE_TIME_FRAC = 0.995

# Preserve approximately the same time per voltage ramp when a waveform has
# more than the four ramps in 0 -> +V -> 0 -> -V -> 0.
NORMALIZE_TSTOP_BY_RAMP_COUNT = True
BASELINE_RAMP_COUNT = 4


# =============================================================================
# Fixed model knobs used by the fitted SPICE template
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
# Analysis selection
# =============================================================================

ENABLE_MEASUREMENT_TRACE = True
ENABLE_FREQUENCY_SWEEP = True
ENABLE_VOLTAGE_SWEEP = True
ENABLE_MULTI_NEGATIVE_SWEEP = True
ENABLE_REPEAT_CYCLE_SWEEP = True
ENABLE_STAIRCASE_SWEEP = False


# =============================================================================
# Input-waveform settings
# =============================================================================

# Common waveform resolution for continuous ramps.
PTS_PER_RAMP = 160
HOLD_PTS = 0

# ----- Frequency sweep -----
# frequency_Hz = 1 / (TSTOP_BASE * time_scale)
# With TSTOP_BASE = 10 s:
#   0.25 -> 0.4 Hz
#   0.50 -> 0.2 Hz
#   1.00 -> 0.1 Hz
#   2.00 -> 0.05 Hz
#   4.00 -> 0.025 Hz
FREQUENCY_TIME_SCALE_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]
FREQUENCY_VPOS = 10.0
FREQUENCY_VNEG = -4.0

# ----- Voltage-amplitude sweep -----
VOLTAGE_SWEEP_TIME_SCALE = 1.0

VPOS_LIST = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
FIXED_VNEG_FOR_VPOS_SWEEP = -4.0

VNEG_LIST = [-2.0, -3.0, -4.0, -5.0, -6.0, -8.0, -10.0]
FIXED_VPOS_FOR_VNEG_SWEEP = 10.0

# ----- Multi-step negative-voltage sequences -----
MULTI_NEGATIVE_VPOS_LIST = [6.0, 8.0, 10.0]
MULTI_NEGATIVE_SEQUENCE_LIST = [
    [-2.0, -4.0, -6.0, -8.0, -10.0],
    [-10.0, -8.0, -6.0, -4.0, -2.0],
]
MULTI_NEGATIVE_TIME_SCALE = 1.0

# ----- Repeated negative cycles -----
REPEAT_VPOS_LIST = [10.0]
REPEAT_VNEG_LIST = [-10.0]
REPEAT_COUNT_LIST = [1, 2, 3, 5]
REPEAT_TIME_SCALE = 1.0

# ----- Optional staircase comparison -----
STAIRCASE_VPOS = 10.0
STAIRCASE_VNEG = -4.0
LEVELS_PER_RAMP_LIST = [3, 17, 256]
HOLD_PTS_PER_LEVEL = 4
STAIRCASE_TIME_SCALE = 1.0


# =============================================================================
# Plot and metric settings
# =============================================================================

I_FLOOR_ABS = 3e-10
SYMLINTHRESH = 1e-9
MAX_PLOT_POINTS = 24000
BRANCH_SAMPLE_V_LIST = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
MAX_BRANCH_V_ERROR = 0.25


# =============================================================================
# Fitted parameters expected in theta_best.csv
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
# Basic utilities
# =============================================================================


def ensure_dirs() -> None:
    for directory in (OUT_DIR, CASES_DIR, OVERLAY_DIR, FREQUENCY_METRIC_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE_SRC.exists():
        raise FileNotFoundError(f"Missing SPICE template: {TEMPLATE_SRC}")

    TEMPLATE_SNAPSHOT.write_text(
        TEMPLATE_SRC.read_text(encoding="utf-8", errors="ignore"),
        encoding="utf-8",
    )


def safe_tag(value: str) -> str:
    value = str(value).strip().replace(" ", "_")
    value = value.replace("/", "_").replace("\\", "_").replace(":", "_")
    value = re.sub(r"[^A-Za-z0-9_.+\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def fmt_num(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(number):
        return "na"
    if abs(number - round(number)) < 1e-12:
        return str(int(round(number)))
    return f"{number:g}"


def make_case_dirs(case_tag: str) -> dict[str, Path]:
    case_dir = CASES_DIR / case_tag
    dirs = {
        "case": case_dir,
        "deck": case_dir / "decks",
        "log": case_dir / "logs",
        "sim": case_dir / "sims",
        "plot": case_dir / "plots",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def read_measurement_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, engine="python")
    if df.shape[1] < 2:
        raise ValueError("DC-IV.csv must contain at least two columns: voltage and current.")

    voltage = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    current = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    valid = voltage.notna() & current.notna()

    v = voltage[valid].to_numpy(dtype=float)
    i = current[valid].to_numpy(dtype=float)
    if len(v) < 10:
        raise ValueError("Too few numeric rows remain after cleaning DC-IV.csv.")
    return v, i


def read_theta_best(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if not {"param", "value"}.issubset(df.columns):
        raise ValueError("theta_best.csv must contain columns named 'param' and 'value'.")

    theta: dict[str, float] = {}
    for raw_name, raw_value in zip(df["param"], df["value"]):
        name = str(raw_name).strip()
        name = ALIASES.get(name, name)
        theta[name] = float(raw_value)

    missing = sorted(REQUIRED_PARAMS.difference(theta))
    if missing:
        raise ValueError(
            f"theta_best.csv is missing required parameters: {missing}\n"
            f"Loaded parameters: {sorted(theta)}"
        )
    return theta


def estimate_positive_compliance(voltage: np.ndarray, current: np.ndarray) -> float:
    mask = voltage > 0.5
    if not np.any(mask):
        return 1e-3
    estimate = float(np.quantile(np.abs(current[mask]), 0.98))
    return float(np.clip(estimate, 1e-6, 5e-2))


def pick_tstep_print(num_points: int, tstop: float) -> float:
    if num_points <= 1:
        return 1e-3
    dt_command = float(tstop) / max(num_points - 1, 1)
    return float(np.clip(dt_command / PRINT_DIV, PRINT_MIN, PRINT_MAX))


def make_time_vector(num_points: int, tstop: float) -> np.ndarray:
    if num_points < 2:
        return np.array([0.0])
    return np.linspace(0.0, float(tstop), int(num_points))


def pwl_inline_from_tv(time: np.ndarray, voltage: np.ndarray, pairs_per_line: int = 8) -> str:
    pairs = [f"{t:.12g} {v:.12g}" for t, v in zip(time, voltage)]
    return "\n".join(
        "+ " + " ".join(pairs[start : start + pairs_per_line])
        for start in range(0, len(pairs), pairs_per_line)
    )


def tail_text(path: Path | None, n_lines: int = 100) -> str:
    if path is None or not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except OSError:
        return ""


def downsample_arrays(
    df: pd.DataFrame,
    columns: Iterable[str],
    max_points: int = MAX_PLOT_POINTS,
) -> list[np.ndarray]:
    columns = list(columns)
    if len(df) <= max_points:
        return [df[column].to_numpy(dtype=float) for column in columns]
    indices = np.linspace(0, len(df) - 1, max_points).astype(int)
    return [df[column].to_numpy(dtype=float)[indices] for column in columns]


# =============================================================================
# Waveform construction
# =============================================================================


def make_voltage_sequence(
    nodes: Iterable[float],
    pts_per_ramp: int = PTS_PER_RAMP,
    hold_pts: int = HOLD_PTS,
) -> np.ndarray:
    """Create a continuous, piecewise-linear command waveform."""
    node_list = [float(value) for value in nodes]
    if len(node_list) < 2:
        raise ValueError("At least two voltage nodes are required.")

    pts_per_ramp = max(2, int(pts_per_ramp))
    repeat_count = max(1, int(hold_pts) + 1)

    segments: list[np.ndarray] = []
    for index, (start, stop) in enumerate(zip(node_list[:-1], node_list[1:])):
        segment = np.linspace(start, stop, pts_per_ramp)
        if index > 0:
            segment = segment[1:]
        if repeat_count > 1:
            segment = np.repeat(segment, repeat_count)
        segments.append(segment)

    return np.concatenate(segments)


def make_staircase_sequence(
    nodes: Iterable[float],
    levels_per_ramp: int,
    hold_pts_per_level: int,
) -> np.ndarray:
    """Create a quantized staircase waveform with held voltage levels."""
    node_list = [float(value) for value in nodes]
    if len(node_list) < 2:
        raise ValueError("At least two voltage nodes are required.")

    levels_per_ramp = max(2, int(levels_per_ramp))
    hold_pts_per_level = max(1, int(hold_pts_per_level))

    segments: list[np.ndarray] = []
    for index, (start, stop) in enumerate(zip(node_list[:-1], node_list[1:])):
        levels = np.linspace(start, stop, levels_per_ramp)
        if index > 0:
            levels = levels[1:]
        segments.append(np.repeat(levels, hold_pts_per_level))

    return np.concatenate(segments)


def build_case(
    *,
    name: str,
    label: str,
    group: str,
    nodes: list[float] | None,
    voltage_command: np.ndarray,
    time_scale: float,
    input_style: str = "linear",
    **metadata: Any,
) -> dict[str, Any]:
    num_ramps = len(nodes) - 1 if nodes is not None else BASELINE_RAMP_COUNT
    case = {
        "name": name,
        "label": label,
        "group": group,
        "nodes": nodes,
        "V_cmd": np.asarray(voltage_command, dtype=float),
        "time_scale": float(time_scale),
        "input_style": input_style,
        "num_ramps": int(num_ramps),
        "pts_per_ramp": metadata.pop("pts_per_ramp", np.nan),
        "hold_pts": metadata.pop("hold_pts", np.nan),
        "levels_per_ramp": metadata.pop("levels_per_ramp", np.nan),
        "hold_pts_per_level": metadata.pop("hold_pts_per_level", np.nan),
    }
    case.update(metadata)
    return case


def build_operating_condition_cases(v_meas: np.ndarray) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    if ENABLE_MEASUREMENT_TRACE:
        cases.append(
            build_case(
                name="measurement_trace",
                label="measurement-driven waveform",
                group="measurement_trace",
                nodes=None,
                voltage_command=v_meas,
                time_scale=1.0,
                input_style="measurement",
                vpos=float(np.max(v_meas)),
                vneg=float(np.min(v_meas)),
            )
        )

    if ENABLE_FREQUENCY_SWEEP:
        nodes = [0.0, FREQUENCY_VPOS, 0.0, FREQUENCY_VNEG, 0.0]
        command = make_voltage_sequence(nodes, PTS_PER_RAMP, HOLD_PTS)
        for time_scale in FREQUENCY_TIME_SCALE_LIST:
            period = TSTOP_BASE * float(time_scale)
            frequency = 1.0 / period
            cases.append(
                build_case(
                    name=f"frequency_{fmt_num(frequency)}Hz",
                    label=f"f = {fmt_num(frequency)} Hz, T = {fmt_num(period)} s",
                    group="frequency",
                    nodes=nodes,
                    voltage_command=command,
                    time_scale=time_scale,
                    pts_per_ramp=PTS_PER_RAMP,
                    hold_pts=HOLD_PTS,
                    vpos=FREQUENCY_VPOS,
                    vneg=FREQUENCY_VNEG,
                    frequency_Hz=frequency,
                )
            )

    if ENABLE_VOLTAGE_SWEEP:
        for vpos in VPOS_LIST:
            nodes = [0.0, float(vpos), 0.0, FIXED_VNEG_FOR_VPOS_SWEEP, 0.0]
            cases.append(
                build_case(
                    name=f"amp_vpos_{fmt_num(vpos)}V",
                    label=f"Vpos = {fmt_num(vpos)} V",
                    group="amp_vpos",
                    nodes=nodes,
                    voltage_command=make_voltage_sequence(nodes, PTS_PER_RAMP, HOLD_PTS),
                    time_scale=VOLTAGE_SWEEP_TIME_SCALE,
                    pts_per_ramp=PTS_PER_RAMP,
                    hold_pts=HOLD_PTS,
                    vpos=float(vpos),
                    vneg=FIXED_VNEG_FOR_VPOS_SWEEP,
                )
            )

        for vneg in VNEG_LIST:
            nodes = [0.0, FIXED_VPOS_FOR_VNEG_SWEEP, 0.0, float(vneg), 0.0]
            cases.append(
                build_case(
                    name=f"amp_vneg_{fmt_num(vneg)}V",
                    label=f"Vneg = {fmt_num(vneg)} V",
                    group="amp_vneg",
                    nodes=nodes,
                    voltage_command=make_voltage_sequence(nodes, PTS_PER_RAMP, HOLD_PTS),
                    time_scale=VOLTAGE_SWEEP_TIME_SCALE,
                    pts_per_ramp=PTS_PER_RAMP,
                    hold_pts=HOLD_PTS,
                    vpos=FIXED_VPOS_FOR_VNEG_SWEEP,
                    vneg=float(vneg),
                )
            )

    if ENABLE_MULTI_NEGATIVE_SWEEP:
        for vpos in MULTI_NEGATIVE_VPOS_LIST:
            for sequence in MULTI_NEGATIVE_SEQUENCE_LIST:
                nodes = [0.0, float(vpos), 0.0]
                for vneg in sequence:
                    nodes.extend([float(vneg), 0.0])
                sequence_tag = "_".join(fmt_num(value) for value in sequence)
                cases.append(
                    build_case(
                        name=f"multi_vpos_{fmt_num(vpos)}_seq_{sequence_tag}",
                        label=(
                            f"Vpos = {fmt_num(vpos)} V, "
                            f"sequence = {' -> '.join(fmt_num(value) for value in sequence)} V"
                        ),
                        group="multi_negative",
                        nodes=nodes,
                        voltage_command=make_voltage_sequence(nodes, PTS_PER_RAMP, HOLD_PTS),
                        time_scale=MULTI_NEGATIVE_TIME_SCALE,
                        pts_per_ramp=PTS_PER_RAMP,
                        hold_pts=HOLD_PTS,
                        vpos=float(vpos),
                        vneg=float(np.min(sequence)),
                        sequence=sequence_tag,
                    )
                )

    if ENABLE_REPEAT_CYCLE_SWEEP:
        for vpos in REPEAT_VPOS_LIST:
            for vneg in REPEAT_VNEG_LIST:
                for repeat_count in REPEAT_COUNT_LIST:
                    nodes = [0.0, float(vpos), 0.0]
                    for _ in range(int(repeat_count)):
                        nodes.extend([float(vneg), 0.0])
                    cases.append(
                        build_case(
                            name=(
                                f"repeat_vpos_{fmt_num(vpos)}_"
                                f"vneg_{fmt_num(vneg)}_x{repeat_count}"
                            ),
                            label=(
                                f"Vneg = {fmt_num(vneg)} V, "
                                f"repeat = {repeat_count}"
                            ),
                            group="repeat_negative",
                            nodes=nodes,
                            voltage_command=make_voltage_sequence(nodes, PTS_PER_RAMP, HOLD_PTS),
                            time_scale=REPEAT_TIME_SCALE,
                            pts_per_ramp=PTS_PER_RAMP,
                            hold_pts=HOLD_PTS,
                            vpos=float(vpos),
                            vneg=float(vneg),
                            repeat_count=int(repeat_count),
                        )
                    )

    if ENABLE_STAIRCASE_SWEEP:
        nodes = [0.0, STAIRCASE_VPOS, 0.0, STAIRCASE_VNEG, 0.0]
        for levels_per_ramp in LEVELS_PER_RAMP_LIST:
            cases.append(
                build_case(
                    name=f"staircase_{levels_per_ramp}_levels_per_ramp",
                    label=f"{levels_per_ramp} levels/ramp",
                    group="staircase",
                    nodes=nodes,
                    voltage_command=make_staircase_sequence(
                        nodes,
                        levels_per_ramp=levels_per_ramp,
                        hold_pts_per_level=HOLD_PTS_PER_LEVEL,
                    ),
                    time_scale=STAIRCASE_TIME_SCALE,
                    input_style="staircase",
                    levels_per_ramp=levels_per_ramp,
                    hold_pts_per_level=HOLD_PTS_PER_LEVEL,
                    vpos=STAIRCASE_VPOS,
                    vneg=STAIRCASE_VNEG,
                )
            )

    return cases


def compute_case_tstop(case: dict[str, Any]) -> float:
    base_tstop = TSTOP_BASE * float(case["time_scale"])
    if case["group"] == "frequency":
        return base_tstop
    if case["input_style"] == "measurement":
        return base_tstop
    if NORMALIZE_TSTOP_BY_RAMP_COUNT:
        return base_tstop * float(case["num_ramps"]) / BASELINE_RAMP_COUNT
    return base_tstop


# =============================================================================
# NGSpice execution
# =============================================================================


def run_ngspice(deck_path: Path, log_path: Path, cwd: Path, timeout_s: int) -> int:
    command = [str(NGSPICE), "-b", "-o", str(log_path), str(deck_path)]
    try:
        completed = subprocess.run(command, cwd=str(cwd), timeout=int(timeout_s), check=False)
        return int(completed.returncode)
    except subprocess.TimeoutExpired:
        return 124
    except OSError as exc:
        log_path.write_text(f"Failed to start NGSpice: {exc}\n", encoding="utf-8")
        return 127


def load_wrdata(path: Path) -> pd.DataFrame:
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise ValueError(f"Expected at least 6 wrdata columns, found {data.shape[1]} in {path}")

    df = pd.DataFrame(
        {
            "time": data[:, 0],
            "Vcmd": data[:, 1],
            "Vp": data[:, 2],
            "I": data[:, 3],
            "x": data[:, 4],
            "xh": data[:, 5],
        }
    )
    final_time = max(float(df["time"].iloc[-1]), 1e-30)
    df["phase"] = df["time"] / final_time
    return df


def is_simulation_complete(df: pd.DataFrame, tstop: float) -> bool:
    if df.empty:
        return False
    return float(df["time"].iloc[-1]) >= INCOMPLETE_TIME_FRAC * float(tstop)


def build_deck_text(
    *,
    theta: dict[str, float],
    voltage_command: np.ndarray,
    simulation_output: Path,
    tstop: float,
    dtmax: float,
    mode_settings: dict[str, float],
) -> str:
    time = make_time_vector(len(voltage_command), tstop)
    pwl_inline = pwl_inline_from_tv(time, voltage_command)
    tstep_print = pick_tstep_print(len(voltage_command), tstop)

    icomp_pos = float(mode_settings["icomp_pos"])
    islope = max(1e-12, ISLOPE_REL * icomp_pos)

    replacements: dict[str, str] = {
        "@PWL_INLINE@": pwl_inline,
        "@SIMOUT@": str(simulation_output),
        "@TSTEP@": f"{tstep_print:.12g}",
        "@DTMAX@": f"{float(dtmax):.12g}",
        "@TSTOP@": f"{float(tstop):.12g}",
        "@KSW@": f"{KSW_FIXED:.12g}",
        "@RH0@": f"{RH0_FIXED:.12g}",
        "@RH_MIN@": f"{RH_MIN_FIXED:.12g}",
        "@RH_MAX@": f"{RH_MAX_FIXED:.12g}",
        "@VSLOPE@": f"{VSLOPE_FIXED:.12g}",
        "@ICOMP_POS@": f"{icomp_pos:.12g}",
        "@VCOMP@": f"{float(mode_settings['vcomp']):.12g}",
        "@RLO@": f"{float(mode_settings['rlo']):.12g}",
        "@RHI@": f"{float(mode_settings['rhi']):.12g}",
        "@ISLOPE@": f"{islope:.12g}",
        "@VSLOPE_POS@": f"{VSLOPE_POS_FIXED:.12g}",
    }

    for name, value in theta.items():
        replacements[f"@{name}@"] = f"{float(value):.12g}"

    text = TEMPLATE_SNAPSHOT.read_text(encoding="utf-8", errors="ignore")
    for token, replacement in replacements.items():
        text = text.replace(token, replacement)

    leftovers = sorted(set(re.findall(r"@[A-Za-z0-9_]+@", text)))
    if leftovers:
        raise RuntimeError(f"Unreplaced SPICE-template placeholders: {leftovers}")
    return text


def simulate_case(
    *,
    theta: dict[str, float],
    case: dict[str, Any],
    case_tag: str,
    tstop: float,
    mode_settings: dict[str, float],
    dtmax: float,
    timeout_s: int,
) -> tuple[pd.DataFrame | None, dict[str, Path], Path, str]:
    dirs = make_case_dirs(case_tag)
    deck_path = dirs["deck"] / f"{case_tag}.cir"
    log_path = dirs["log"] / f"{case_tag}.log"
    sim_path = dirs["sim"] / f"{case_tag}.dat"

    deck_text = build_deck_text(
        theta=theta,
        voltage_command=case["V_cmd"],
        simulation_output=sim_path,
        tstop=tstop,
        dtmax=dtmax,
        mode_settings=mode_settings,
    )
    deck_path.write_text(deck_text, encoding="utf-8")

    return_code = run_ngspice(deck_path, log_path, dirs["case"], timeout_s)
    if return_code != 0:
        return None, dirs, log_path, f"ngspice_return_code_{return_code}"
    if not sim_path.exists():
        return None, dirs, log_path, "missing_simulation_output"

    try:
        df = load_wrdata(sim_path)
    except (OSError, ValueError) as exc:
        return None, dirs, log_path, f"invalid_simulation_output: {exc}"

    df.to_csv(dirs["sim"] / f"{case_tag}_sim.csv", index=False)
    if not is_simulation_complete(df, tstop):
        return df, dirs, log_path, "truncated_simulation"
    return df, dirs, log_path, "ok"


# =============================================================================
# Metrics
# =============================================================================


def fill_zero_sign(values: np.ndarray) -> np.ndarray:
    signs = np.asarray(values, dtype=float).copy()
    if len(signs) == 0:
        return signs

    for index in range(1, len(signs)):
        if signs[index] == 0:
            signs[index] = signs[index - 1]
    for index in range(len(signs) - 2, -1, -1):
        if signs[index] == 0:
            signs[index] = signs[index + 1]
    return signs


def sweep_direction(voltage_command: np.ndarray) -> np.ndarray:
    voltage_command = np.asarray(voltage_command, dtype=float)
    if len(voltage_command) <= 1:
        return np.zeros_like(voltage_command)
    signs = fill_zero_sign(np.sign(np.diff(voltage_command)))
    return np.r_[signs, signs[-1]]


def nearest_masked_index(
    mask: np.ndarray,
    target_voltage: float,
    voltage: np.ndarray,
) -> int | None:
    indices = np.where(mask)[0]
    if indices.size == 0:
        return None
    best = int(indices[np.argmin(np.abs(voltage[indices] - target_voltage))])
    if abs(float(voltage[best]) - float(target_voltage)) > MAX_BRANCH_V_ERROR:
        return None
    return best


def safe_logabs(value: float) -> float:
    return float(np.log10(abs(float(value)) + I_FLOOR_ABS))


def loop_metrics(voltage: np.ndarray, current: np.ndarray) -> dict[str, float]:
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(voltage) < 2:
        return {"loop_area_signed_A_V": np.nan, "loop_area_abs_path_A_V": np.nan}

    delta_v = np.diff(voltage)
    current_mid = 0.5 * (current[:-1] + current[1:])
    return {
        "loop_area_signed_A_V": float(np.sum(current_mid * delta_v)),
        "loop_area_abs_path_A_V": float(np.sum(np.abs(current_mid * delta_v))),
    }


def compute_case_metrics(df: pd.DataFrame, case: dict[str, Any], tstop: float) -> dict[str, Any]:
    time = df["time"].to_numpy(dtype=float)
    vcmd = df["Vcmd"].to_numpy(dtype=float)
    vp = df["Vp"].to_numpy(dtype=float)
    current = df["I"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    xh = df["xh"].to_numpy(dtype=float)

    metrics: dict[str, Any] = {
        "N": int(len(df)),
        "time_end_s": float(time[-1]),
        "expected_tstop_s": float(tstop),
        "frequency_Hz": case.get("frequency_Hz", np.nan),
        "Vcmd_min": float(np.min(vcmd)),
        "Vcmd_max": float(np.max(vcmd)),
        "Vp_min": float(np.min(vp)),
        "Vp_max": float(np.max(vp)),
        "Imax_abs": float(np.max(np.abs(current))),
        "I_pos_max": float(np.max(current)),
        "I_neg_min": float(np.min(current)),
        "x_start": float(x[0]),
        "x_end": float(x[-1]),
        "x_recovery_signed": float(x[-1] - x[0]),
        "x_recovery_abs": float(abs(x[-1] - x[0])),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "x_span": float(np.max(x) - np.min(x)),
        "xh_start": float(xh[0]),
        "xh_end": float(xh[-1]),
        "xh_recovery_signed": float(xh[-1] - xh[0]),
        "xh_recovery_abs": float(abs(xh[-1] - xh[0])),
        "xh_min": float(np.min(xh)),
        "xh_max": float(np.max(xh)),
        "xh_span": float(np.max(xh) - np.min(xh)),
    }
    metrics.update(loop_metrics(vp, current))

    direction = sweep_direction(vcmd)
    positive_up = (vp >= 0) & (direction > 0)
    positive_down = (vp >= 0) & (direction < 0)
    negative_down = (vp <= 0) & (direction < 0)
    negative_up = (vp <= 0) & (direction > 0)

    for target in BRANCH_SAMPLE_V_LIST:
        tag = str(target).replace(".", "p")
        indices = {
            "pos_up": nearest_masked_index(positive_up, target, vp),
            "pos_down": nearest_masked_index(positive_down, target, vp),
            "neg_down": nearest_masked_index(negative_down, -target, vp),
            "neg_up": nearest_masked_index(negative_up, -target, vp),
        }

        for branch_name, index in indices.items():
            metrics[f"I_{branch_name}_at_{tag}V"] = (
                float(current[index]) if index is not None else np.nan
            )

        pos_up_idx = indices["pos_up"]
        pos_down_idx = indices["pos_down"]
        neg_down_idx = indices["neg_down"]
        neg_up_idx = indices["neg_up"]

        if pos_up_idx is not None and pos_down_idx is not None:
            metrics[f"branch_sep_pos_absI_at_{tag}V"] = float(
                abs(current[pos_up_idx] - current[pos_down_idx])
            )
            metrics[f"branch_sep_pos_logdec_at_{tag}V"] = float(
                abs(safe_logabs(current[pos_up_idx]) - safe_logabs(current[pos_down_idx]))
            )
        else:
            metrics[f"branch_sep_pos_absI_at_{tag}V"] = np.nan
            metrics[f"branch_sep_pos_logdec_at_{tag}V"] = np.nan

        if neg_down_idx is not None and neg_up_idx is not None:
            metrics[f"branch_sep_neg_absI_at_{tag}V"] = float(
                abs(current[neg_down_idx] - current[neg_up_idx])
            )
            metrics[f"branch_sep_neg_logdec_at_{tag}V"] = float(
                abs(safe_logabs(current[neg_down_idx]) - safe_logabs(current[neg_up_idx]))
            )
        else:
            metrics[f"branch_sep_neg_absI_at_{tag}V"] = np.nan
            metrics[f"branch_sep_neg_logdec_at_{tag}V"] = np.nan

    return metrics


def flatten_branch_metrics(
    *,
    case_tag: str,
    case: dict[str, Any],
    mode: str,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in BRANCH_SAMPLE_V_LIST:
        tag = str(target).replace(".", "p")
        rows.append(
            {
                "case_tag": case_tag,
                "case": case["name"],
                "mode": mode,
                "group": case["group"],
                "sample_absV": float(target),
                "I_pos_up": metrics.get(f"I_pos_up_at_{tag}V", np.nan),
                "I_pos_down": metrics.get(f"I_pos_down_at_{tag}V", np.nan),
                "pos_abs_sep": metrics.get(f"branch_sep_pos_absI_at_{tag}V", np.nan),
                "pos_logdec_sep": metrics.get(
                    f"branch_sep_pos_logdec_at_{tag}V", np.nan
                ),
                "I_neg_down": metrics.get(f"I_neg_down_at_{tag}V", np.nan),
                "I_neg_up": metrics.get(f"I_neg_up_at_{tag}V", np.nan),
                "neg_abs_sep": metrics.get(f"branch_sep_neg_absI_at_{tag}V", np.nan),
                "neg_logdec_sep": metrics.get(
                    f"branch_sep_neg_logdec_at_{tag}V", np.nan
                ),
            }
        )
    return rows


# =============================================================================
# Plotting
# =============================================================================


def save_current_figure(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_vcmd_vs_time(df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    time, voltage = downsample_arrays(df, ["time", "Vcmd"])
    plt.figure()
    plt.plot(time, voltage, linewidth=1.2)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xlabel("time (s)")
    plt.ylabel("Vcmd (V)")
    plt.title(title)
    save_current_figure(path)


def plot_iv_symlog(df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    voltage, current = downsample_arrays(df, ["Vp", "I"])
    plt.figure()
    plt.yscale("symlog", linthresh=SYMLINTHRESH)
    plt.plot(voltage, current, ".", markersize=2)
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.xlabel("Vp (V)")
    plt.ylabel("I (A)")
    plt.title(title)
    save_current_figure(path)


def plot_logabs_current(df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    voltage, current = downsample_arrays(df, ["Vp", "I"])
    plt.figure()
    plt.semilogy(voltage, np.abs(current) + I_FLOOR_ABS, ".", markersize=2)
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.xlabel("Vp (V)")
    plt.ylabel("|I| (A)")
    plt.title(title)
    save_current_figure(path)


def plot_state_vs_time(
    df: pd.DataFrame,
    state_column: str,
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    time, state = downsample_arrays(df, ["time", state_column])
    plt.figure()
    plt.plot(time, state, linewidth=1.2)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xlabel("time (s)")
    plt.ylabel(state_column)
    plt.title(title)
    save_current_figure(path)


def plot_state_vs_voltage(
    df: pd.DataFrame,
    state_column: str,
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    voltage, state = downsample_arrays(df, ["Vp", state_column])
    plt.figure()
    plt.plot(voltage, state, linewidth=1.0)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xlabel("Vp (V)")
    plt.ylabel(state_column)
    plt.title(title)
    save_current_figure(path)


def make_all_case_plots(df: pd.DataFrame, plot_dir: Path, title: str) -> None:
    plot_vcmd_vs_time(df, plot_dir / "input_voltage_vs_time.png", title)
    plot_iv_symlog(df, plot_dir / "iv_symlog.png", title)
    plot_logabs_current(df, plot_dir / "logabsI_vs_Vp.png", title)
    plot_state_vs_time(df, "x", plot_dir / "x_vs_time.png", title)
    plot_state_vs_time(df, "xh", plot_dir / "xh_vs_time.png", title)
    plot_state_vs_voltage(df, "x", plot_dir / "x_vs_Vp.png", title)
    plot_state_vs_voltage(df, "xh", plot_dir / "xh_vs_Vp.png", title)


def plot_overlay_iv(
    records: list[dict[str, Any]],
    path: Path,
    title: str,
    log_abs: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if not records:
        return

    plt.figure()
    if not log_abs:
        plt.yscale("symlog", linthresh=SYMLINTHRESH)

    for record in records:
        voltage, current = downsample_arrays(record["df"], ["Vp", "I"])
        if log_abs:
            plt.semilogy(voltage, np.abs(current) + I_FLOOR_ABS, linewidth=1.0, label=record["label"])
        else:
            plt.plot(voltage, current, linewidth=1.0, label=record["label"])

    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.xlabel("Vp (V)")
    plt.ylabel("|I| (A)" if log_abs else "I (A)")
    plt.title(title)
    plt.legend(fontsize=8)
    save_current_figure(path)


def plot_overlay_state_phase(
    records: list[dict[str, Any]],
    state_column: str,
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if not records:
        return

    plt.figure()
    for record in records:
        phase, state = downsample_arrays(record["df"], ["phase", state_column])
        plt.plot(phase, state, linewidth=1.0, label=record["label"])

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xlabel("normalized phase (0-1)")
    plt.ylabel(state_column)
    plt.title(title)
    plt.legend(fontsize=8)
    save_current_figure(path)


def generate_group_overlays(
    registry: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []

    for mode, groups in registry.items():
        for group, records in groups.items():
            if len(records) < 2:
                continue

            stem = safe_tag(f"{mode}_{group}")
            title = f"{group.replace('_', ' ')} ({mode})"
            outputs = [
                ("iv_symlog", OVERLAY_DIR / f"{stem}_iv_symlog.png"),
                ("logabsI", OVERLAY_DIR / f"{stem}_logabsI.png"),
                ("x_phase", OVERLAY_DIR / f"{stem}_x_vs_phase.png"),
                ("xh_phase", OVERLAY_DIR / f"{stem}_xh_vs_phase.png"),
            ]

            plot_overlay_iv(records, outputs[0][1], title, log_abs=False)
            plot_overlay_iv(records, outputs[1][1], title, log_abs=True)
            plot_overlay_state_phase(records, "x", outputs[2][1], title)
            plot_overlay_state_phase(records, "xh", outputs[3][1], title)

            for plot_type, output_path in outputs:
                manifest.append(
                    {
                        "mode": mode,
                        "group": group,
                        "plot_type": plot_type,
                        "path": str(output_path),
                    }
                )

    return manifest


def generate_frequency_metric_plots(summary_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if summary_df.empty or "frequency_Hz" not in summary_df.columns:
        return

    frequency_df = summary_df[
        (summary_df["group"] == "frequency")
        & pd.to_numeric(summary_df["frequency_Hz"], errors="coerce").notna()
    ].copy()
    if frequency_df.empty:
        return

    for mode in frequency_df["mode"].dropna().unique():
        mode_df = frequency_df[frequency_df["mode"] == mode].copy()
        mode_df["frequency_Hz"] = pd.to_numeric(mode_df["frequency_Hz"], errors="coerce")
        mode_df = mode_df.sort_values("frequency_Hz")

        metric_specs = [
            ("loop_area_abs_path_A_V", "absolute hysteresis path area (A V)"),
            ("x_recovery_abs", "|x_end - x_start|"),
            ("xh_recovery_abs", "|xh_end - xh_start|"),
            ("x_span", "x span"),
            ("Imax_abs", "maximum |I| (A)"),
        ]

        for metric, ylabel in metric_specs:
            if metric not in mode_df.columns:
                continue
            x_values = mode_df["frequency_Hz"].to_numpy(dtype=float)
            y_values = pd.to_numeric(mode_df[metric], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x_values) & np.isfinite(y_values)
            if not np.any(valid):
                continue

            plt.figure()
            plt.plot(x_values[valid], y_values[valid], marker="o")
            plt.xscale("log")
            plt.grid(True, which="both", linestyle="--", alpha=0.35)
            plt.xlabel("frequency (Hz)")
            plt.ylabel(ylabel)
            plt.title(f"{metric} vs frequency ({mode})")
            save_current_figure(
                FREQUENCY_METRIC_DIR / safe_tag(f"{mode}_{metric}_vs_frequency.png")
            )


def plot_measurement_vs_simulation(
    v_meas: np.ndarray,
    i_meas: np.ndarray,
    record: dict[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    vp, current = downsample_arrays(record["df"], ["Vp", "I"])
    plt.figure()
    plt.semilogy(v_meas, np.abs(i_meas) + I_FLOOR_ABS, ".", markersize=2, label="measurement")
    plt.semilogy(vp, np.abs(current) + I_FLOOR_ABS, linewidth=1.1, label="simulation")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.xlabel("voltage (V)")
    plt.ylabel("|I| (A)")
    plt.title(f"Measured vs simulated ({record['mode']})")
    plt.legend()
    save_current_figure(path)


# =============================================================================
# Main workflow
# =============================================================================


def main() -> None:
    ensure_dirs()

    for required_path, description in (
        (CSV_MEAS, "measurement CSV"),
        (TEMPLATE_SRC, "SPICE template"),
        (THETA_PATH, "fitted-parameter CSV"),
        (NGSPICE, "NGSpice executable"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing {description}: {required_path}")

    v_meas, i_meas = read_measurement_csv(CSV_MEAS)
    theta = read_theta_best(THETA_PATH)
    estimated_compliance = estimate_positive_compliance(v_meas, i_meas)

    mode_settings: dict[str, dict[str, float]] = {}
    if RUN_NOLIMIT_MODE:
        mode_settings["NOLIMIT"] = {
            "icomp_pos": 1e30,
            "vcomp": VCOMP_FIXED,
            "rlo": 1e-3,
            "rhi": 1e-3,
        }
    if RUN_LIMIT_MODE:
        mode_settings["LIMIT"] = {
            "icomp_pos": estimated_compliance,
            "vcomp": VCOMP_FIXED,
            "rlo": RLO_FIXED,
            "rhi": RHI_FIXED,
        }
    if not mode_settings:
        raise ValueError("Enable at least one of RUN_NOLIMIT_MODE or RUN_LIMIT_MODE.")

    cases = build_operating_condition_cases(v_meas)
    total_runs = len(cases) * len(mode_settings)

    print(f"[INFO] NGSpice: {NGSPICE}")
    print(f"[INFO] Template: {TEMPLATE_SRC}")
    print(f"[INFO] Parameters: {THETA_PATH}")
    print(f"[INFO] Measurement points: {len(v_meas)}")
    print(f"[INFO] Estimated positive compliance: {estimated_compliance:.6g} A")
    print(f"[INFO] Operating-condition cases: {len(cases)}")
    print(f"[INFO] Planned simulation runs: {total_runs}")

    summary_rows: list[dict[str, Any]] = []
    settings_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    registry: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {} for mode in mode_settings
    }
    measurement_records: list[dict[str, Any]] = []

    successful_runs = 0
    failed_runs = 0

    for case_index, case in enumerate(cases, start=1):
        tstop = compute_case_tstop(case)

        for mode, settings in mode_settings.items():
            case_tag = safe_tag(
                f"{case['group']}_{case['name']}_{mode}_ts{fmt_num(case['time_scale'])}"
            )

            attempts: list[tuple[float, int]] = [(DTMAX_SIM, NGSPICE_TIMEOUT_S)]
            if RETRY_ON_FAILURE_OR_TRUNCATION:
                attempts.extend(
                    (DTMAX_SIM * factor, timeout)
                    for factor, timeout in zip(RETRY_DTMAX_FACTORS, RETRY_TIMEOUTS_S)
                )

            final_df: pd.DataFrame | None = None
            final_dirs: dict[str, Path] | None = None
            final_log_path: Path | None = None
            final_reason = "not_run"
            final_attempt = 0

            for attempt_index, (dtmax, timeout_s) in enumerate(attempts, start=1):
                final_attempt = attempt_index
                df, dirs, log_path, reason = simulate_case(
                    theta=theta,
                    case=case,
                    case_tag=case_tag,
                    tstop=tstop,
                    mode_settings=settings,
                    dtmax=dtmax,
                    timeout_s=timeout_s,
                )
                final_df = df
                final_dirs = dirs
                final_log_path = log_path
                final_reason = reason

                if reason == "ok":
                    break

                print(
                    f"[WARN] {case_tag}: attempt {attempt_index}/{len(attempts)} "
                    f"ended with {reason}"
                )

            if final_reason != "ok" or final_df is None or final_dirs is None:
                failed_runs += 1
                failed_rows.append(
                    {
                        "case_tag": case_tag,
                        "case": case["name"],
                        "group": case["group"],
                        "mode": mode,
                        "reason": final_reason,
                        "attempts": final_attempt,
                        "expected_tstop_s": tstop,
                        "time_end_s": (
                            float(final_df["time"].iloc[-1])
                            if final_df is not None and not final_df.empty
                            else np.nan
                        ),
                        "log_path": str(final_log_path) if final_log_path else "",
                        "log_tail": tail_text(final_log_path, 80),
                    }
                )
                print(f"[FAIL] {case_tag}: {final_reason}")
                continue

            successful_runs += 1
            make_all_case_plots(final_df, final_dirs["plot"], case["label"])
            metrics = compute_case_metrics(final_df, case, tstop)

            row = dict(metrics)
            row.update(
                {
                    "case_tag": case_tag,
                    "case": case["name"],
                    "label": case["label"],
                    "mode": mode,
                    "group": case["group"],
                    "input_style": case["input_style"],
                    "time_scale": case["time_scale"],
                    "tstop": tstop,
                    "vpos": case.get("vpos", np.nan),
                    "vneg": case.get("vneg", np.nan),
                    "repeat_count": case.get("repeat_count", np.nan),
                    "sequence": case.get("sequence", ""),
                    "num_ramps": case["num_ramps"],
                    "pts_per_ramp": case.get("pts_per_ramp", np.nan),
                    "hold_pts": case.get("hold_pts", np.nan),
                    "levels_per_ramp": case.get("levels_per_ramp", np.nan),
                    "hold_pts_per_level": case.get("hold_pts_per_level", np.nan),
                    "sim_dir": str(final_dirs["sim"]),
                    "plot_dir": str(final_dirs["plot"]),
                }
            )
            summary_rows.append(row)

            settings_rows.append(
                {
                    key: row.get(key, np.nan)
                    for key in (
                        "case_tag",
                        "case",
                        "label",
                        "mode",
                        "group",
                        "input_style",
                        "time_scale",
                        "tstop",
                        "frequency_Hz",
                        "vpos",
                        "vneg",
                        "repeat_count",
                        "sequence",
                        "num_ramps",
                        "pts_per_ramp",
                        "hold_pts",
                        "levels_per_ramp",
                        "hold_pts_per_level",
                    )
                }
            )
            branch_rows.extend(
                flatten_branch_metrics(
                    case_tag=case_tag,
                    case=case,
                    mode=mode,
                    metrics=metrics,
                )
            )

            record = {
                "case_tag": case_tag,
                "case": case,
                "df": final_df,
                "label": case["label"],
                "mode": mode,
            }
            registry[mode].setdefault(case["group"], []).append(record)
            if case["group"] == "measurement_trace":
                measurement_records.append(record)

            print(
                f"[OK] {successful_runs + failed_runs}/{total_runs} "
                f"case-group {case_index}/{len(cases)}: {case_tag}"
            )

    summary_df = pd.DataFrame(summary_rows)
    settings_df = pd.DataFrame(settings_rows)
    branch_df = pd.DataFrame(branch_rows)

    summary_df.to_csv(OUT_DIR / "summary_all_cases.csv", index=False)
    settings_df.to_csv(OUT_DIR / "case_settings_table.csv", index=False)
    branch_df.to_csv(OUT_DIR / "branch_metrics_table.csv", index=False)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(OUT_DIR / "failed_cases.csv", index=False)

    overlay_manifest = generate_group_overlays(registry)
    if overlay_manifest:
        pd.DataFrame(overlay_manifest).to_csv(OUT_DIR / "overlay_manifest.csv", index=False)

    generate_frequency_metric_plots(summary_df)

    for record in measurement_records:
        plot_measurement_vs_simulation(
            v_meas,
            i_meas,
            record,
            OVERLAY_DIR / f"measurement_vs_simulation_{record['mode']}.png",
        )

    print("\n[COMPLETE]")
    print(f"Output directory: {OUT_DIR}")
    print(f"Successful simulations: {successful_runs}")
    print(f"Failed simulations: {failed_runs}")
    print("Generated tables:")
    print("  - summary_all_cases.csv")
    print("  - case_settings_table.csv")
    print("  - branch_metrics_table.csv")
    if failed_rows:
        print("  - failed_cases.csv")
    if overlay_manifest:
        print("  - overlay_manifest.csv")


if __name__ == "__main__":
    main()
