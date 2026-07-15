# fit_full_guided_init_switch.py
# ============================================================
# Practical guided fitter for memdiode_bipolar (NGSpice + PSO)
#
# Key points:
#   - Uses seed_theta from previous theta_best.csv if available
#   - Supports INIT_MODE = "auto" / "seed" / "random"
#   - Coarse fit on ~200 points (fast) -> refine on full data
#   - Compliance FIXED (prevents PSO cheating via compliance)
#   - Reweighted cost focuses on: neg branch + right-lower tail + pos HRS
#   - Outputs: ./fit_result_guided/
#
# Usage:
#   1) Put this file next to DC-IV.csv and ngspice.exe
#   2) Run:  py .\fit_full_guided_init_switch.py
# ============================================================

import random
import re
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import json
from datetime import datetime

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
CSV_MEAS = BASE_DIR / "data" / "DC-IV.csv"
NGSPICE = Path(shutil.which("ngspice") or (BASE_DIR / "ngspice.exe"))

OUT_DIR = BASE_DIR / "results" / "fit"
LOG_DIR  = OUT_DIR / "fit_logs"
TEMPLATE_PATH = OUT_DIR / "fitdeck_embedded.cir"

# =========================
# Initial population mode
# =========================
# "random" = ignore old result, always random start
# "seed"   = must load old theta file, otherwise stop with error
# "auto"   = try old theta first, fallback to random if not found/invalid
INIT_MODE = "random"

SEED_CANDIDATES = [
    OUT_DIR / "theta_best.csv",
    OUT_DIR / "theta_best_good.csv",
    OUT_DIR / "theta_coarse_stop.csv",
]

# =========================
# Initial baseline export
# =========================
# 在 PSO 真正更新之前，先把「初始族群中最佳粒子」另存成 baseline，
# 方便與 coarse/refine 後結果做前後比較。
EXPORT_INITIAL_BASELINE = True
INITIAL_BASELINE_PICK = "best_in_swarm"   # "best_in_swarm" / "particle0"


# =========================
# PSO settings (tuned for speed)
# =========================
SEED = 7

# Coarse stage (downsampled)
N_PARTICLES_C = 20 #28
N_ITERS_C     = 60   # iterations per coarse batch

# Refine stage (full data, seeded)
N_PARTICLES_R = 20 #24
N_ITERS_R     = 80

W_MAX = 0.85
W_MIN = 0.40
C1, C2 = 1.5, 1.5

# Separate stagnation / reheat controls for coarse vs refine.
# Coarse should explore more aggressively; refine should exploit gently.
COARSE_STAG_ITERS  = 6 #8
COARSE_IMPROVE_EPS = 8e-4
REFINE_STAG_ITERS  = 8 #12
REFINE_IMPROVE_EPS = 2e-5

# Stage-transition controls
# coarse 必須先達標才進 refine；否則停止在 coarse
COARSE_TARGET_COST = 1.0
COARSE_MIN_ITERS   = 6
COARSE_MAX_ROUNDS  = 6

# 可選：refine 若達標可提早停；設成 None 代表只跑固定 N_ITERS_R
REFINE_TARGET_COST = 0.3

# coarse fallback / anti-stall
COARSE_SUBSET_TARGET_N = 320
FORCE_REFINE_IF_COARSE_STALLED = True
COARSE_FORCE_REFINE_MAX_COST = 2.20
COARSE_STALL_BATCH_WINDOW = 3
COARSE_STALL_EPS = 5e-4

# PSO anti-stall
REHEAT_FRAC = 0.75
REHEAT_KEEP = 2
REHEAT_JITTER_SCALE = 0.18
ALL_FAIL_JITTER_SCALE = 0.22
REFINE_MIN_ITERS   = 6
REFINE_MAX_ROUNDS  = 10

# Explicit per-stage reheat / probe controls
COARSE_REHEAT_FRAC = 0.75
COARSE_REHEAT_KEEP = 2
COARSE_REHEAT_JITTER_SCALE = 0.18

REFINE_REHEAT_FRAC = 0.10
REFINE_REHEAT_KEEP = 2
REFINE_REHEAT_JITTER_SCALE = 0.05
REFINE_LOCAL_PROBE_COUNT = 10
REFINE_LOCAL_PROBE_SCALE = 0.035

# Final deterministic polish after PSO
FINAL_LOCAL_POLISH = True
LOCAL_POLISH_ROUNDS = 3
LOCAL_POLISH_ACCEPT_EPS = 5e-6
LOCAL_POLISH_LINEAR_FRAC = 0.06
LOCAL_POLISH_LOG_STEP_DECADES = 0.18
LOCAL_POLISH_SHRINK = 0.55
# =========================
# Balanced objective tuning
# =========================
BALANCED_OBJECTIVE = True
LR_PICK_MIN = 36
LR_PICK_FRAC = 0.40

BALANCED_WEIGHTS = {
    "zero":   0.10,
    "poshrs": 0.16,
    "set":    0.12,
    "lr":     0.15,
    "dlo":    0.10,
    "dhi":    0.03,
    "nfar":   0.12,
    "npeak":  0.15,
    "nknee":  0.05,
    "nfocus": 0.16,
}
BALANCED_NEG_GAIN = 1.00

LOCAL_POLISH_PARAM_ORDER = [
    "VSET","VRES","ETA_SET","ETA_RES",
    "ALPHA_MAX","ALPHA_MIN","BETAA","ISCALE",
    "H0","IMAX","IMIN","CH0","EI","ROFF"
]


# =========================
# Simulation timing (speed > perfection)
# =========================
TSTOP_FULL = 10.0
DTMAX_SIM  = 2e-4   # a bit looser than 1e-4 for speed

# print-step: larger = fewer points output = faster
PRINT_DIV  = 6      # (was 10) -> faster
PRINT_MIN  = 2e-5
PRINT_MAX  = 4e-3


# =========================
# Cost settings
# =========================
I_FLOOR   = 3e-10
COST_FAIL = 1e9
VOPEN_MAX   = 1.2      # 0 -> 1.2V 是最常見的開口形狀區
VOPEN_MIN   = 0.02
OPEN_MIN_N  = 26
OPEN_WCAP   = 55.0

# =========================
# Mask debug visualization
# =========================
DEBUG_MASK_PLOTS = True          # 需要時開/關
DEBUG_MASK_EVERY = 0             # 0=只畫一次；例如 25=每25次objective畫一次(會慢)
DEBUG_MASK_MAX_FILES = 6         # 最多輸出幾組(避免爆檔)

# =========================
# Fixed model knobs
# =========================
KSW_FIXED     = 3
RH0_FIXED     = 1e3
RH_MIN_FIXED  = 1.0
RH_MAX_FIXED  = 1e7
VSLOPE_FIXED  = 0.5


# =========================
# Compliance (FIXED, NOT FITTED)
# =========================
RLO_FIXED      = 1.0
RHI_FIXED      = 2e8       # strong clamp
VCOMP_FIXED    = 0.0       # "any V>0"
VSLOPE_POS_FIX = 0.02      # sharp voltage gating
ISLOPE_REL     = 0.02      # sharp current gating


# =========================
# Knee / 0V locking (do not over-dominate)
# =========================
V0_WINDOW   = 0.22
I_KNEE      = 1e-6
KNEE_GAIN   = 0.9


# =========================
# Hysteresis penalties (keep but not too strict)
# =========================
XH_RANGE_MIN       = 0.16
XH_HYST_MIN        = 0.04
XH_PEN_GAIN_RANGE  = 7.0
XH_PEN_GAIN_HYST   = 9.0


# =========================
# Parameters / bounds
# =========================
MEM_PARAMS = [
    "IMAX","IMIN","ALPHA_MAX","ALPHA_MIN","BETAA",
    "VSET","VRES","ETA_SET","ETA_RES",
    "CH0","ISCALE","H0","EI","ROFF"
]

BOUNDS = {
    "BETAA":     (0.05, 0.95),
    "EI":        (1e-40, 1e-3),      # don't let EI go crazy (often causes slope artifacts)
    "ROFF":      (1e6  , 1e14),

    "IMAX":      (1e-7,  2e-1),
    "IMIN":      (1e-10, 5e-4),

    "ALPHA_MAX": (0.8,  15.0),
    "ALPHA_MIN": (1e-20,  12.0),

    "VSET":      (0.2,  20.0),        # allow a bit higher than 3V if needed
    "VRES":      (0.2,  20.0),

    "ETA_SET":   (0.3,  120.0),
    "ETA_RES":   (1e-20,  35.0),

    "CH0":       (1e-40, 3e-5),
    "ISCALE":    (1e-10, 3e-1),
    "H0":        (1e-20, 0.97),
}

LOG_PARAMS = {"IMAX","IMIN","CH0","ISCALE","EI","ROFF","ETA_SET","ETA_RES"}


# =========================
# Embedded SPICE template
# =========================
FITDECK_TEMPLATE = r"""* ============================================================
* fitdeck_embedded.cir (FULL waveform + fixed compliance clamp)
* Replacements:
*  @PWL_INLINE@ @SIMOUT@ @TSTEP@ @DTMAX@ @TSTOP@
*  @IMAX@ @IMIN@ @ALPHA_MAX@ @ALPHA_MIN@ @VSET@ @VRES@ @ETA_SET@ @ETA_RES@ @CH0@ @ISCALE@ @H0@ @EI@ @ROFF@
*  @KSW@ @RH0@ @RH_MIN@ @RH_MAX@ @VSLOPE@
*  @ICOMP_POS@ @VCOMP@ @RLO@ @RHI@ @ISLOPE@ @VSLOPE_POS@
* ============================================================

Vcmd vc 0 PWL(
@PWL_INLINE@
+ )

* sense current (i(Vsen))
Vsen vc p0 0

.param ICOMP = @ICOMP_POS@
.param VCOMP = @VCOMP@
.param RLO   = @RLO@
.param RHI   = @RHI@
.param ISLOPE = @ISLOPE@
.param VSLOPE_POS = @VSLOPE_POS@

.func s(u) { 0.5*(1+tanh(u)) }
.func gpos(v) { s((v - VCOMP)/VSLOPE_POS) }        ; ~1 when v>VCOMP
.func glim(i) { s((abs(i)-ICOMP)/ISLOPE) }         ; ~1 when |i|>ICOMP

* series R increases only when (Vcmd>VCOMP AND |I|>ICOMP)
* NOTE: use V(vc) (command) so compliance triggers consistently like SMU behavior
Rlim p0 p r={ RLO + (RHI-RLO)*gpos(V(vc))*glim(i(Vsen)) }

X1 p 0 x xh memdiode_bipolar CH0=@CH0@ H0=@H0@

.options method=gear reltol=4e-3 abstol=2e-15 vntol=2e-9 chgtol=2e-16 gmin=1e-12 itl1=900 itl4=350

.control
  set noaskquit
  set filetype=ascii
  set wr_singlescale
  tran @TSTEP@ @TSTOP@ 0 @DTMAX@ uic
  * columns: time, Vcmd, Vp, I(Vcmd), x, xh
  wrdata @SIMOUT@ v(vc) v(p) i(Vcmd) v(x) v(xh)
  quit
.endc

* ============================================================
* memdiode_bipolar
* ============================================================
.subckt memdiode_bipolar p n x xh CH0=@CH0@ H0=@H0@

.param KSW    = @KSW@
.param RH0    = @RH0@
.param RH_MIN = @RH_MIN@
.param RH_MAX = @RH_MAX@

.param ROFF   = @ROFF@
.param BETA   = @BETAA@
.param EI     = @EI@
.param CPAR   = 1p

.func softclip(v) { 0.5*(tanh(4*(v-0.5)) + 1) }
.func blend(w,a,b) { w*a + (1-w)*b }
.func safe_exp(v)  { exp(min(max(v,-25),25)) }

.param IMAX    = @IMAX@
.param IMIN    = @IMIN@
.param ALPHAMX = @ALPHA_MAX@
.param ALPHAMN = @ALPHA_MIN@
.param ISCALE  = @ISCALE@

.param VSET    = @VSET@
.param VRES    = @VRES@
.param ETA_SET = @ETA_SET@
.param ETA_RES = @ETA_RES@

Bxh xh 0 V = { softclip(V(x)) }

.func I0(h)    { IMAX*h + IMIN*(1-h) }
.func Acoef(h) { ALPHAMX*h + ALPHAMN*(1-h) }

* small series R for numerical stability
Rs p d 1

* polarity-based mode switch (smooth)
.param VSLOPE = @VSLOPE@
.func sstep(v) { 0.5*(1 + tanh(v/VSLOPE)) }
BA A 0 V = { sstep( V(d,n) ) }

* Voltage-dependent time constants
.func Stau(vdn) { RH0*safe_exp( -ETA_SET*( vdn - VSET ) ) }
.func Rtau(vdn) { RH0*safe_exp(  ETA_RES*( vdn + VRES ) ) }

Rh x A r={ min(RH_MAX, max(RH_MIN, blend(V(A), Stau(V(d,n)), Rtau(V(d,n))) )) }
Cx x 0 {CH0} ic={H0}

* state leak to avoid floating node
Rxl x 0 1e12

.func Idiff(v,h) { safe_exp(BETA*Acoef(h)*v) - safe_exp(-(1-BETA)*Acoef(h)*v) }
Bcond d n I = { ISCALE*( I0(V(xh))*Idiff(V(d,n), V(xh)) + EI*V(d,n) ) }

Rleak d n {ROFF}
Cpar  d 0 {CPAR}

.ends memdiode_bipolar
.end
"""


# =========================
# Utilities
# =========================
def make_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def collect_run_config():
    return {
        # paths / mode
        "INIT_MODE": INIT_MODE,
        "SEED": SEED,
        "OUT_DIR": str(OUT_DIR),
        "LOG_DIR": str(LOG_DIR),
        "EXPORT_INITIAL_BASELINE": EXPORT_INITIAL_BASELINE,
        "INITIAL_BASELINE_PICK": INITIAL_BASELINE_PICK,

        # PSO
        "N_PARTICLES_C": N_PARTICLES_C,
        "N_ITERS_C": N_ITERS_C,
        "N_PARTICLES_R": N_PARTICLES_R,
        "N_ITERS_R": N_ITERS_R,
        "W_MAX": W_MAX,
        "W_MIN": W_MIN,
        "C1": C1,
        "C2": C2,
        "COARSE_STAG_ITERS": COARSE_STAG_ITERS,
        "COARSE_IMPROVE_EPS": COARSE_IMPROVE_EPS,
        "REFINE_STAG_ITERS": REFINE_STAG_ITERS,
        "REFINE_IMPROVE_EPS": REFINE_IMPROVE_EPS,

        # stage transition
        "COARSE_TARGET_COST": COARSE_TARGET_COST,
        "COARSE_MIN_ITERS": COARSE_MIN_ITERS,
        "COARSE_MAX_ROUNDS": COARSE_MAX_ROUNDS,
        "REFINE_TARGET_COST": REFINE_TARGET_COST,
        "REFINE_MIN_ITERS": REFINE_MIN_ITERS,
        "REFINE_MAX_ROUNDS": REFINE_MAX_ROUNDS,
        "COARSE_SUBSET_TARGET_N": COARSE_SUBSET_TARGET_N,
        "FORCE_REFINE_IF_COARSE_STALLED": FORCE_REFINE_IF_COARSE_STALLED,
        "COARSE_FORCE_REFINE_MAX_COST": COARSE_FORCE_REFINE_MAX_COST,
        "COARSE_STALL_BATCH_WINDOW": COARSE_STALL_BATCH_WINDOW,
        "COARSE_STALL_EPS": COARSE_STALL_EPS,
        "COARSE_REHEAT_FRAC": COARSE_REHEAT_FRAC,
        "COARSE_REHEAT_KEEP": COARSE_REHEAT_KEEP,
        "COARSE_REHEAT_JITTER_SCALE": COARSE_REHEAT_JITTER_SCALE,
        "REFINE_REHEAT_FRAC": REFINE_REHEAT_FRAC,
        "REFINE_REHEAT_KEEP": REFINE_REHEAT_KEEP,
        "REFINE_REHEAT_JITTER_SCALE": REFINE_REHEAT_JITTER_SCALE,
        "REFINE_LOCAL_PROBE_COUNT": REFINE_LOCAL_PROBE_COUNT,
        "REFINE_LOCAL_PROBE_SCALE": REFINE_LOCAL_PROBE_SCALE,
        "ALL_FAIL_JITTER_SCALE": ALL_FAIL_JITTER_SCALE,

        # sim timing
        "TSTOP_FULL": TSTOP_FULL,
        "DTMAX_SIM": DTMAX_SIM,
        "PRINT_DIV": PRINT_DIV,
        "PRINT_MIN": PRINT_MIN,
        "PRINT_MAX": PRINT_MAX,

        # cost
        "I_FLOOR": I_FLOOR,
        "COST_FAIL": COST_FAIL,
        "VOPEN_MAX": VOPEN_MAX,
        "VOPEN_MIN": VOPEN_MIN,
        "OPEN_MIN_N": OPEN_MIN_N,
        "OPEN_WCAP": OPEN_WCAP,

        # debug
        "DEBUG_MASK_PLOTS": DEBUG_MASK_PLOTS,
        "DEBUG_MASK_EVERY": DEBUG_MASK_EVERY,
        "DEBUG_MASK_MAX_FILES": DEBUG_MASK_MAX_FILES,

        # fixed model knobs
        "KSW_FIXED": KSW_FIXED,
        "RH0_FIXED": RH0_FIXED,
        "RH_MIN_FIXED": RH_MIN_FIXED,
        "RH_MAX_FIXED": RH_MAX_FIXED,
        "VSLOPE_FIXED": VSLOPE_FIXED,

        # compliance
        "RLO_FIXED": RLO_FIXED,
        "RHI_FIXED": RHI_FIXED,
        "VCOMP_FIXED": VCOMP_FIXED,
        "VSLOPE_POS_FIX": VSLOPE_POS_FIX,
        "ISLOPE_REL": ISLOPE_REL,

        # penalties
        "V0_WINDOW": V0_WINDOW,
        "I_KNEE": I_KNEE,
        "KNEE_GAIN": KNEE_GAIN,
        "XH_RANGE_MIN": XH_RANGE_MIN,
        "XH_HYST_MIN": XH_HYST_MIN,
        "XH_PEN_GAIN_RANGE": XH_PEN_GAIN_RANGE,
        "XH_PEN_GAIN_HYST": XH_PEN_GAIN_HYST,

        # params
        "MEM_PARAMS": MEM_PARAMS,
        "LOG_PARAMS": sorted(list(LOG_PARAMS)),
        "BOUNDS": {k: [float(v[0]), float(v[1])] for k, v in BOUNDS.items()},
    }

def save_run_config(run_id, extra=None):
    ensure_dirs()
    cfg = collect_run_config()
    cfg["run_id"] = run_id
    cfg["saved_at"] = datetime.now().isoformat(timespec="seconds")
    if extra:
        cfg.update(extra)

    out_path = OUT_DIR / f"run_config_{run_id}.json"
    out_path.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"[config] saved -> {out_path}")
    return out_path

def append_run_summary(run_id, summary: dict):
    ensure_dirs()
    csv_path = OUT_DIR / "run_summary.csv"

    row = {
        "run_id": run_id,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "INIT_MODE": INIT_MODE,
        "SEED": SEED,
        "N_PARTICLES_C": N_PARTICLES_C,
        "N_ITERS_C": N_ITERS_C,
        "N_PARTICLES_R": N_PARTICLES_R,
        "N_ITERS_R": N_ITERS_R,
        "W_MAX": W_MAX,
        "W_MIN": W_MIN,
        "C1": C1,
        "C2": C2,
        "COARSE_TARGET_COST": COARSE_TARGET_COST,
        "REFINE_TARGET_COST": REFINE_TARGET_COST,
    }
    row.update(summary)

    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[summary] appended -> {csv_path}")
    return csv_path

def ensure_dirs():
    OUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

def write_template():
    TEMPLATE_PATH.write_text(FITDECK_TEMPLATE, encoding="utf-8")

def read_meas_csv(path: Path):
    df = pd.read_csv(path, engine="python")
    if df.shape[1] < 2:
        raise ValueError("DC-IV.csv must have at least 2 columns: V, I")
    v_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    i_raw = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    m = (~v_raw.isna()) & (~i_raw.isna())
    v = v_raw[m].to_numpy(dtype=float)
    i = i_raw[m].to_numpy(dtype=float)
    if len(v) < 20:
        raise ValueError("Too few numeric rows after cleaning. Check DC-IV.csv format.")
    return v, i

def estimate_icomp_pos(V: np.ndarray, I: np.ndarray):
    m = V > 0.5
    if not np.any(m):
        return 1e-3
    x = np.abs(I[m])
    ic = float(np.quantile(x, 0.98))
    return float(np.clip(ic, 1e-6, 5e-2))

def make_time_vector(N: int, TSTOP: float):
    return np.linspace(0.0, TSTOP, N) if N >= 2 else np.array([0.0])

def pick_tstep_print(N_full: int):
    dt_meas = TSTOP_FULL / max(1, (N_full - 1))
    tstep = dt_meas / PRINT_DIV
    return float(np.clip(tstep, PRINT_MIN, PRINT_MAX))

def pwl_inline_from_tv(t: np.ndarray, v: np.ndarray, pairs_per_line=8):
    items = [f"{ti:.12g} {vi:.12g}" for ti, vi in zip(t, v)]
    lines = []
    for i in range(0, len(items), pairs_per_line):
        lines.append("+ " + " ".join(items[i:i+pairs_per_line]))
    return "\n".join(lines)

def render_template(out_path: Path, repl: dict):
    txt = TEMPLATE_PATH.read_text(encoding="utf-8", errors="ignore")
    for k, v in repl.items():
        txt = txt.replace(k, str(v))
    leftovers = re.findall(r"@[A-Za-z0-9_]+@", txt)
    if leftovers:
        raise RuntimeError(f"Unreplaced placeholders in deck: {sorted(set(leftovers))[:12]}")
    out_path.write_text(txt, encoding="utf-8")

def run_ngspice(deck_path: Path, log_path: Path, cwd: Path, timeout_s=80):
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
        raise ValueError(f"wrdata file has {data.shape[1]} columns (<6): {path}")
    t    = data[:,0]
    vcmd = data[:,1]
    vp   = data[:,2]
    idev = data[:,3]
    vx   = data[:,4]
    vxh  = data[:,5]
    return t, vcmd, vp, idev, vx, vxh

def tail_text(path: Path, n_lines=160):
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""

def log_rmse(a: np.ndarray, b: np.ndarray, floor=1e-12):
    la = np.log10(np.abs(a) + floor)
    lb = np.log10(np.abs(b) + floor)
    return float(np.sqrt(np.mean((la - lb) ** 2)))

def log_rmse_weighted(a: np.ndarray, b: np.ndarray, w: np.ndarray, floor=1e-12):
    la = np.log10(np.abs(a) + floor)
    lb = np.log10(np.abs(b) + floor)
    w = np.asarray(w, dtype=float)
    wsum = float(np.sum(w))
    if wsum <= 0:
        return float(np.sqrt(np.mean((la - lb) ** 2)))
    return float(np.sqrt(np.sum(w * (la - lb) ** 2) / wsum))

def clip_to_bounds(theta: np.ndarray, param_order: list[str]):
    y = theta.copy()
    for i, name in enumerate(param_order):
        lo, hi = BOUNDS[name]
        y[i] = min(max(y[i], lo), hi)
    return y

def sample_uniform(param_order: list[str]):
    arr = []
    for n in param_order:
        lo, hi = BOUNDS[n]
        if n in LOG_PARAMS:
            arr.append(10**random.uniform(np.log10(lo), np.log10(hi)))
        else:
            arr.append(random.uniform(lo, hi))
    return np.array(arr, dtype=float)

def jitter_from_seed(seed: np.ndarray, param_order: list[str], scale=0.10):
    x = seed.copy()
    for i, name in enumerate(param_order):
        lo, hi = BOUNDS[name]
        if name in LOG_PARAMS:
            x[i] = x[i] * (10**np.random.uniform(-scale, scale))
        else:
            x[i] = x[i] + np.random.uniform(-scale, scale) * (hi - lo)
        x[i] = min(max(x[i], lo), hi)
    return x

def sample_uniform_rng(param_order: list[str], py_rng: random.Random):
    arr = []
    for n in param_order:
        lo, hi = BOUNDS[n]
        if n in LOG_PARAMS:
            arr.append(10 ** py_rng.uniform(np.log10(lo), np.log10(hi)))
        else:
            arr.append(py_rng.uniform(lo, hi))
    return np.array(arr, dtype=float)

def jitter_from_seed_rng(seed: np.ndarray, param_order: list[str], np_rng, scale=0.10):
    x = seed.copy()
    for i, name in enumerate(param_order):
        lo, hi = BOUNDS[name]
        if name in LOG_PARAMS:
            x[i] = x[i] * (10 ** np_rng.uniform(-scale, scale))
        else:
            x[i] = x[i] + np_rng.uniform(-scale, scale) * (hi - lo)
        x[i] = min(max(x[i], lo), hi)
    return x

def build_initial_swarm(n_particles: int, seed_theta=None, seed: int = SEED):
    """Build the exact initial swarm that PSO will start from, without changing global RNG state."""
    py_rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    swarm = np.stack([sample_uniform_rng(MEM_PARAMS, py_rng) for _ in range(n_particles)], axis=0)
    vel = np.zeros_like(swarm)

    if seed_theta is not None:
        seed0 = clip_to_bounds(seed_theta, MEM_PARAMS)
        swarm[0] = seed0
        for k in range(1, min(5, n_particles)):
            swarm[k] = jitter_from_seed_rng(seed0, MEM_PARAMS, np_rng, scale=0.10)

    return swarm, vel

def evaluate_initial_population(
    swarm: np.ndarray,
    objective_fn,
    V_eval: np.ndarray,
    I_eval: np.ndarray,
    pick_mode: str = "best_in_swarm",
    iter_offset: int = 900000,
):
    """Evaluate initial swarm before any PSO update and return one representative theta."""
    costs = []
    for i in range(len(swarm)):
        c = objective_fn(swarm[i], V_eval, I_eval, iter_id=iter_offset + i + 1)
        costs.append(float(c))
    costs = np.array(costs, dtype=float)

    if pick_mode == "particle0":
        best_idx = 0
    else:
        best_idx = int(np.argmin(costs))

    return {
        "theta": swarm[best_idx].copy(),
        "cost": float(costs[best_idx]),
        "index": int(best_idx),
        "all_costs": costs,
    }

def knee_v(V, Iabs, mask, Itarget):
    idx = np.where(mask & (Iabs >= Itarget))[0]
    if len(idx) == 0:
        return None
    return float(V[idx[0]])

def load_seed_theta_from_csv(csv_path: Path, param_order: list[str]):
    print(f"[seed-debug] trying: {csv_path}")

    if not csv_path.exists():
        print(f"[seed-debug] file not found")
        return None

    try:
        df = pd.read_csv(csv_path)
        print(f"[seed-debug] columns = {list(df.columns)}")

        if ("param" not in df.columns) or ("value" not in df.columns):
            print(f"[seed-debug] invalid columns, need ['param', 'value']")
            return None

        m = {str(p).strip(): float(v) for p, v in zip(df["param"], df["value"])}

        theta = []
        missing = 0
        for k in param_order:
            if k in m:
                theta.append(m[k])
            else:
                missing += 1
                theta.append(np.nan)

        theta = np.array(theta, dtype=float)

        if missing:
            print(f"[seed-debug] missing {missing} params -> random fill")
            for i, k in enumerate(param_order):
                if np.isnan(theta[i]):
                    lo, hi = BOUNDS[k]
                    if k in LOG_PARAMS:
                        theta[i] = 10 ** np.random.uniform(np.log10(lo), np.log10(hi))
                    else:
                        theta[i] = np.random.uniform(lo, hi)

        theta = clip_to_bounds(theta, param_order)
        print(f"[seed-debug] load OK")
        return theta

    except Exception as e:
        print(f"[seed-debug] failed to read {csv_path}: {e}")
        return None

def resolve_initial_seed(init_mode: str, param_order: list[str]):
    mode = str(init_mode).strip().lower()

    if mode not in {"random", "seed", "auto"}:
        raise ValueError(f"INIT_MODE must be 'random', 'seed', or 'auto', got: {init_mode}")

    if mode == "random":
        print("[seed] INIT_MODE=random -> ignore old theta, random start")
        return None

    for p in SEED_CANDIDATES:
        theta = load_seed_theta_from_csv(p, param_order)
        if theta is not None:
            print(f"[seed] loaded from {p}")
            return theta

    if mode == "seed":
        raise RuntimeError(
            "[seed] INIT_MODE=seed but no valid seed file found.\n"
            + "\n".join([f"  - {p}" for p in SEED_CANDIDATES])
        )

    print("[seed] INIT_MODE=auto -> no valid seed file found, fallback to random")
    return None


# =========================
# Downsample (keep turns + endpoints)
# =========================
def make_fit_subset(V: np.ndarray, I: np.ndarray, target_n=220):
    N = len(V)
    if N <= target_n:
        return V.copy(), I.copy()

    stride = max(1, N // target_n)
    idx = set(range(0, N, stride))
    idx.add(0); idx.add(N-1)

    # keep turn points (where dV sign flips)
    dV = np.diff(V)
    s = np.sign(dV)
    for k in range(1, len(s)):
        if s[k] == 0:
            s[k] = s[k-1]
    turns = np.where(np.diff(s) != 0)[0] + 1
    for t in turns.tolist():
        for dd in (-2,-1,0,1,2):
            j = int(np.clip(t+dd, 0, N-1))
            idx.add(j)

    # keep dense around 0V window
    near0 = np.where(np.abs(V) <= 0.35)[0]
    for j in near0[::max(1, len(near0)//40)].tolist():
        idx.add(int(j))

    idx = np.array(sorted(idx), dtype=int)
    return V[idx], I[idx]

# =========================
# Mask building + plotting helpers (embed into fit_0130)
# =========================
import matplotlib.pyplot as plt

def _knee_v(V, Iabs, mask, Itarget):
    idx = np.where(mask & (Iabs >= Itarget))[0]
    if len(idx) == 0:
        return None
    return float(V[idx[0]])

def _build_masks_meas_only(V, I, icomp_pos_fixed, I_FLOOR=3e-10, V0_WINDOW=0.22, I_KNEE=1e-6):
    V = V.astype(float)
    I = I.astype(float)
    Iabs = np.abs(I)

    # --- segment by monotonic pieces (same idea as objective) ---
    dV = np.diff(V, prepend=V[0])
    sgn = np.sign(dV)
    for k in range(1, len(sgn)):
        if sgn[k] == 0:
            sgn[k] = sgn[k-1]
    if sgn[0] == 0:
        nz = np.where(sgn != 0)[0]
        sgn[0] = sgn[nz[0]] if len(nz) else 1.0

    turns = np.where(np.diff(sgn) != 0)[0] + 1
    bounds = [0] + turns.tolist() + [len(V)]
    seg_id = np.zeros(len(V), dtype=int)
    for j in range(len(bounds) - 1):
        seg_id[bounds[j]:bounds[j+1]] = j

    seg_info = []
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j+1]
        vv = V[a:b]
        if len(vv) < 2:
            continue
        dv = vv[-1] - vv[0]
        seg_info.append({
            "id": j, "a": a, "b": b,
            "inc": dv > 0, "dec": dv < 0,
            "vmin": float(np.min(vv)), "vmax": float(np.max(vv)),
        })

    def pick_seg(kind: str):
        cands = []
        for si in seg_info:
            inc, dec = si["inc"], si["dec"]
            vmin, vmax = si["vmin"], si["vmax"]
            if kind == "pos_up" and inc and vmax > 0.2:
                cands.append(si)
            elif kind == "pos_down" and dec and vmax > 0.2:
                cands.append(si)
            elif kind == "neg_up" and inc and vmin < -0.2:
                cands.append(si)
            elif kind == "neg_down" and dec and vmin < -0.2:
                cands.append(si)
        if not cands:
            return None
        def score(si):
            span = si["vmax"] - si["vmin"]
            if kind in ("pos_up", "neg_up"):
                return (span, si["a"])
            else:
                return (span, -si["a"])
        return max(cands, key=score)["id"]

    sid_pos_up   = pick_seg("pos_up")
    sid_pos_down = pick_seg("pos_down")
    sid_neg_up   = pick_seg("neg_up")
    sid_neg_down = pick_seg("neg_down")

    m_pos_up   = (seg_id == sid_pos_up)   & (V > 0)
    m_pos_down = (seg_id == sid_pos_down) & (V > 0)
    m_neg_up   = (seg_id == sid_neg_up)   & (V < 0)
    m_neg_down = (seg_id == sid_neg_down) & (V < 0)

    # fallbacks
    if sid_pos_up is None:
        m_pos_up = (V > 0) & (dV >= 0)
    if sid_pos_down is None:
        m_pos_down = (V > 0) & (dV < 0)
    if sid_neg_up is None:
        m_neg_up = (V < 0) & (dV > 0)
    if sid_neg_down is None:
        m_neg_down = (V < 0) & (dV <= 0)

    # --- estimate v_set_edge from measured pos_up (same core logic) ---
    def estimate_vset_edge():
        m = m_pos_up & (V > 0.05)
        if int(np.sum(m)) < 8:
            posV = V[V > 0.2]
            return float(np.quantile(posV, 0.65)) if len(posV) else 3.0
        Vp2 = V[m]
        Ip2 = Iabs[m]
        order = np.argsort(Vp2)
        Vp2 = Vp2[order]
        logIp = np.log10(Ip2[order] + I_FLOOR)
        wwin = (Vp2 >= 0.2) & (Vp2 <= 8.0)
        if int(np.sum(wwin)) < 6:
            wwin = np.ones_like(Vp2, dtype=bool)
        Vw = Vp2[wwin]
        Lw = logIp[wwin]
        dVw = np.diff(Vw)
        dLw = np.diff(Lw)
        dVw = np.where(np.abs(dVw) < 1e-12, 1e-12, dVw)
        slope = dLw / dVw
        if len(slope) == 0:
            return float(np.median(Vw))
        k = int(np.argmax(slope))
        return float(Vw[k])

    v_set_edge = estimate_vset_edge()

    m_zero = (np.abs(V) <= V0_WINDOW)

    v_hrs_max = max(0.25, v_set_edge - 0.25)
    m_pos_hrs = m_pos_up & (V >= 0.05) & (V <= v_hrs_max) & (Iabs < 0.80 * icomp_pos_fixed)

    dv_left, dv_right = 0.8, 1.0
    m_pos_setedge = m_pos_up & (V >= (v_set_edge - dv_left)) & (V <= (v_set_edge + dv_right)) & (V > 0.05)

    vsplit = max(3.6, v_set_edge + 0.8)
    m_pos_down_lowV  = m_pos_down & (V >= 0.05) & (V <= vsplit)
    m_pos_down_highV = m_pos_down & (V >  vsplit)

    # right-lower tail (lowest current points on pos_down)
    v_lr_hi = min(3.4, vsplit)
    base_lr = m_pos_down & (V >= 0.10) & (V <= v_lr_hi)
    idx_lr = np.where(base_lr)[0]
    m_down_lr = np.zeros_like(base_lr, dtype=bool)
    if len(idx_lr) > 0:
        K = int(max(LR_PICK_MIN, LR_PICK_FRAC * len(idx_lr)))
        order = np.argsort(Iabs[idx_lr])
        pick = idx_lr[order[:K]]
        m_down_lr[pick] = True

    # negative regions
    m_neg_far  = m_neg_down & (V <= -2.0)
    m_neg_peak = m_neg_down & (V >= -3.2) & (V <= -0.9)
    m_neg_knee = m_neg_down & (V >= -1.2) & (V <= -0.05)

    v_k_meas = _knee_v(V, Iabs, m_pos_up & (V > 0.02) & (Iabs < 0.9*icomp_pos_fixed), I_KNEE)

    info = {
        "v_set_edge": float(v_set_edge),
        "vsplit": float(vsplit),
        "v_hrs_max": float(v_hrs_max),
        "v_knee_meas": None if v_k_meas is None else float(v_k_meas),
    }

    masks = {
        # segments
        "pos_up": m_pos_up, "pos_down": m_pos_down, "neg_up": m_neg_up, "neg_down": m_neg_down,
        # objective blocks
        "zero": m_zero,
        "pos_hrs": m_pos_hrs,
        "pos_setedge": m_pos_setedge,
        "pos_down_lowV": m_pos_down_lowV,
        "pos_down_highV": m_pos_down_highV,
        "down_lr": m_down_lr,
        "neg_far": m_neg_far,
        "neg_peak": m_neg_peak,
        "neg_knee": m_neg_knee,
        "knee_zone": m_pos_up & (V > 0.02) & (Iabs < 0.9*icomp_pos_fixed),
    }
    return masks, info

def _plot_masks_overlay(V, I, masks, info, outdir: Path, tag: str, I_FLOOR=3e-10):
    outdir.mkdir(parents=True, exist_ok=True)

    show = ["zero","pos_hrs","pos_setedge","down_lr",
            "pos_down_lowV","pos_down_highV",
            "neg_far","neg_peak","neg_knee"]

    # --- symlog signed ---
    plt.figure()
    plt.yscale("symlog", linthresh=1e-9)
    plt.plot(V, I, ".", alpha=0.18, label="all")
    for k in show:
        m = masks.get(k, None)
        if m is not None and np.any(m):
            plt.plot(V[m], I[m], ".", alpha=0.95, label=k)
    for lbl, x in [("v_set_edge", info["v_set_edge"]), ("vsplit", info["vsplit"]), ("v_hrs_max", info["v_hrs_max"] )]:
        plt.axvline(x, linestyle=":", alpha=0.55)
    if info.get("v_knee_meas") is not None:
        plt.axvline(info["v_knee_meas"], linestyle=":", alpha=0.55)

    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.xlabel("V (V)")
    plt.ylabel("I (A) [symlog]")
    plt.title(f"Mask overlay (symlog) - {tag}")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"mask_overlay_symlog_{tag}.png", dpi=180)
    plt.close()

    # --- semilogy abs ---
    plt.figure()
    plt.semilogy(V, np.abs(I)+I_FLOOR, ".", alpha=0.18, label="all")
    for k in show:
        m = masks.get(k, None)
        if m is not None and np.any(m):
            plt.semilogy(V[m], np.abs(I[m])+I_FLOOR, ".", alpha=0.95, label=k)

    for lbl, x in [("v_set_edge", info["v_set_edge"]), ("vsplit", info["vsplit"]), ("v_hrs_max", info["v_hrs_max"] )]:
        plt.axvline(x, linestyle=":", alpha=0.55)
    if info.get("v_knee_meas") is not None:
        plt.axvline(info["v_knee_meas"], linestyle=":", alpha=0.55)

    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.xlabel("V (V)")
    plt.ylabel("|I| (A) [log]")
    plt.title(f"Mask overlay (|I|) - {tag}")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"mask_overlay_abs_{tag}.png", dpi=180)
    plt.close()

def _maybe_dump_masks(V, I, icomp, outdir: Path, tag: str,
                      call_idx: int, printed_state: dict,
                      DEBUG_MASK_PLOTS=True, DEBUG_MASK_EVERY=0, DEBUG_MASK_MAX_FILES=6):
    if not DEBUG_MASK_PLOTS:
        return
    if printed_state.get("mask_dump_count", 0) >= DEBUG_MASK_MAX_FILES:
        return
    if printed_state.get("mask_dump_once", False) is False and DEBUG_MASK_EVERY == 0:
        do = True
        printed_state["mask_dump_once"] = True
    else:
        do = (DEBUG_MASK_EVERY > 0) and (call_idx % DEBUG_MASK_EVERY == 0)

    if not do:
        return

    masks, info = _build_masks_meas_only(V, I, icomp_pos_fixed=icomp)
    _plot_masks_overlay(V, I, masks, info, outdir=outdir, tag=f"{tag}_{call_idx:06d}")
    printed_state["mask_dump_count"] = printed_state.get("mask_dump_count", 0) + 1

# =========================
# Objective (segment weighted) - STABLE + focus on circled neg region
# =========================
def objective_factory(icomp_pos_fixed: float, label: str):
    # printed: avoid spamming + allow mask dump control
    printed = {"mask": False, "fail": False, "calls": 0, "mask_dump_count": 0, "mask_dump_once": False}

    def objective(theta: np.ndarray, V_meas: np.ndarray, I_meas: np.ndarray, iter_id: int):
        ensure_dirs()
        printed["calls"] += 1

        # (optional) dump measurement masks overlay once / periodically
        _maybe_dump_masks(
            V_meas, I_meas, icomp_pos_fixed,
            outdir=OUT_DIR / "mask_debug",
            tag=label,
            call_idx=printed["calls"],
            printed_state=printed,
            DEBUG_MASK_PLOTS=DEBUG_MASK_PLOTS,
            DEBUG_MASK_EVERY=DEBUG_MASK_EVERY,
            DEBUG_MASK_MAX_FILES=DEBUG_MASK_MAX_FILES
        )

        tmap = {k: float(v) for k, v in zip(MEM_PARAMS, theta)}

        # Build PWL
        N_full = len(V_meas)
        t_full = make_time_vector(N_full, TSTOP_FULL)
        pwl_inline = pwl_inline_from_tv(t_full, V_meas, pairs_per_line=8)

        sim_name  = f"{label}_sim_{iter_id:06d}.dat"
        deck_name = f"{label}_deck_{iter_id:06d}.cir"
        log_name  = f"{label}_ng_{iter_id:06d}.log"

        deck_path = LOG_DIR / deck_name
        log_path  = LOG_DIR / log_name
        sim_path  = LOG_DIR / sim_name

        tstep_print = pick_tstep_print(N_full)
        islope = max(1e-12, ISLOPE_REL * icomp_pos_fixed)

        repl = {
            "@PWL_INLINE@": pwl_inline,
            "@SIMOUT@": sim_name,
            "@TSTEP@": f"{tstep_print:.12g}",
            "@DTMAX@": f"{DTMAX_SIM:.12g}",
            "@TSTOP@": f"{TSTOP_FULL:.12g}",

            "@KSW@": f"{KSW_FIXED:.12g}",
            "@RH0@": f"{RH0_FIXED:.12g}",
            "@RH_MIN@": f"{RH_MIN_FIXED:.12g}",
            "@RH_MAX@": f"{RH_MAX_FIXED:.12g}",
            "@VSLOPE@": f"{VSLOPE_FIXED:.12g}",

            "@ICOMP_POS@": f"{icomp_pos_fixed:.12g}",
            "@VCOMP@":     f"{VCOMP_FIXED:.12g}",
            "@RLO@":       f"{RLO_FIXED:.12g}",
            "@RHI@":       f"{RHI_FIXED:.12g}",
            "@ISLOPE@":    f"{islope:.12g}",
            "@VSLOPE_POS@":f"{VSLOPE_POS_FIX:.12g}",
        }

        for name in MEM_PARAMS:
            repl[f"@{name}@"] = f"{float(tmap[name]):.12g}"

        try:
            render_template(deck_path, repl)
        except Exception as e:
            if not printed["fail"]:
                printed["fail"] = True
                print(f"[{label} FAIL] render:", e)
            return COST_FAIL

        rc = run_ngspice(deck_path, log_path, cwd=LOG_DIR, timeout_s=80)
        if rc != 0 or (not sim_path.exists()):
            if (not printed["fail"]) and (rc != 124):
                printed["fail"] = True
                print(f"[{label} FAIL] ngspice rc={rc}")
                print("[log tail]\n" + tail_text(log_path))
            return COST_FAIL

        try:
            t, vcmd, vp, idev, vx, vxh = load_wrdata(sim_path)
        except Exception as e:
            if not printed["fail"]:
                printed["fail"] = True
                print(f"[{label} FAIL] wrdata:", e)
            return COST_FAIL

        # compare on command axis (like your measurement)
        I_sim = np.interp(t_full, t, idev)
        xh    = np.interp(t_full, t, vxh)

        V = V_meas.astype(float)
        I = I_meas.astype(float)
        Iabs = np.abs(I)
        Iabs_sim = np.abs(I_sim)

        # -------------------------
        # Segment by monotonic pieces
        # -------------------------
        dV = np.diff(V, prepend=V[0])
        sgn = np.sign(dV)
        for k in range(1, len(sgn)):
            if sgn[k] == 0:
                sgn[k] = sgn[k-1]
        if sgn[0] == 0:
            nz = np.where(sgn != 0)[0]
            sgn[0] = sgn[nz[0]] if len(nz) else 1.0

        turns = np.where(np.diff(sgn) != 0)[0] + 1
        bounds = [0] + turns.tolist() + [len(V)]
        seg_id = np.zeros(len(V), dtype=int)
        for j in range(len(bounds) - 1):
            seg_id[bounds[j]:bounds[j+1]] = j

        seg_info = []
        for j in range(len(bounds) - 1):
            a, b = bounds[j], bounds[j+1]
            vv = V[a:b]
            if len(vv) < 2:
                continue
            dv = vv[-1] - vv[0]
            seg_info.append({
                "id": j, "a": a, "b": b,
                "inc": dv > 0, "dec": dv < 0,
                "vmin": float(np.min(vv)), "vmax": float(np.max(vv)),
            })

        def pick_seg(kind: str):
            cands = []
            for si in seg_info:
                inc, dec = si["inc"], si["dec"]
                vmin, vmax = si["vmin"], si["vmax"]
                if kind == "pos_up" and inc and vmax > 0.2:
                    cands.append(si)
                elif kind == "pos_down" and dec and vmax > 0.2:
                    cands.append(si)
                elif kind == "neg_up" and inc and vmin < -0.2:
                    cands.append(si)
                elif kind == "neg_down" and dec and vmin < -0.2:
                    cands.append(si)
            if not cands:
                return None
            def score(si):
                span = si["vmax"] - si["vmin"]
                if kind in ("pos_up", "neg_up"):
                    return (span, si["a"])
                else:
                    return (span, -si["a"])
            return max(cands, key=score)["id"]

        sid_pos_up   = pick_seg("pos_up")
        sid_pos_down = pick_seg("pos_down")
        sid_neg_up   = pick_seg("neg_up")
        sid_neg_down = pick_seg("neg_down")

        m_pos_up   = (seg_id == sid_pos_up)   & (V > 0)
        m_pos_down = (seg_id == sid_pos_down) & (V > 0)
        m_neg_up   = (seg_id == sid_neg_up)   & (V < 0)
        m_neg_down = (seg_id == sid_neg_down) & (V < 0)

        if sid_pos_up is None:
            m_pos_up = (V > 0) & (dV >= 0)
        if sid_pos_down is None:
            m_pos_down = (V > 0) & (dV < 0)
        if sid_neg_up is None:
            m_neg_up = (V < 0) & (dV > 0)
        if sid_neg_down is None:
            m_neg_down = (V < 0) & (dV <= 0)

        # -------------------------
        # Estimate "set edge" from measured pos_up
        # -------------------------
        def estimate_vset_edge():
            m = m_pos_up & (V > 0.05)
            if int(np.sum(m)) < 8:
                posV = V[V > 0.2]
                return float(np.quantile(posV, 0.65)) if len(posV) else 3.0
            Vp2 = V[m]
            Ip2 = Iabs[m]
            order = np.argsort(Vp2)
            Vp2 = Vp2[order]
            logIp = np.log10(Ip2[order] + I_FLOOR)
            wwin = (Vp2 >= 0.2) & (Vp2 <= 8.0)
            if int(np.sum(wwin)) < 6:
                wwin = np.ones_like(Vp2, dtype=bool)
            Vw = Vp2[wwin]
            Lw = logIp[wwin]
            dVw = np.diff(Vw)
            dLw = np.diff(Lw)
            dVw = np.where(np.abs(dVw) < 1e-12, 1e-12, dVw)
            slope = dLw / dVw
            if len(slope) == 0:
                return float(np.median(Vw))
            k = int(np.argmax(slope))
            return float(Vw[k])

        v_set_edge = estimate_vset_edge()

        # -------------------------
        # Masks
        # -------------------------
        m_zero = (np.abs(V) <= V0_WINDOW)

        v_hrs_max = max(0.25, v_set_edge - 0.25)
        m_pos_hrs = m_pos_up & (V >= 0.05) & (V <= v_hrs_max) & (Iabs < 0.80 * icomp_pos_fixed)

        dv_left, dv_right = 0.8, 1.0
        m_pos_setedge = m_pos_up & (V >= (v_set_edge - dv_left)) & (V <= (v_set_edge + dv_right)) & (V > 0.05)

        vsplit = max(3.6, v_set_edge + 0.8)
        m_pos_down_lowV  = m_pos_down & (V >= 0.05) & (V <= vsplit)
        m_pos_down_highV = m_pos_down & (V >  vsplit)

        # Right-lower tail on pos_down (lowest current points)
        v_lr_hi = min(3.4, vsplit)
        base_lr = m_pos_down & (V >= 0.10) & (V <= v_lr_hi)
        idx_lr = np.where(base_lr)[0]
        m_down_lr = np.zeros_like(base_lr, dtype=bool)
        if len(idx_lr) > 0:
            K = int(max(LR_PICK_MIN, LR_PICK_FRAC * len(idx_lr)))
            order = np.argsort(Iabs[idx_lr])
            pick = idx_lr[order[:K]]
            m_down_lr[pick] = True

        # Negative regions
        m_neg_far  = m_neg_down & (V <= -2.0)
        m_neg_peak = m_neg_down & (V >= -3.2) & (V <= -0.9)
        m_neg_knee = m_neg_down & (V >= -1.2) & (V <= -0.05)

        # NEW: focus region for your circled area (covers both neg_up and neg_down)
        m_neg_focus = (V < 0) & (V >= -1.8) & (V <= -0.15) & (m_neg_up | m_neg_down)

        # -------------------------
        # Segment costs
        # -------------------------
        def seg_cost(mask, min_n=10, miss_pen=0.25, weighted=False, wcap=30.0):
            n = int(np.sum(mask))
            if n < 3:
                return None, n
            if not weighted:
                c = log_rmse(I_sim[mask], I[mask], floor=I_FLOOR)
            else:
                eps = 1e-18
                ww = (0.35 * icomp_pos_fixed) / (Iabs[mask] + eps)
                ww = np.clip(ww, 1.0, wcap)
                c = log_rmse_weighted(I_sim[mask], I[mask], ww, floor=I_FLOOR)
            if n < min_n:
                c += miss_pen * (min_n - n) / float(min_n)
            return float(c), n

        c_zero,   n_zero   = seg_cost(m_zero,        min_n=28, miss_pen=0.55, weighted=True,  wcap=18.0)
        c_poshrs, n_poshrs = seg_cost(m_pos_hrs,     min_n=16, miss_pen=0.40, weighted=True,  wcap=24.0)
        c_set,    n_set    = seg_cost(m_pos_setedge, min_n=10, miss_pen=0.35, weighted=False)
        c_lr,     n_lr     = seg_cost(m_down_lr,     min_n=18, miss_pen=0.55, weighted=True,  wcap=38.0)
        c_dlo,    n_dlo    = seg_cost(m_pos_down_lowV,  min_n=14, miss_pen=0.35, weighted=False)
        c_dhi,    n_dhi    = seg_cost(m_pos_down_highV, min_n=10, miss_pen=0.30, weighted=False)

        c_nfar,   n_nfar   = seg_cost(m_neg_far,     min_n=18, miss_pen=0.35, weighted=False)
        c_npeak,  n_npeak  = seg_cost(m_neg_peak,    min_n=20, miss_pen=0.40, weighted=False)
        c_nknee,  n_nknee  = seg_cost(m_neg_knee,    min_n=16, miss_pen=0.55, weighted=True,  wcap=30.0)

        # NEW focus cost (do NOT weighted; you care about high-current shape)
        c_nfocus, n_nfocus = seg_cost(m_neg_focus,   min_n=18, miss_pen=0.45, weighted=False)

        if not printed["mask"]:
            printed["mask"] = True
            print(f"[{label}] mask cnt: zero={n_zero} pos_hrs={n_poshrs} setedge={n_set} down_lr={n_lr} down_lo={n_dlo} down_hi={n_dhi} "
                  f"neg_far={n_nfar} neg_peak={n_npeak} neg_knee={n_nknee} neg_focus={n_nfocus}")
            print(f"[{label}] v_set_edge≈{v_set_edge:.3g}V  vsplit={vsplit:.3g}  ICOMP={icomp_pos_fixed:.3g}")

        # -------------------------
        # Weights (stable): focus region emphasized WITHOUT blowing up global fit
        # -------------------------
        if BALANCED_OBJECTIVE:
            weights = dict(BALANCED_WEIGHTS)
            NEG_GAIN = BALANCED_NEG_GAIN
        else:
            weights = {
                "zero":    0.08,
                "poshrs":  0.16,
                "set":     0.12,
                "lr":      0.22,
                "dlo":     0.08,
                "dhi":     0.03,

                "nfar":    0.12,
                "npeak":   0.15,
                "nknee":   0.05,

                "nfocus":  0.22,
            }
            NEG_GAIN = 1.15

        # Small negative emphasis only (avoid global collapse)
        for k in ("nfar","npeak","nknee","nfocus"):
            weights[k] *= NEG_GAIN

        segs = {
            "zero": c_zero, "poshrs": c_poshrs, "set": c_set, "lr": c_lr, "dlo": c_dlo, "dhi": c_dhi,
            "nfar": c_nfar, "npeak": c_npeak, "nknee": c_nknee,
            "nfocus": c_nfocus,
        }

        acc = 0.0
        wsum = 0.0
        for k, c in segs.items():
            if c is None:
                continue
            w = weights.get(k, 0.0)
            if w <= 0:
                continue
            acc += w * c
            wsum += w
        if wsum <= 0:
            return COST_FAIL
        cost = acc / wsum

        # -------------------------
        # Knee alignment penalty (pos_up opening)
        # -------------------------
        m_knee_zone = m_pos_up & (V > 0.02) & (Iabs < 0.9*icomp_pos_fixed)
        v_k_meas = knee_v(V, Iabs,     m_knee_zone, I_KNEE)
        v_k_sim  = knee_v(V, Iabs_sim, m_knee_zone, I_KNEE)
        if (v_k_meas is not None) and (v_k_sim is not None):
            cost += KNEE_GAIN * abs(v_k_sim - v_k_meas)

        # -------------------------
        # Hysteresis penalties
        # -------------------------
        xh_range = float(np.max(xh) - np.min(xh))
        if xh_range < XH_RANGE_MIN:
            cost += XH_PEN_GAIN_RANGE * (XH_RANGE_MIN - xh_range)

        xh_hyst = float(abs(xh[-1] - xh[0]))
        if xh_hyst < XH_HYST_MIN:
            cost += XH_PEN_GAIN_HYST * (XH_HYST_MIN - xh_hyst)

        return float(cost)

    return objective


# =========================
# PSO core
# =========================
def pso_fit(
    V_meas,
    I_meas,
    n_particles,
    n_iters,
    objective_fn,
    seed_theta=None,
    tag="PSO",
    target_cost=None,
    min_iters_before_stop=1,
    run_seed=None,
    stag_iters=7,
    improve_eps=8e-4,
    reheat_frac=0.75,
    reheat_keep=2,
    reheat_jitter_scale=0.18,
    local_probe_count=0,
    local_probe_scale=0.0,
):
    local_seed = SEED if run_seed is None else int(run_seed)
    py_rng = random.Random(local_seed)
    np_rng = np.random.RandomState(local_seed)

    dim = len(MEM_PARAMS)
    swarm, vel = build_initial_swarm(n_particles=n_particles, seed_theta=seed_theta, seed=local_seed)

    if seed_theta is not None:
        print(f"[{tag}] seed loaded into particles 0..{min(4,n_particles-1)}")
    else:
        print(f"[{tag}] no seed (random start)")

    pbest = swarm.copy()
    pbest_cost = np.array([COST_FAIL] * n_particles, dtype=float)
    gbest = None
    gbest_cost = COST_FAIL

    best_prev = np.inf
    no_improve = 0

    for it in range(1, n_iters + 1):
        succ = 0
        fail = 0

        w = W_MAX - (W_MAX - W_MIN) * (it - 1) / max(1, (n_iters - 1))

        for i in range(n_particles):
            iter_id = (it - 1) * n_particles + i + 1
            cost = objective_fn(swarm[i], V_meas, I_meas, iter_id=iter_id)
            if cost >= COST_FAIL:
                fail += 1
            else:
                succ += 1
            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest[i] = swarm[i].copy()
            if cost < gbest_cost:
                gbest_cost = cost
                gbest = swarm[i].copy()

        print(f"[{tag}] iter={it}/{n_iters} best_cost={gbest_cost:.12g} succ={succ}/{n_particles} fail={fail}")

        if (
            target_cost is not None
            and it >= min_iters_before_stop
            and gbest is not None
            and gbest_cost <= target_cost
        ):
            print(f"[{tag}] target reached: best_cost={gbest_cost:.12g} <= {target_cost:.12g}")
            break

        if gbest_cost < best_prev - improve_eps:
            best_prev = gbest_cost
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= stag_iters and gbest is not None:
            # In refine, first do a tiny local probe around gbest before any larger reheat.
            if local_probe_count > 0 and local_probe_scale > 0:
                probe_best = gbest.copy()
                probe_best_cost = gbest_cost
                base_iter_id = 9000000 + it * 100 + local_probe_count
                for jj in range(local_probe_count):
                    cand = jitter_from_seed_rng(gbest, MEM_PARAMS, np_rng, scale=local_probe_scale)
                    cc = objective_fn(cand, V_meas, I_meas, iter_id=base_iter_id + jj)
                    if cc < probe_best_cost:
                        probe_best = cand.copy()
                        probe_best_cost = cc
                if probe_best_cost < gbest_cost:
                    print(f"[{tag}] [local-probe] best_cost {gbest_cost:.12g} -> {probe_best_cost:.12g}")
                    gbest = probe_best.copy()
                    gbest_cost = float(probe_best_cost)
                    swarm[0] = gbest.copy()
                    pbest[0] = gbest.copy()
                    pbest_cost[0] = gbest_cost
                    best_prev = gbest_cost
                    no_improve = 0
                    continue

            keep = min(reheat_keep, n_particles)
            kcount = max(0, int(reheat_frac * max(0, (n_particles - keep))))
            print(f"[{tag}] [reheat] stagnation -> randomize {100*reheat_frac:.0f}% swarm (keep 0..{max(keep-1,0)})")
            if kcount > 0:
                idx = np_rng.choice(np.arange(keep, n_particles), size=kcount, replace=False)
                for j in idx:
                    if py_rng.random() < 0.5:
                        swarm[j] = sample_uniform_rng(MEM_PARAMS, py_rng)
                    else:
                        swarm[j] = jitter_from_seed_rng(gbest, MEM_PARAMS, np_rng, scale=reheat_jitter_scale)
                pbest[idx] = swarm[idx].copy()
                pbest_cost[idx] = COST_FAIL
                vel[idx] = 0.0
            no_improve = 0

        if succ == 0:
            print(f"[{tag}] [WARN] all failed -> reseed around gbest")
            if gbest is not None:
                swarm[0] = gbest.copy()
                for k in range(1, min(10, n_particles)):
                    swarm[k] = jitter_from_seed_rng(swarm[0], MEM_PARAMS, np_rng, scale=ALL_FAIL_JITTER_SCALE)
                for k in range(10, n_particles):
                    swarm[k] = sample_uniform_rng(MEM_PARAMS, py_rng)
            else:
                swarm = np.stack([sample_uniform_rng(MEM_PARAMS, py_rng) for _ in range(n_particles)], axis=0)
            vel = np.zeros_like(swarm)
            continue

        r1 = np_rng.rand(n_particles, dim)
        r2 = np_rng.rand(n_particles, dim)
        vel = w * vel + C1 * r1 * (pbest - swarm) + C2 * r2 * (gbest - swarm)
        swarm = swarm + vel
        for i in range(n_particles):
            swarm[i] = clip_to_bounds(swarm[i], MEM_PARAMS)

    return gbest, float(gbest_cost)




def local_coordinate_polish(
    theta0: np.ndarray,
    cost0: float,
    objective_fn,
    V_meas: np.ndarray,
    I_meas: np.ndarray,
    tag: str = "LOCAL",
    param_order: list[str] | None = None,
    n_rounds: int = 3,
    linear_frac: float = 0.06,
    log_step_decades: float = 0.18,
    shrink: float = 0.55,
    accept_eps: float = 1e-5,
):
    """
    Deterministic local search after PSO.
    One parameter at a time, try +/- perturbations and keep improvements.
    This is slower than PSO per step, but much better at squeezing the last
    few hundredths once PSO has already found a decent basin.
    """
    if theta0 is None:
        return None, float(cost0)

    order = list(param_order) if param_order is not None else list(MEM_PARAMS)
    theta = clip_to_bounds(np.array(theta0, dtype=float).copy(), MEM_PARAMS)
    best_cost = float(cost0)
    eval_id = 80_000_000

    print(f"[{tag}] start local polish from cost={best_cost:.12g}")

    for rr in range(1, n_rounds + 1):
        round_improved = False
        lin_frac_r = linear_frac * (shrink ** (rr - 1))
        log_dec_r = log_step_decades * (shrink ** (rr - 1))

        print(f"[{tag}] round {rr}/{n_rounds} lin_frac={lin_frac_r:.4g} log_dec={log_dec_r:.4g}")

        for name in order:
            idx = MEM_PARAMS.index(name)
            lo, hi = BOUNDS[name]

            cands = []

            if name in LOG_PARAMS:
                mul = 10 ** log_dec_r
                cand_p = theta.copy()
                cand_m = theta.copy()
                cand_p[idx] = min(max(theta[idx] * mul, lo), hi)
                cand_m[idx] = min(max(theta[idx] / mul, lo), hi)
                cands = [cand_p, cand_m]
            else:
                step = lin_frac_r * (hi - lo)
                if step <= 0:
                    continue
                cand_p = theta.copy()
                cand_m = theta.copy()
                cand_p[idx] = min(max(theta[idx] + step, lo), hi)
                cand_m[idx] = min(max(theta[idx] - step, lo), hi)
                cands = [cand_p, cand_m]

            cand_best = None
            cand_best_cost = best_cost

            for cand in cands:
                # skip no-op moves
                if np.allclose(cand, theta, rtol=0.0, atol=0.0):
                    continue
                eval_id += 1
                cc = float(objective_fn(cand, V_meas, I_meas, iter_id=eval_id))
                if cc < cand_best_cost - accept_eps:
                    cand_best = cand.copy()
                    cand_best_cost = cc

            if cand_best is not None:
                print(f"[{tag}] {name}: {best_cost:.12g} -> {cand_best_cost:.12g}")
                theta = cand_best.copy()
                best_cost = cand_best_cost
                round_improved = True

        if not round_improved:
            print(f"[{tag}] no improvement in round {rr}")
            break

    print(f"[{tag}] final local polish cost={best_cost:.12g}")
    return theta, float(best_cost)


# =========================
# Final sim + plots
# =========================
def simulate_full(theta: np.ndarray, V_full: np.ndarray, icomp_pos: float, out_prefix="best"):
    ensure_dirs()

    N_full = len(V_full)
    t_full = make_time_vector(N_full, TSTOP_FULL)
    pwl_inline = pwl_inline_from_tv(t_full, V_full, pairs_per_line=8)

    deck_path = OUT_DIR / f"{out_prefix}.cir"
    log_path  = OUT_DIR / f"{out_prefix}.log"
    sim_name  = f"{out_prefix}_sim.dat"
    sim_path  = OUT_DIR / sim_name

    tstep_print = pick_tstep_print(N_full)
    islope = max(1e-12, ISLOPE_REL * icomp_pos)

    repl = {
        "@PWL_INLINE@": pwl_inline,
        "@SIMOUT@": sim_name,
        "@TSTEP@": f"{tstep_print:.12g}",
        "@DTMAX@": f"{DTMAX_SIM:.12g}",
        "@TSTOP@": f"{TSTOP_FULL:.12g}",

        "@KSW@": f"{KSW_FIXED:.12g}",
        "@RH0@": f"{RH0_FIXED:.12g}",
        "@RH_MIN@": f"{RH_MIN_FIXED:.12g}",
        "@RH_MAX@": f"{RH_MAX_FIXED:.12g}",
        "@VSLOPE@": f"{VSLOPE_FIXED:.12g}",

        "@ICOMP_POS@": f"{icomp_pos:.12g}",
        "@VCOMP@":     f"{VCOMP_FIXED:.12g}",
        "@RLO@":       f"{RLO_FIXED:.12g}",
        "@RHI@":       f"{RHI_FIXED:.12g}",
        "@ISLOPE@":    f"{islope:.12g}",
        "@VSLOPE_POS@":f"{VSLOPE_POS_FIX:.12g}",
    }

    for name, val in zip(MEM_PARAMS, theta):
        repl[f"@{name}@"] = f"{float(val):.12g}"

    render_template(deck_path, repl)

    rc = run_ngspice(deck_path, log_path, cwd=OUT_DIR, timeout_s=140)
    if rc != 0 or (not sim_path.exists()):
        print("[FULL] ngspice failed. log tail:\n" + tail_text(log_path))
        return False, None

    t, vcmd, vp, idev, vx, vxh = load_wrdata(sim_path)
    df = pd.DataFrame({"time": t, "Vcmd": vcmd, "Vp": vp, "I": idev, "x": vx, "xh": vxh})
    df.to_csv(OUT_DIR / f"{out_prefix}_I_sim_full.csv", index=False)
    return True, df

def make_plots(V_meas, I_meas, df_sim: pd.DataFrame, tag="best"):
    import matplotlib.pyplot as plt

    Vcmd = df_sim["Vcmd"].to_numpy(float)
    Vp   = df_sim["Vp"].to_numpy(float)
    Isim = df_sim["I"].to_numpy(float)

    # abs semilogy (Vcmd)
    plt.figure()
    plt.semilogy(V_meas, np.abs(I_meas)+I_FLOOR, ".", label="meas")
    plt.semilogy(Vcmd,   np.abs(Isim)+I_FLOOR,   ".", label="sim (Vcmd)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlabel("Vcmd (V)")
    plt.ylabel("|I| (A)")
    plt.title("log(|I|): measured vs simulated (x=Vcmd)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_plot_abs_Vcmd.png", dpi=170)
    plt.close()

    # abs semilogy (Vp)
    plt.figure()
    plt.semilogy(V_meas, np.abs(I_meas)+I_FLOOR, ".", label="meas")
    plt.semilogy(Vp,     np.abs(Isim)+I_FLOOR,   ".", label="sim (Vp)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlabel("Vp (V)")
    plt.ylabel("|I| (A)")
    plt.title("log(|I|): measured vs simulated (x=Vp)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_plot_abs_Vp.png", dpi=170)
    plt.close()

    # signed symlog (Vcmd)
    plt.figure()
    plt.yscale("symlog", linthresh=1e-9)
    plt.plot(V_meas, I_meas, ".", label="meas")
    plt.plot(Vcmd,   Isim,   ".", label="sim (Vcmd)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlabel("Vcmd (V)")
    plt.ylabel("I (A)")
    plt.title("I-V (symlog): measured vs simulated (x=Vcmd)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_plot_symlog_Vcmd.png", dpi=170)
    plt.close()

    # signed symlog (Vp)
    plt.figure()
    plt.yscale("symlog", linthresh=1e-9)
    plt.plot(V_meas, I_meas, ".", label="meas")
    plt.plot(Vp,     Isim,   ".", label="sim (Vp)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlabel("Vp (V)")
    plt.ylabel("I (A)")
    plt.title("I-V (symlog): measured vs simulated (x=Vp)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_plot_symlog_Vp.png", dpi=170)
    plt.close()


def make_comparison_plots(V_meas, I_meas, df_init: pd.DataFrame, df_best: pd.DataFrame, tag="init_vs_best"):
    import matplotlib.pyplot as plt

    Vcmd_i = df_init["Vcmd"].to_numpy(float)
    I_i    = df_init["I"].to_numpy(float)
    Vcmd_b = df_best["Vcmd"].to_numpy(float)
    I_b    = df_best["I"].to_numpy(float)

    plt.figure()
    plt.semilogy(V_meas, np.abs(I_meas)+I_FLOOR, ".", label="meas")
    plt.semilogy(Vcmd_i, np.abs(I_i)+I_FLOOR, ".", label="initial")
    plt.semilogy(Vcmd_b, np.abs(I_b)+I_FLOOR, ".", label="fitted")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlabel("Vcmd (V)")
    plt.ylabel("|I| (A)")
    plt.title("log(|I|): measured vs initial vs fitted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_plot_abs_Vcmd.png", dpi=170)
    plt.close()

    plt.figure()
    plt.yscale("symlog", linthresh=1e-9)
    plt.plot(V_meas, I_meas, ".", label="meas")
    plt.plot(Vcmd_i, I_i, ".", label="initial")
    plt.plot(Vcmd_b, I_b, ".", label="fitted")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.xlabel("Vcmd (V)")
    plt.ylabel("I (A)")
    plt.title("I-V (symlog): measured vs initial vs fitted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_plot_symlog_Vcmd.png", dpi=170)
    plt.close()

def export_param_compare(theta_init: np.ndarray, theta_final: np.ndarray, fname="param_compare_init_vs_final.csv"):
    rows = []
    for k, v0, v1 in zip(MEM_PARAMS, theta_init, theta_final):
        v0 = float(v0)
        v1 = float(v1)
        if abs(v0) > 1e-300:
            rel = (v1 - v0) / abs(v0)
        else:
            rel = np.nan
        rows.append({
            "param": k,
            "initial": v0,
            "final": v1,
            "delta": v1 - v0,
            "relative_change": rel,
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / fname, index=False, encoding="utf-8-sig")

def export_fit_compare_summary(summary: dict, fname="fit_compare_summary.csv"):
    pd.DataFrame([summary]).to_csv(OUT_DIR / fname, index=False, encoding="utf-8-sig")

def export_theta(theta: np.ndarray, icomp_pos: float, fname="theta_best.csv"):
    rows = [(k, float(v)) for k, v in zip(MEM_PARAMS, theta)]
    rows += [("ICOMP_POS_FIXED", float(icomp_pos)),
             ("RHI_FIXED", float(RHI_FIXED)),
             ("VCOMP_FIXED", float(VCOMP_FIXED)),
             ("VSLOPE_POS_FIXED", float(VSLOPE_POS_FIX)),
             ("ISLOPE_REL", float(ISLOPE_REL))]
    pd.DataFrame(rows, columns=["param","value"]).to_csv(OUT_DIR / fname, index=False)


# =========================
# Main
# =========================
def main():
    ensure_dirs()
    write_template()

    if not CSV_MEAS.exists():
        raise FileNotFoundError(f"Missing {CSV_MEAS}")
    if not NGSPICE.exists():
        raise FileNotFoundError(f"Missing {NGSPICE} (edit NGSPICE path in script)")

    V_full, I_full = read_meas_csv(CSV_MEAS)

    ic_est = estimate_icomp_pos(V_full, I_full)
    print("BASE_DIR :", BASE_DIR)
    print("CSV_MEAS :", CSV_MEAS)
    print("NGSPICE  :", NGSPICE)
    print("OUT_DIR  :", OUT_DIR)
    print(f"[meas] points={len(V_full)} Vmin={V_full.min():.6g} Vmax={V_full.max():.6g}")
    print(f"[meas] suggested tran print step = {pick_tstep_print(len(V_full)):.3g} s")
    print(f"[compliance] ICOMP_FIXED≈{ic_est:.4g}  RHI_FIXED={RHI_FIXED:.3g}  VCOMP_FIXED={VCOMP_FIXED}  ISLOPE_REL={ISLOPE_REL}")
    print(f"[seed] INIT_MODE={INIT_MODE}")

    if DEBUG_MASK_PLOTS:
        dbg_dir = OUT_DIR / "mask_debug"
        masks0, info0 = _build_masks_meas_only(V_full, I_full, icomp_pos_fixed=ic_est, I_FLOOR=I_FLOOR, V0_WINDOW=V0_WINDOW, I_KNEE=I_KNEE)
        _plot_masks_overlay(V_full, I_full, masks0, info0, outdir=dbg_dir, tag="MEAS_FULL", I_FLOOR=I_FLOOR)

    # Make fast subset
    V_fit, I_fit = make_fit_subset(V_full, I_full, target_n=COARSE_SUBSET_TARGET_N)
    print(f"[fit-subset] using {len(V_fit)} points for coarse stage (target={COARSE_SUBSET_TARGET_N})")

    # Seed / random init switch
    seed_theta = resolve_initial_seed(INIT_MODE, MEM_PARAMS)

    # -------------------------
    # Coarse (must pass threshold before refine)
    # -------------------------
    obj_c = objective_factory(ic_est, label="COARSE")

    init_theta = None
    init_subset_cost = None
    init_full_cost = None
    init_df_sim = None

    if EXPORT_INITIAL_BASELINE:
        print("\n[INIT] evaluate initial population before any PSO update")
        init_swarm, _ = build_initial_swarm(
            n_particles=N_PARTICLES_C,
            seed_theta=seed_theta,
            seed=SEED,
        )
        init_info = evaluate_initial_population(
            init_swarm,
            obj_c,
            V_fit,
            I_fit,
            pick_mode=INITIAL_BASELINE_PICK,
            iter_offset=900000,
        )
        init_theta = init_info["theta"].copy()
        init_subset_cost = float(init_info["cost"])
        print(f"[INIT] picked idx={init_info['index']} subset_cost={init_subset_cost:.12g}")

        export_theta(init_theta, ic_est, fname="theta_init_population_best.csv")
        ok_init, init_df_sim = simulate_full(init_theta, V_full, ic_est, out_prefix="init_population_best")
        if ok_init:
            make_plots(V_full, I_full, init_df_sim, tag="init_population_best")
            obj_init_full = objective_factory(ic_est, label="INIT_FULL")
            init_full_cost = float(obj_init_full(init_theta, V_full, I_full, iter_id=990001))
            print(f"[INIT] full_cost={init_full_cost:.12g}")
        else:
            print("[INIT] FAILED to create initial full simulation output")

    # -------------------------
    # Coarse (must pass threshold before refine)
    # -------------------------

    theta_c = seed_theta
    cost_c = COST_FAIL
    coarse_round = 0
    coarse_force_refine = False
    coarse_cost_hist = []

    while cost_c > COARSE_TARGET_COST and coarse_round < COARSE_MAX_ROUNDS:
        coarse_round += 1
        print(f"\n[COARSE] batch {coarse_round}/{COARSE_MAX_ROUNDS}")

        theta_c, cost_c = pso_fit(
            V_fit,
            I_fit,
            N_PARTICLES_C,
            N_ITERS_C,
            obj_c,
            seed_theta=theta_c,
            tag=f"COARSE_B{coarse_round}",
            target_cost=COARSE_TARGET_COST,
            min_iters_before_stop=COARSE_MIN_ITERS,
            run_seed=SEED + 1000 * coarse_round,
            stag_iters=COARSE_STAG_ITERS,
            improve_eps=COARSE_IMPROVE_EPS,
            reheat_frac=COARSE_REHEAT_FRAC,
            reheat_keep=COARSE_REHEAT_KEEP,
            reheat_jitter_scale=COARSE_REHEAT_JITTER_SCALE,
        )
        print(f"[COARSE] batch {coarse_round} done, best_cost={cost_c:.12g}")

        coarse_cost_hist.append(float(cost_c))
        if FORCE_REFINE_IF_COARSE_STALLED and len(coarse_cost_hist) >= COARSE_STALL_BATCH_WINDOW:
            win = coarse_cost_hist[-COARSE_STALL_BATCH_WINDOW:]
            if (max(win) - min(win) <= COARSE_STALL_EPS) and (min(win) <= COARSE_FORCE_REFINE_MAX_COST):
                print(f"[COARSE] stalled across {COARSE_STALL_BATCH_WINDOW} batches near cost={cost_c:.12g} -> force enter REFINE")
                coarse_force_refine = True
                break

    if theta_c is None:
        raise RuntimeError("Coarse stage failed: theta_c is None")

    if cost_c > COARSE_TARGET_COST and not coarse_force_refine:
        print(f"[STOP] coarse best_cost={cost_c:.12g} still > target={COARSE_TARGET_COST:.12g}")
        print("[STOP] not entering REFINE")
        export_theta(theta_c, ic_est, fname="theta_coarse_stop.csv")
        ok, df_sim = simulate_full(theta_c, V_full, ic_est, out_prefix="coarse_stop")
        final_full_cost = None
        if ok:
            make_plots(V_full, I_full, df_sim, tag="coarse_stop")
            obj_final_full = objective_factory(ic_est, label="COARSE_STOP_FULL")
            final_full_cost = float(obj_final_full(theta_c, V_full, I_full, iter_id=990002))
            if init_df_sim is not None:
                make_comparison_plots(V_full, I_full, init_df_sim, df_sim, tag="init_vs_coarse_stop")
            if init_theta is not None:
                export_param_compare(init_theta, theta_c, fname="param_compare_init_vs_coarse_stop.csv")
            export_fit_compare_summary({
                "mode": INIT_MODE,
                "initial_pick": INITIAL_BASELINE_PICK,
                "initial_subset_cost": init_subset_cost,
                "initial_full_cost": init_full_cost,
                "final_stage": "coarse_stop",
                "final_cost_stage_native": cost_c,
                "final_full_cost": final_full_cost,
                "full_cost_improvement_ratio": (init_full_cost / final_full_cost) if (init_full_cost is not None and final_full_cost not in [None, 0]) else np.nan,
            }, fname="fit_compare_summary.csv")
            print("\nOutputs:")
            print(" ", OUT_DIR / "theta_coarse_stop.csv")
            print(" ", OUT_DIR / "coarse_stop.cir")
            print(" ", OUT_DIR / "coarse_stop.log")
            print(" ", OUT_DIR / "coarse_stop_sim.dat")
            print(" ", OUT_DIR / "coarse_stop_I_sim_full.csv")
            print(" ", OUT_DIR / "coarse_stop_plot_abs_Vcmd.png")
            print(" ", OUT_DIR / "coarse_stop_plot_abs_Vp.png")
            print(" ", OUT_DIR / "coarse_stop_plot_symlog_Vcmd.png")
            print(" ", OUT_DIR / "coarse_stop_plot_symlog_Vp.png")
            if init_df_sim is not None:
                print(" ", OUT_DIR / "init_vs_coarse_stop_plot_abs_Vcmd.png")
                print(" ", OUT_DIR / "init_vs_coarse_stop_plot_symlog_Vcmd.png")
            if init_theta is not None:
                print(" ", OUT_DIR / "param_compare_init_vs_coarse_stop.csv")
            print(" ", OUT_DIR / "fit_compare_summary.csv")
        else:
            print("[coarse_stop] FAILED to create full sim output")
        return

    print(f"[COARSE] enter REFINE, best_cost={cost_c:.12g}, forced={coarse_force_refine}")

    # -------------------------
    # Refine (full, seeded from coarse) - multi-batch
    # -------------------------
    obj_r = objective_factory(ic_est, label="REFINE")

    theta_r = theta_c.copy()
    cost_r = COST_FAIL
    refine_round = 0

    while refine_round < REFINE_MAX_ROUNDS:
        refine_round += 1
        print(f"\n[REFINE] batch {refine_round}/{REFINE_MAX_ROUNDS}")

        theta_r, cost_r = pso_fit(
            V_full,
            I_full,
            N_PARTICLES_R,
            N_ITERS_R,
            obj_r,
            seed_theta=theta_r,
            tag=f"REFINE_B{refine_round}",
            target_cost=REFINE_TARGET_COST,
            min_iters_before_stop=REFINE_MIN_ITERS,
            run_seed=SEED + 50000 + 1000 * refine_round,
            stag_iters=REFINE_STAG_ITERS,
            improve_eps=REFINE_IMPROVE_EPS,
            reheat_frac=REFINE_REHEAT_FRAC,
            reheat_keep=REFINE_REHEAT_KEEP,
            reheat_jitter_scale=REFINE_REHEAT_JITTER_SCALE,
            local_probe_count=REFINE_LOCAL_PROBE_COUNT,
            local_probe_scale=REFINE_LOCAL_PROBE_SCALE,
        )

        print(f"[REFINE] batch {refine_round} done, best_cost={cost_r:.12g}")

        if REFINE_TARGET_COST is not None and cost_r <= REFINE_TARGET_COST:
            print(f"[REFINE] reached target -> stop refine")
            break

    print(f"[REFINE] final best_cost={cost_r:.12g}")

    if FINAL_LOCAL_POLISH and theta_r is not None:
        theta_lp, cost_lp = local_coordinate_polish(
            theta_r,
            cost_r,
            obj_r,
            V_full,
            I_full,
            tag="LOCAL_POLISH",
            param_order=LOCAL_POLISH_PARAM_ORDER,
            n_rounds=LOCAL_POLISH_ROUNDS,
            linear_frac=LOCAL_POLISH_LINEAR_FRAC,
            log_step_decades=LOCAL_POLISH_LOG_STEP_DECADES,
            shrink=LOCAL_POLISH_SHRINK,
            accept_eps=LOCAL_POLISH_ACCEPT_EPS,
        )
        if theta_lp is not None and cost_lp < cost_r:
            print(f"[LOCAL_POLISH] improved best_cost {cost_r:.12g} -> {cost_lp:.12g}")
            theta_r, cost_r = theta_lp, cost_lp
        else:
            print(f"[LOCAL_POLISH] no further improvement")

    export_theta(theta_r, ic_est, fname="theta_best.csv")

    ok, df_sim = simulate_full(theta_r, V_full, ic_est, out_prefix="best")
    if ok:
        make_plots(V_full, I_full, df_sim, tag="best")
        obj_final_full = objective_factory(ic_est, label="BEST_FULL")
        final_full_cost = float(obj_final_full(theta_r, V_full, I_full, iter_id=990003))
        if init_df_sim is not None:
            make_comparison_plots(V_full, I_full, init_df_sim, df_sim, tag="init_vs_best")
        if init_theta is not None:
            export_param_compare(init_theta, theta_r, fname="param_compare_init_vs_best.csv")
        export_fit_compare_summary({
            "mode": INIT_MODE,
            "initial_pick": INITIAL_BASELINE_PICK,
            "initial_subset_cost": init_subset_cost,
            "initial_full_cost": init_full_cost,
            "final_stage": "refine",
            "final_cost_stage_native": cost_r,
            "final_full_cost": final_full_cost,
            "full_cost_improvement_ratio": (init_full_cost / final_full_cost) if (init_full_cost is not None and final_full_cost not in [None, 0]) else np.nan,
        }, fname="fit_compare_summary.csv")
        print("\nOutputs:")
        print(" ", OUT_DIR / "theta_best.csv")
        print(" ", OUT_DIR / "best.cir")
        print(" ", OUT_DIR / "best.log")
        print(" ", OUT_DIR / "best_sim.dat")
        print(" ", OUT_DIR / "best_I_sim_full.csv")
        print(" ", OUT_DIR / "best_plot_abs_Vcmd.png")
        print(" ", OUT_DIR / "best_plot_abs_Vp.png")
        print(" ", OUT_DIR / "best_plot_symlog_Vcmd.png")
        print(" ", OUT_DIR / "best_plot_symlog_Vp.png")
        if init_df_sim is not None:
            print(" ", OUT_DIR / "init_vs_best_plot_abs_Vcmd.png")
            print(" ", OUT_DIR / "init_vs_best_plot_symlog_Vcmd.png")
        if init_theta is not None:
            print(" ", OUT_DIR / "theta_init_population_best.csv")
            print(" ", OUT_DIR / "init_population_best.cir")
            print(" ", OUT_DIR / "init_population_best.log")
            print(" ", OUT_DIR / "init_population_best_sim.dat")
            print(" ", OUT_DIR / "init_population_best_I_sim_full.csv")
            print(" ", OUT_DIR / "init_population_best_plot_abs_Vcmd.png")
            print(" ", OUT_DIR / "init_population_best_plot_abs_Vp.png")
            print(" ", OUT_DIR / "init_population_best_plot_symlog_Vcmd.png")
            print(" ", OUT_DIR / "init_population_best_plot_symlog_Vp.png")
            print(" ", OUT_DIR / "param_compare_init_vs_best.csv")
        print(" ", OUT_DIR / "fit_compare_summary.csv")
    else:
        print("[best] FAILED to create full sim output")

if __name__ == "__main__":
    main()
