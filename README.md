# Dynamic Compact Modeling and NGSpice Circuit Implementation of Bipolar Memristive Devices 

A reproducible Python + NGSpice workflow for dynamic compact modeling of bipolar
memristive devices. The project connects measured I-V data, SPICE-in-the-loop
parameter extraction, internal-state analysis, operating-condition validation,
and equation-term interpretation.

The repository is based on the research project **Dynamic Compact Modeling and
NGSpice Circuit Implementation of Bipolar Memristive Devices**.

## What this repository demonstrates

- A dynamic memristive compact model implemented as an NGSpice-compatible
  subcircuit.
- Coarse-to-refined particle-swarm parameter extraction from a complete bipolar
  I-V trajectory.
- Validation under different sweep frequencies, voltage amplitudes, voltage
  sequences, and repeated cycles.
- Equation-term sensitivity analysis linking model terms to `x`, `xh`, and the
  final hysteretic I-V response.
- Comparison of three state-update philosophies within a common transport and
  simulation framework.

## Repository structure

```text
memristor-ngspice-modeling/
├── README.md
├── requirements.txt
├── fit_memristor_pso.py
├── analyze_operating_condition_sweeps.py
├── analyze_equation_term_sweep.py
├── models/
│   ├── model_dmm_relaxation.cir
│   ├── model_direct_voltage_state.cir
│   └── model_yakopcic_like_rate.cir
├── data/
│   ├── README.md
│   └── example_synthetic.csv
├── docs/
│   └── MODEL_OVERVIEW.md
└── .github/workflows/
    └── python-syntax.yml
```

Generated simulation and fitting outputs are written under `results/` and are
excluded from version control.

## Core workflow

### 1. Parameter fitting

`fit_memristor_pso.py` performs a two-stage PSO workflow:

1. coarse fitting on a reduced trajectory;
2. refined fitting on the complete I-V data.

The fitted parameters are:

```text
IMAX, IMIN, ALPHA_MAX, ALPHA_MIN, BETAA,
VSET, VRES, ETA_SET, ETA_RES,
CH0, ISCALE, H0, EI, ROFF
```

The main outputs are:

```text
results/fit/fitdeck_embedded.cir
results/fit/theta_best.csv
results/fit/best_simulation.csv
results/fit/fit_summary.csv
```

### 2. Operating-condition analysis

`analyze_operating_condition_sweeps.py` keeps the fitted model parameters fixed
and changes external operating conditions:

- sweep frequency / time scale;
- positive-voltage amplitude;
- negative-voltage amplitude;
- progressive and reverse multi-negative sequences;
- repeated negative-voltage cycles;
- optional staircase input;
- optional replay of the measurement voltage trajectory.

It generates per-case SPICE decks, logs, simulation tables, state plots,
grouped overlays, branch metrics, hysteresis metrics, and frequency summaries.

### 3. Equation-term sensitivity analysis

`analyze_equation_term_sweep.py` keeps the voltage waveform fixed while varying
eight equation-term families:

```text
I0_TERM
ACOEF_TERM
IDIFF_TERM
SET_TAU_TERM
RESET_TAU_TERM
RH_TERM
MEMORY_TERM
CURRENT_SCALE_LEAK_TERM
```

This analysis traces the effect of each term from the equation level to the
internal state and final I-V response.

## SPICE model implementations

The `models/` directory contains three related state-equation implementations.
They are not three unrelated devices; they are controlled variants used to
compare how terminal voltage is translated into internal-state evolution.

### 1. DMM-based relaxation model

The baseline uses a smooth polarity target and voltage-dependent relaxation
time constant:

```math
\frac{dx}{dt}=\frac{A(V)-x}{\tau(V)},
\qquad \tau(V)=R_h(V)C_h.
```

### 2. Direct terminal-voltage-driven state model

This variant removes the explicit target-state node and lets `V(p,n)` activate
the state-update path more directly. Smooth threshold gates and state-boundary
factors are used to suppress low-voltage drift and maintain numerical stability.

### 3. Yakopcic-like threshold-rate model

This variant replaces the relaxation equation with threshold-activated SET and
RESET rates:

```math
\frac{dx}{dt}=r_{\mathrm{SET}}(V,x_h)-r_{\mathrm{RESET}}(V,x_h).
```

It is described as **Yakopcic-like**, rather than as an exact reproduction,
because the state-update concept is adapted to the common memdiode transport
and NGSpice implementation framework used here.

A detailed equation-level description is available in
[`docs/MODEL_OVERVIEW.md`](docs/MODEL_OVERVIEW.md).

## Requirements

- Python 3.10 or newer
- NGSpice available on the system `PATH`, or `ngspice.exe` placed in the
  repository root on Windows
- Python packages listed in `requirements.txt`

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start

### Use the public synthetic example

`data/example_synthetic.csv` is generated demonstration data, not an
experimental result. The fitting script automatically uses it when
`data/DC-IV.csv` is absent.

Run fitting:

```bash
python fit_memristor_pso.py
```

The analysis scripts require the fitted artifacts created above. The operating
condition analyzer uses `data/DC-IV.csv` when available and otherwise falls back
to the public synthetic example.

```bash
python analyze_operating_condition_sweeps.py
python analyze_equation_term_sweep.py
```

### Use private measurement data

Place a two-column voltage/current file at:

```text
data/DC-IV.csv
```

The repository's `.gitignore` excludes this path by default so laboratory data
are not accidentally committed.

## Standalone NGSpice examples

Each model deck can also be run independently:

```bash
ngspice -b models/model_dmm_relaxation.cir
ngspice -b models/model_direct_voltage_state.cir
ngspice -b models/model_yakopcic_like_rate.cir
```

## Reproducibility notes

- PSO initialization uses a fixed random seed, but small differences can still
  occur across NGSpice versions, operating systems, and numerical settings.
- The fitting deck uses smooth functions, protected exponentials, bounded state
  mapping, finite leakage paths, and Gear integration for convergence.
- `Vcmd` denotes the commanded waveform, while `Vp` denotes the actual device
  terminal voltage after the sense/compliance path.
- Real measurement data and large generated result directories are intentionally
  excluded from the public repository.

## Author

**Chi-Chen Chang (Cora)**  
M.S. research in semiconductor-device compact modeling and NGSpice circuit
implementation.
