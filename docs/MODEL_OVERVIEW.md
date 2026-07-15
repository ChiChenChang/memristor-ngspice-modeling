# Model Overview

## 1. Terminal Conduction

The compact model separates terminal conduction from internal-state evolution. The current branch uses a state-dependent current scale and nonlinear voltage response. A bounded state representation prevents the conduction equation from becoming unstable when the raw state node moves outside its intended physical interval.

Typical model roles include:

- current magnitude controlled by state-dependent `I0`
- nonlinear slope controlled by state-dependent `alpha`
- positive/negative asymmetry controlled by a weighting parameter
- leakage and parasitic paths for circuit-level robustness

## 2. Internal State

The state variable represents the device's switching condition, ranging conceptually from high resistance to low resistance. Three development variants are included.

### Material-rate state model

Uses explicit voltage-activated SET and RESET rates. The state equation includes saturation terms and optional retention leakage.

### Direct Rh state path

Maps terminal voltage to a dynamic resistance. The state capacitor is directly driven through this resistance, with monitor nodes provided for `Rh`, the core state derivative, limiter contribution, and total derivative.

### Target-state model

Defines a smooth voltage-dependent target state. SET and RESET time constants are blended according to polarity, and the state approaches the target through an effective dynamic resistance.

## 3. Parameter Extraction

The fitting script performs a two-stage guided PSO search:

- coarse optimization on a reduced data subset
- refinement on the full measurement set
- reheating and local probing when progress stalls
- optional deterministic local polishing

The objective uses region-specific weighting so that one high-current branch does not dominate the entire fit.

## 4. Validation

The analysis scripts calculate and export quantities such as:

- signed and absolute hysteresis-loop area
- internal-state range and end-state recovery
- branch separation at selected voltages
- near-zero current behavior
- frequency and sweep-rate dependence
- voltage-amplitude and repeated-cycle response
