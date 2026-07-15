# Model overview

## Common compact-model framework

All implementations separate the terminal-current equation from the internal
state equation. The raw state `x` is integrated by an internal SPICE node and is
mapped to a bounded effective state `xh` before it enters the current equation.

```math
x_h = \operatorname{softclip}(x)
```

The state-dependent transport terms are

```math
I_0(x_h)=I_{\max}x_h+I_{\min}(1-x_h)
```

```math
\alpha(x_h)=\alpha_{\max}x_h+\alpha_{\min}(1-x_h)
```

```math
I_{\mathrm{diff}}(V,x_h)=
\exp\!\left(\beta\alpha(x_h)V\right)
-
\exp\!\left(-(1-\beta)\alpha(x_h)V\right)
```

and the terminal current is represented as

```math
I(V,x_h)=I_{\mathrm{scale}}
\left[I_0(x_h)I_{\mathrm{diff}}(V,x_h)+E_I V\right]
+\frac{V}{R_{\mathrm{off}}}.
```

Protected exponentials, smooth polarity functions, finite state bounds, a small
series resistance, and Gear integration are used to improve transient
convergence.

## State-equation variants

### DMM-based relaxation

The DMM-oriented implementation maps terminal voltage into a smooth target
state and voltage-dependent SET/RESET time constants:

```math
A(V)=\frac{1}{2}\left[1+\tanh\left(\frac{V}{V_{\mathrm{slope}}}\right)\right]
```

```math
\frac{dx}{dt}=\frac{A(V)-x}{\tau(V)},
\qquad \tau(V)=R_h(V)C_h.
```

This is the baseline target-state relaxation formulation.

### Direct terminal-voltage-driven state path

This implementation removes the explicit `A(V)` target node and lets terminal
voltage activate the state-update path more directly. Smooth voltage gates and
boundary factors suppress low-voltage drift and prevent uncontrolled motion
outside the useful state range. It is used to study how the voltage-to-state
mapping changes switching sharpness, state recovery, and numerical behavior.

### Yakopcic-like threshold-rate state equation

The third implementation uses separate threshold-activated SET and RESET rates:

```math
\frac{dx}{dt}=
 r_{\mathrm{SET}}(V)(1-x_h)^{p_{\mathrm{SET}}}
-r_{\mathrm{RESET}}(V)x_h^{p_{\mathrm{RESET}}}.
```

The implementation is described as **Yakopcic-like** because the threshold-rate
concept is adapted to the common transport equation and NGSpice framework used
in this repository; it is not claimed to be an exact reproduction of the
original model.

## Eight equation-term families

`analyze_equation_term_sweep.py` holds the external waveform fixed and studies
how the following equation-term families propagate into `x`, `xh`, and the final
I-V response:

| Term family | Main members | Main role |
|---|---|---|
| `I0_TERM` | `IMAX`, `IMIN` | State-dependent conduction scale |
| `ACOEF_TERM` | `ALPHA_MAX`, `ALPHA_MIN` | State-dependent nonlinear slope |
| `IDIFF_TERM` | `BETAA` | Positive/negative exponential asymmetry |
| `SET_TAU_TERM` | `VSET`, `ETA_SET` | SET onset and switching rate |
| `RESET_TAU_TERM` | `VRES`, `ETA_RES` | RESET onset and switching rate |
| `RH_TERM` | `RH0`, `VSLOPE` | Effective state-update resistance |
| `MEMORY_TERM` | `CH0`, `H0` | State time scale and initial condition |
| `CURRENT_SCALE_LEAK_TERM` | `ISCALE`, `EI`, `ROFF` | Overall current and leakage branches |

This term-level-to-result-level analysis is intended to make the compact model
more interpretable than a parameter table alone.
