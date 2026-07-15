# Data directory

The fitting and operating-condition scripts look for the private measurement
file at:

```text
data/DC-IV.csv
```

The file must contain at least two columns. The first column is the applied
voltage in volts and the second is the measured current in amperes. Column names
may be arbitrary because the scripts read the first two columns.

Measured laboratory data are intentionally excluded from this public
repository. `example_synthetic.csv` is a generated bipolar hysteresis trajectory
provided only to demonstrate the expected file format and program flow. It is
not an experimental result and should not be used to evaluate model accuracy.

To use your own data, place it at `data/DC-IV.csv`. The `.gitignore` file prevents
that path from being committed by default.
