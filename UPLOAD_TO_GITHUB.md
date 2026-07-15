# Upload checklist

1. Download and extract the prepared repository package.
2. In the existing GitHub repository, remove obsolete duplicate files such as
   the separate frequency and voltage analyzers.
3. Upload the contents of this folder to the repository root. Upload the
   contents, not an extra outer folder.
4. Confirm that these three files are visible at the top level:

```text
fit_memristor_pso.py
analyze_operating_condition_sweeps.py
analyze_equation_term_sweep.py
```

5. Confirm that `.github/workflows/python-syntax.yml` is present. GitHub may
   hide dot-prefixed folders in some upload dialogs; uploading the ZIP through
   Git, GitHub Desktop, or a local clone preserves it reliably.
6. Commit with a message such as:

```text
Reorganize memristor NGSpice modeling workflow
```

7. Open the **Actions** tab and verify that `Python syntax check` passes.

Do not upload private `data/DC-IV.csv`, `ngspice.exe`, or generated `results/`
folders.
