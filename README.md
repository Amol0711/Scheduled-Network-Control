# Scheduled network control — numerical companion

This repository snapshot contains the computational methods, simulation code,
fixed configurations and archived numerical evidence for scheduled control of
six-node Stuart–Landau networks, together with scalar schedule/noise calculations.
It contains no article manuscript, mathematical supplement, author list, affiliations,
funding records or private revision history.

## Start with integrity verification

From the repository root, run

```bash
python3 -B verify_repository.py
python3 -B simulation/companion_1/reproduce.py verify
```

Both commands use only the Python standard library, read the sealed files, and
write no result or cache. The first verifies the whole repository payload. The
second verifies the unchanged 129-file numerical companion. **Do not omit
`verify` from the second command.** The original launcher defaults to `quick`,
which recomputes statistics and certificate diagnostics.

The accompanying article and mathematical supplement are separate documents.
The [current result map](docs/CURRENT_RESULT_MAP.md) connects their figures, tables
and revised labels to the archived numerical records. The computational methods
PDF below is not the article or its mathematical supplement.

Start reading the [computational companion](simulation/companion_1/docs/COMPUTATIONAL_COMPANION.md),
its [PDF](simulation/companion_1/docs/COMPUTATIONAL_COMPANION.pdf), or the offline
[HTML landing page](simulation/companion_1/index.html).
The companion's local sections C01–C09 and tables C1–C4 provide models, protocols,
complete statistical tables and numerical qualifications. HTML/MathML and PDF
preview support varies by host; the Markdown and downloadable files remain available.

## Repository organization

| Path | Contents |
| --- | --- |
| `simulation/companion_1/` | Exact original 129-file computational snapshot, including 32 Python files, the shell launcher, dependency pins, numerical records and methods documentation. |
| `verified_current/journal/source/data/` | Twelve unchanged publication plotting CSVs only. The inherited directory layout is retained for the original exporter interfaces; it contains no manuscript source. |
| `tools/export_fg2_plot_data.py` | Original plotting transforms for the deterministic figure and graph-rate residuals. |
| `tools/export_fg3_plot_data.py` | Original scalar-ratio and network-interval display transforms. |
| `docs/CURRENT_RESULT_MAP.md` | Current figure/table locations and the historical-label crosswalk. |
| `docs/EXECUTION_SCOPE.md` | Effects, prerequisites and output requirements of every available route. |
| `SHA256SUMS.txt` | SHA-256 inventory for every repository payload file except this checksum file itself. |

The nested `companion-1` release descriptor, dates, original README and record-location
maps describe the frozen computational snapshot. They have not been rewritten as
current hosting claims. This root README and the current result map explain the new
wrapper. The original companion's internal checksum list and all scientific bytes
remain unchanged.

## Reproduction routes and dependencies

The recorded numerical environment is CPython 3.13.5 on Linux x86_64, with NumPy
2.3.5, SciPy 1.17.0, SymPy 1.14.0, mpmath 1.3.0, Numba 0.65.1 and llvmlite 0.47.0.
These are the inherited reproducibility pins, not a statement of current latest
versions or a new cross-platform execution test.

Install dependencies into an environment **outside this repository**. Installation
may require network access. The simulation programs do not download dependencies.
For example, from the repository root:

```bash
python3.13 -m venv ../simulation_env
../simulation_env/bin/python -m pip install -r simulation/companion_1/requirements.txt
../simulation_env/bin/python -B simulation/companion_1/reproduce.py quick --output-dir ../results
```

`quick` generates no trajectories but does recompute statistics and certificate
checks. It is not an integrity-only test. The reduced dependency file
`requirements-quick.txt` is available for quick/statistics-only environments.
Use the [execution scope](docs/EXECUTION_SCOPE.md) before selecting a route.
Always supply an explicit output directory outside the **entire repository**.
The original launcher protects the nested companion, not the wrapper around it.

The primary experiment comprises 20,000 paths over 16 caps with common random
numbers, seed 20260829 and 200-path batching. Its nested 10,000-path coarse/fine
study is dependent. The supplied NPZ stores terminal squared-disagreement arrays,
not every intermediate trajectory or every Brownian increment. Replay generates
its recorded stream from the frozen implementation. Sampling intervals describe
fixed-step, fixed-grid quantities, not continuous-time optimizer uncertainty.

## Plotting-only export interfaces

The twelve publication CSVs are already supplied. Regeneration is optional and is
not needed for inspection or verification. To reproduce only the display transforms,
use new, empty, external output directories:

```bash
python3 -B tools/export_fg2_plot_data.py --source verified_current/journal/source --output ../new_fg2_exports
python3 -B tools/export_fg3_plot_data.py --root . --out ../new_fg3_exports --report ../new_fg3_report.json
```

These programs perform arithmetic display transformations; they do not simulate,
fit curves or estimate intervals. The original FG3 helper can overwrite its output
files. Never point it at this repository or an existing result directory. Neither
exporter was executed in preparing this snapshot.

## Integrity and use boundary

The original numerical programs, configurations, data, methods documents and
plotting exporters are unchanged. This distribution refreshes only the top-level
reading guide, result map and repository checksum inventory. No numerical
experiment, optimizer, curve fit or statistical reconstruction was rerun.


The payload contains no Git history, workflows, remote badges, tracking resources,
source-repository link or credentials. The only optional ignored verification entry
is a real root `.git` directory in a local checkout; its history and identity are not
certified by the payload checker. Other unexpected files, symbolic links and hard
links cause verification failure. Keep the supplied `.gitattributes` so Git does
not change byte-sensitive line endings; it does not install LFS filters.

The PDF methods document contains relative-file links, including links to code and
data. Some readers block these actions. Use the Markdown/HTML versions or open
the corresponding repository files directly when a PDF link does not work.

No redistribution license has been selected or newly granted by this snapshot.
The existing companion's rights statement is retained. This local package is not
an assertion of publication, hosting, journal approval or anonymous-service testing.
