# Execution scope

Only the two integrity commands were executed while preparing this package.
The other commands are preserved reproduction interfaces, not newly completed runs.
Read `simulation/companion_1/README.txt` for full original instructions.

| Route | What it does | New stochastic paths? | Files written? |
| --- | --- | --- | --- |
| `python3 -B verify_repository.py` | Checks the entire repository manifest, paths and fixed local package boundaries. Standard library only. | No | No |
| `python3 -B simulation/companion_1/reproduce.py verify` | Original nested 129-file companion check. Standard library only. | No | No |
| `quick` | Reconstructs cap statistics, seven-point fit, step/extrapolation/split diagnostics and the numerical local-certificate checks. | No | Yes |
| `statistics` / `tables` | Reconstructs stochastic reporting and rounded display tables from stored terminal arrays. | No | Yes |
| `deterministic` / `figures` | Reintegrates deterministic histories and the certified local trajectory; evaluates analytical series and solver comparisons; reuses specified archived fits/records. | No | Yes |
| `checks` | Nine numerical-check modules, including optimization/quadrature and some fixed random test points. Not formal proof verification. | No stochastic trajectories | Yes |
| `reference` | High-precision roots, physical-effort integration, graph/displacement reintegration, endpoint refinements and covariance-aware reconstruction. | No | Yes |
| `replay` | First 200 primary paths at 16 caps and nested coarse counterparts. Same-stream replay, not independent replication. | Yes | Yes |
| `monte-carlo` | 20,000 primary paths with the nested 10,000-path coarse/fine study. Requires `--confirm-full`; fixed stream and batching. | Yes | Yes |
| `full` | Complete original pipeline and targeted reference stages. Requires `--confirm-full`. | Yes | Yes |
| FG2/FG3 plotting exporters | Decimal differences, ratios, error lengths and display units from the frozen source columns. No fitting or interval estimation. | No | Yes |
| `tools/build_companion.py` inside companion | Rebuilds the companion document into an external directory with Pandoc/pdfLaTeX. Not a numerical experiment. | No | Yes |

For any numerical mode, launch from the repository root using
`python3 -B simulation/companion_1/reproduce.py MODE --output-dir ../new_results`.
Substitute one listed mode, and include the additional confirmation required by
stochastic generation. The literal placeholder `MODE` is not a runnable mode.
Do not run the launcher without a mode: it defaults to `quick`.

Every environment, output, log and cache directory must be outside the entire
repository. The original program prevents writing inside its companion directory;
it cannot recognize the enclosing repository as another protected boundary.
The repository checker rejects unlisted outputs even when `.gitignore` prevents
their accidental addition to Git. Existing result directories are not reusable
interchangeably across snapshot manifests. Never use `--reset` without intending
to discard the selected external numerical checkpoint.

The pinned historical environment is documented in the companion. Other Python,
NumPy, SciPy, Numba or operating-system combinations have not been validated by
this packaging step. Exact bytes, numerical tolerances, same-seed replay and
independent replication are separate notions. Do not rename a failed comparison
as successful reproduction.
