# Evidence-to-code/data index

All paths are local to this snapshot. Numerical records preserve their original filenames. The archive contains no journal source or principal theorem proofs. `quick` reads the primary arrays; `reference` produces targeted deterministic/calculation checks; `full` regenerates the complete numerical study with explicit confirmation. A route's successful execution is not a mathematical proof.

| Journal evidence | Companion section | Data and execution |
|:--|:--|:--|
| Figure 2(a,b), time histories | [C04](COMPUTATIONAL_COMPANION.html#c04-main) and [C03](COMPUTATIONAL_COMPANION.html#c03) | `data/figure_series/deterministic_timeseries.csv`; `code/figure_series.py`, `reference` |
| Figure 2(c), graph rates | [C04](COMPUTATIONAL_COMPANION.html#c04-graphs) | `data/records/exponent_validation.csv`; `targeted/protocol_reference.py` |
| Figure 2(d), displacement | [C04](COMPUTATIONAL_COMPANION.html#c04-displacement) | `data/records/tangential_scaling.csv`; `targeted/protocol_reference.py` |
| Figure 2(e), scalar accuracy | [C09](COMPUTATIONAL_COMPANION.html#c09-scalar) | `data/figure_series/optimized_noise_scaling.csv`; `code/schedule_noise.py`, `code/dirichlet_envelope.py` |
| Figure 2(f), phase tails | [C05](COMPUTATIONAL_COMPANION.html#c05) | `data/records/phase_map_validation.csv`; `code/network_experiments.py`, `targeted/phase_reference.py` |
| Local numerical trajectory (Figure S1) | [C08](COMPUTATIONAL_COMPANION.html#c08) | `data/figure_series/local_certified_trajectory.csv`; `code/local_certificate.py` |
| Main Table I / former Table S1 | [Table C1](COMPUTATIONAL_COMPANION.html#table-c1) | `data/records/comparison_metrics.csv`; `code/comparison_metrics.py` |
| Former Table S2 | [Table C2](COMPUTATIONAL_COMPANION.html#table-c2) | `data/statistics/stochastic_cap_statistics.csv`; `quick` |
| Former Table S3 | [Table C3](COMPUTATIONAL_COMPANION.html#table-c3) | `data/statistics/stochastic_step_statistics.csv`; `quick` |
| Former Table S4 | [Table C4](COMPUTATIONAL_COMPANION.html#table-c4) | `data/statistics/stochastic_richardson_statistics.csv`; `quick` |

Figure 1 is a geometric schematic, not a numerical record. Its artwork and theoretical meaning remain in the journal. The local-certificate figure's complete analytic construction also remains journal material.

## Section asset directory

### C01 — Model, metric and experiment families

[Companion location](COMPUTATIONAL_COMPANION.html#c01)

- [`docs/MODEL.md`](../docs/MODEL.md)
- [`docs/NUMERICAL_PROTOCOL.md`](../docs/NUMERICAL_PROTOCOL.md)
- [`config/experiments.json`](../config/experiments.json)
- [`code/network_experiments.py`](../code/network_experiments.py)
- [`code/stochastic_network.py`](../code/stochastic_network.py)

### C02 — Deterministic schedules and complete comparison

[Companion location](COMPUTATIONAL_COMPANION.html#c02)

- [`data/records/comparison_metrics.csv`](../data/records/comparison_metrics.csv)
- [`data/records/method_metrics.csv`](../data/records/method_metrics.csv)
- [`data/records/functional_hierarchy_values.csv`](../data/records/functional_hierarchy_values.csv)
- [`data/figure_series/comparison_metrics.csv`](../data/figure_series/comparison_metrics.csv)

### C03 — Full-clock energy and phase diagnostics

[Companion location](COMPUTATIONAL_COMPANION.html#c03)

- [`data/targeted_reference/effort/full_clock_snapshot.csv`](../data/targeted_reference/effort/full_clock_snapshot.csv)
- [`data/targeted_reference/effort/full_clock_cutoff_series.csv`](../data/targeted_reference/effort/full_clock_cutoff_series.csv)
- [`data/targeted_reference/protocols/full_clock_phase_grid.csv`](../data/targeted_reference/protocols/full_clock_phase_grid.csv)
- [`targeted/effort_reference.py`](../targeted/effort_reference.py)

### C04 — Contraction, graph and displacement protocols

[Companion location](COMPUTATIONAL_COMPANION.html#c04)

- [`data/records/exponent_validation.csv`](../data/records/exponent_validation.csv)
- [`data/records/tangential_scaling.csv`](../data/records/tangential_scaling.csv)
- [`data/targeted_reference/protocols/graph_protocol_and_reintegration.csv`](../data/targeted_reference/protocols/graph_protocol_and_reintegration.csv)
- [`data/targeted_reference/protocols/displacement_protocol_and_reintegration.csv`](../data/targeted_reference/protocols/displacement_protocol_and_reintegration.csv)
- [`targeted/protocol_reference.py`](../targeted/protocol_reference.py)

### C05 — Terminal-phase curve and endpoint diagnostics

[Companion location](COMPUTATIONAL_COMPANION.html#c05)

- [`data/records/phase_map_validation.csv`](../data/records/phase_map_validation.csv)
- [`data/figure_series/phase_map_validation.csv`](../data/figure_series/phase_map_validation.csv)
- [`data/targeted_reference/phase/phase_reference.json`](../data/targeted_reference/phase/phase_reference.json)
- [`targeted/phase_reference.py`](../targeted/phase_reference.py)

### C06 — Complete primary stochastic cap evidence

[Companion location](COMPUTATIONAL_COMPANION.html#c06)

- [`data/records/stochastic_crn_pathwise_terminal_mse.npz`](../data/records/stochastic_crn_pathwise_terminal_mse.npz)
- [`data/statistics/stochastic_cap_statistics.csv`](../data/statistics/stochastic_cap_statistics.csv)
- [`data/records/stochastic_summary.json`](../data/records/stochastic_summary.json)
- [`code/table_reporting.py`](../code/table_reporting.py)

### C07 — Covariance-aware vertex and nested refinement

[Companion location](COMPUTATIONAL_COMPANION.html#c07)

- [`data/statistics/stochastic_quadratic_fit.json`](../data/statistics/stochastic_quadratic_fit.json)
- [`data/statistics/stochastic_step_statistics.csv`](../data/statistics/stochastic_step_statistics.csv)
- [`data/statistics/stochastic_richardson_statistics.csv`](../data/statistics/stochastic_richardson_statistics.csv)
- [`data/statistics/stochastic_sample_split.csv`](../data/statistics/stochastic_sample_split.csv)
- [`targeted/statistics_reference.py`](../targeted/statistics_reference.py)

### C08 — Numerical evaluation of analytic certificates

[Companion location](COMPUTATIONAL_COMPANION.html#c08)

- [`data/records/local_certificate.json`](../data/records/local_certificate.json)
- [`data/records/local_initial_states.csv`](../data/records/local_initial_states.csv)
- [`data/records/local_solver_comparison.csv`](../data/records/local_solver_comparison.csv)
- [`data/records/local_trajectory.csv`](../data/records/local_trajectory.csv)
- [`data/figure_series/local_certified_trajectory.csv`](../data/figure_series/local_certified_trajectory.csv)
- [`code/benchmark_certificate.py`](../code/benchmark_certificate.py)

### C09 — Scalar and modal diagnostic records and supersessions

[Companion location](COMPUTATIONAL_COMPANION.html#c09)

- [`data/records/modal_lambert_remainder_and_slopes.csv`](../data/records/modal_lambert_remainder_and_slopes.csv)
- [`data/records/asymptotic_power_p3_accuracy.csv`](../data/records/asymptotic_power_p3_accuracy.csv)
- [`data/targeted_reference/power/historical_power_p3_accuracy.csv`](../data/targeted_reference/power/historical_power_p3_accuracy.csv)
- [`data/targeted_reference/power/power_p3_successor.csv`](../data/targeted_reference/power/power_p3_successor.csv)
- [`config/supersession.json`](../config/supersession.json)
- [`targeted/power_reference.py`](../targeted/power_reference.py)

