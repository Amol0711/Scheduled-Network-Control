# Current result map

The article and its mathematical supplement are intentionally not stored here.
This map identifies numerical assets for the accompanying 14-page main article
(34 references) and seven-page mathematical supplement. It does not make the repository a dependency for the main's proofs.
All paths below are relative to `simulation/companion_1/` unless stated otherwise.

| Current object | Numerical support and interpretation |
| --- | --- |
| Main Figure 1, page 4 | Conceptual schematic, not a numerical simulation. No raw simulation dataset is asserted for it. |
| Main Figure 2, page 11 | C02–C05; deterministic series, separate displacement study and original phase-tail records. Publication columns are in the root `verified_current/journal/source/data/` directory. |
| Main Figure 3, page 13 | C06 and C09; 31 scalar noise levels and 16 nonlinear network cap entries. Marginal MSE intervals and simultaneous paired-contrast intervals remain separate. |
| Main Table I, page 2 | Literature-positioning table, not the historical numerical Table I. No reproduction script is asserted. |
| Main Table II, page 12 | C02, Table C1 and `data/records/comparison_metrics.csv`. Coupling-only energy at the stated cutoff; total native-plant actuation is a distinct C03 calculation. |
| Main unnumbered three-row certificate summary | C08 and the expanded supplementary certificate arithmetic. The main retains the definitions needed to reconstruct omitted intermediate entries. |
| Supplement Figure S1, page 3 | C08; `data/figure_series/local_certified_trajectory.csv`, local initialization and solver diagnostics. |
| Supplement Figure S2, page 3 | C04, Nine graph-rate cases; `data/targeted_reference/protocols/graph_rate_samples.csv` and original fit records. The initial state is assigned at t = 1.6, not advanced from zero. |
| Supplement Section III, page 7 | C07; nested step comparison and pathwise extrapolation. The first 10,000 of the 20,000 primary paths are reused; no independent-replication or time-discretization-error guarantee is implied. |

## Historical names in the immutable companion

The companion's `docs/RECORD_LOCATIONS.md` is a historical crosswalk. Its earlier
journal equation/table numbers must not be treated as current cross-document links.

| Historical name | Stable companion location | Current interpretation |
| --- | --- | --- |
| Numerical Main Table I and former Table S1 | C02, Table C1 | Main Table II is now the numerical comparison; current main Table I is literature positioning. |
| Former Table S2 | C06, Table C2 | Full 16-cap table; summarized in main Figure 3. |
| Former Table S3 | C07, Table C3 | Nested coarse/fine comparison supporting supplementary Section III. |
| Former Table S4 | C07, Table C4 | Pathwise Richardson diagnostic supporting supplementary Section III. |
| Former mean/SE and paired-interval equations | C06, E07–E08 | Current primary inference is in main Section V; use its semantic definitions rather than old S-numbers. |
| Former extrapolation/covariance equations | C07, E09–E10 | Current supplementary Section III. |

## Frozen study distinctions

C01 specifies the deterministic network and stochastic model; C02 the schedules;
C03 the native-plant actuation and continuous-phase diagnostics; C04 the main,
graph and displacement fitting protocols; C05 the raw phase curve and transported
endpoint checks; C06 the primary cap experiment; C07 the quadratic projection and
nested refinement; C08 the local certificate; C09 the scalar/modal and high-precision
successor records.

The finite-amplitude, local-certified, graph-rate and displacement initializations
are different simulation studies. The 1,061-row publication time series is a display
selection from the original 4,001-point deterministic grid; display selection does
not change a fit population. The 240-row phase curve uses the original raw endpoint
at clock 30; separately transported endpoints are diagnostics, not replacement data.
The scalar 31-row sweep must not be confused with the nonlinear 16-cap study.

The archived p = 3 double-precision diagnostic remains historical. Its later
high-precision record under `data/targeted_reference/power/` corrects only the
stated numerical-precision interpretation. No theorem, stochastic result or
original plotted record was replaced. C1–C4 are complete companion tables; no
current supplementary Table S1–S4 floats are asserted by this map.

## Current labels and unchanged numerical records

The article's wording has been clarified without changing the records below.
The companion's original text, internal equation identifiers and numerical tables
remain intact; use this map for current manuscript terminology.

| Current wording or quantity | Relation to the archived record |
| --- | --- |
| Modal reference cap, approximately 2.540 | In the deterministic comparison this is a fixed cap borrowed from the separate modal-noise study, not an optimum for the finite-amplitude initial state. The stored full-precision cap and the rounded table value are unchanged. |
| 8-scale slope 1.976 | Figure 2's quadratic-displacement fit uses the same eight smallest amplitude scales among twelve displayed scales. There is no new fit or fitting mask. |
| Network slope 3.407 | Figure 2's original finite-window fit of the network phase tail. The analytical exponent 3.4 and the scalar exponent 2.2 remain distinct from fitted values. |
| Radius of the node mean | The norm of the arithmetic node mean, not the mean of individual node radii. The local trajectory's time samples are not Monte Carlo paths. |
| Resolution-limited disagreement | The original common cutoff, 10^-14 display floor and all plotted entries are retained. Above-floor numerical variations have not been flattened or replaced. |
| Marginal and simultaneous intervals | Figure 3's MSE intervals and its paired cap-contrast intervals have different estimands and calibrations. Three nonreference ratios, 0.990, 0.995 and 1.005, are unresolved; the reference contrast at ratio 1 is identically zero. |

The deterministic tube certificates and the separate scalar gain-fed white-noise
model have different hypotheses. The numerical records do not establish stochastic
tube confinement, exact noisy target contact or a general nonlinear stochastic
phase theorem. Current theoretical qualifications belong to the article and its
mathematical supplement, not to an inferred extension of these simulation records.

A later addition of the repository URL to those documents does not require a
change to the numerical payload. Page numbers in this map describe the matched
article supplied with this distribution; object titles and section identities are
the stable navigation aids if pagination changes later.
