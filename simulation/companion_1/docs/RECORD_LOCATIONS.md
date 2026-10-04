# Record locations and numbering

This snapshot adds a computational companion without replacing the journal documents. The journal continues to supply the analytic arguments. The mappings below identify numerical records formerly tabulated in supplementary Section III; they do not assert that those journal copies have been removed.

| Earlier record | Companion location | Current numerical record |
|:--|:--|:--|
| Table S1 | [Table C1](COMPUTATIONAL_COMPANION.html#table-c1) | `data/records/comparison_metrics.csv` and functional-integral records |
| Table S2 | [Table C2](COMPUTATIONAL_COMPANION.html#table-c2) | `data/statistics/stochastic_cap_statistics.csv` |
| Table S3 | [Table C3](COMPUTATIONAL_COMPANION.html#table-c3) | `data/statistics/stochastic_step_statistics.csv` |
| Table S4 | [Table C4](COMPUTATIONAL_COMPANION.html#table-c4) | `data/statistics/stochastic_richardson_statistics.csv` |
| Mean and standard error (S18) | E07, [C06](COMPUTATIONAL_COMPANION.html#c06-estimands) | Primary pathwise NPZ and reporting module |
| Bonferroni contrasts (S19) | E08, [C06](COMPUTATIONAL_COMPANION.html#c06-paired) | Same primary paths, 15 nonzero contrasts |
| Richardson estimator (S20) | E09, [C07](COMPUTATIONAL_COMPANION.html#c07-richardson) | Nested coarse/fine subset |
| Richardson covariance (S21) | E10, [C07](COMPUTATIONAL_COMPANION.html#c07-richardson) | Same-path covariance, not independent samples |
| Local terminal checkpoint (S16) | [C08](COMPUTATIONAL_COMPANION.html#c08-trajectory) | `data/records/local_summary.json` |
| Exact graph fits and displacement direction | [C04](COMPUTATIONAL_COMPANION.html#c04) | Original grid, state ordering, gains and fit masks |
| Raw phase curve and transported endpoints | [C05](COMPUTATIONAL_COMPANION.html#c05) | Original curve versus separately named reference diagnostics |
| Scalar fits and precision comparison | [C09](COMPUTATIONAL_COMPANION.html#c09) | Original optimizer versus high-precision successor |

E01–E06 provide standalone definitions of the computational model, input accounting and sampling grids. Earlier mathematical labels are not imported as unresolved cross-references. The full 16-cap table and all uncertainty definitions remain readable. Original and successor results retain separate paths and provenance.
