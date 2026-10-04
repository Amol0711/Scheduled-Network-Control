COMPUTATIONAL COMPANION SNAPSHOT 1
=================================

This local, unpublished candidate adds a standalone readable computational
companion to the preserved numerical release. No hosted URL or license is
selected. The numerical models, kernels, records, seeds and estimators are
unchanged. Tables C1-C4 reproduce all four numerical tables. No journal
source or principal theorem proof is distributed here.

START HERE
----------
Open index.html, then docs/COMPUTATIONAL_COMPANION.html (offline MathML)
or docs/COMPUTATIONAL_COMPANION.pdf. The Markdown source is also supplied.
Evidence and former-number maps are in docs/EVIDENCE_INDEX.html and
docs/RECORD_LOCATIONS.html. All links and resources are local.

RELEASE AND REBUILD
-------------------
RELEASE.json identifies the new version and the unchanged upstream snapshot.
SHA256SUMS.txt seals the exact current tree. Do not resume old checkpoints
across a changed package-manifest hash; choose a new output directory.
The document can be rebuilt outside the snapshot with

    python tools/build_companion.py --output ../rebuilt_companion

That optional document build needs Pandoc and pdfLaTeX. Numerical commands
need neither. No command uploads or publishes this candidate.

PRESERVED NUMERICAL INSTRUCTIONS
-------------------------------
The following instructions describe the unchanged numerical routes. Reported
historical checks are not new checks merely because a companion was added.

CONSOLIDATED NUMERICAL RELEASE
=============================

The original scientific code and 45 data assets are unchanged. This release
adds precision/actuation/endpoint reference calculations and explicit model
and protocol metadata. The original p=3 bounded-minimizer record remains
historical; use data/targeted_reference/power/ for its precision correction.
No historical plot or stochastic outcome has been replaced.

NEW TARGETED ROUTE
------------------
    python reproduce.py reference --output-dir "../reference results"

This performs high-precision p=3 derivative-root calculations at 60/80 digits,
full-clock physical-effort integrations including the cross term, nine graph
rate and twelve displacement re-integrations, transported phase-endpoint
refinements, and QR/full-covariance reconstruction from existing pathwise data.
It generates no new stochastic paths. Choose a new external output root for
each reference run; existing reference results are never overwritten.
The full route now also runs these targeted stages after the original pipeline.
Quick still reconstructs archived stochastic reporting without ODE integration.

MODEL AND PROTOCOL AUTHORITY
----------------------------
The actual SDE, initialization and metric are documented in docs/MODEL.md.
All grids, fits, separate initializations, integration conventions and
limitations are in docs/NUMERICAL_PROTOCOL.md and config/experiments.json.
config/supersession.json binds old and new references by hash. The reference
reader pins all original scientific code/data using config/reference_inputs.json.
New targeted reference outputs are under data/targeted_reference/; the
original records and original plotted series remain under data/records/
and data/figure_series/. Historical supplementary validation counts are preserved in
docs/HISTORICAL_VALIDATION.txt, not claimed as newly executed checks.
Some original check modules regenerate the historical
double-precision p=3 diagnostic: that output is not the high-precision successor.
No finite-grid check, numerical refinement, or round trip is a formal proof
or a validated continuous-time error bound.

SCHEDULED NETWORK CONTROL — NUMERICAL CODE AND DATA
=================================================

This archive contains numerical implementations, sealed experiment records,
configuration checks, and reproducibility commands for scheduled control of
six-node Stuart–Landau networks. It also provides scalar schedule/noise
calculations, actuator-realization checks, modal limits, and a local annular
certificate. Outputs are CSV, JSON, TXT and NumPy arrays. No typesetting
system, external repository, account, or network service is used at runtime.

QUICK START
-----------

The standard-library integrity check requires no numerical dependencies.

    python reproduce.py verify

With the numerical dependencies already installed, reconstruct the complete
stochastic reporting and the local-certificate diagnostics without generating
trajectories.

    python reproduce.py quick --output-dir "../simulation results"

The equivalent POSIX launcher is

    sh reproduce_all.sh quick --output-dir "../simulation results"

Run the Python command directly on systems without a POSIX shell. Commands
may be launched from any working directory by supplying the path to
reproduce.py. Relative output paths are interpreted relative to the caller's
working directory. The default output root is a sibling directory named
simulation_results. Outputs cannot be written inside, or over an ancestor
of, the code/data package.

ENVIRONMENT
-----------

The tested interpreter is CPython 3.13.5 on Linux x86_64. requirements.txt pins
NumPy 2.3.5, SciPy 1.17.0, SymPy 1.14.0, mpmath 1.3.0, Numba 0.65.1 and
llvmlite 0.47.0. These are reproducibility pins, not a claim that they are the
latest versions. requirements-quick.txt contains only the dependencies of
the quick/statistics routes. The verify route needs the standard library only.

Create the environment outside the archive, for example

    python3.13 -m venv ../simulation_env
    ../simulation_env/bin/python -m pip install -r requirements.txt
    ../simulation_env/bin/python reproduce.py quick --output-dir ../results

A dependency installation can require internet access. The archive does not
bundle an interpreter or platform-specific dependency wheels. With a prepared
wheel directory, installation can instead be performed offline.

    ../simulation_env/bin/python -m pip install --no-index \
        --find-links /path/to/wheels -r requirements.txt

The numerical commands do not install packages or make network requests.
Use the environment's Python executable for both installation and execution.
The POSIX wrapper honors the PYTHON environment variable. For example

    PYTHON="../simulation_env/bin/python" sh reproduce_all.sh verify

Cross-platform bitwise equality is not assumed. Each numerical comparison
records both exact equality and its explicit tolerance. The tested versions
and actual platform are written to each run_summary.json. Other platforms
have not been execution-tested in the supplied verification record.

COMMANDS AND THEIR SCOPE
-----------------------

reference
    Executes all five fresh targeted stages described above. CSV tolerances and
    byte equality of every JSON/CSV reference are reported. All internal
    high-precision and diagnostic checks must pass. The old primary stochastic
    archive is read, not regenerated. On other platforms JSON byte inequality
    is reported for review, not silently rounded away.

    python reproduce.py reference --output-dir "../new reference results"


verify
    Verifies every delivered file against SHA256SUMS.txt. Missing, modified,
    unexpected, symlinked or hard-linked files cause failure. Nothing is written.

quick
    Reconstructs all 16 cap rows, marginal and paired uncertainties, the
    fixed seven-point quadratic fit, the three step-refinement rows, the
    covariance-preserving Richardson statistics and sample-split diagnostics.
    Checks the annular certificate at 80 digits and the supplied trajectory
    records. No trajectory integration or random path generation is performed.

statistics (alias tables)
    Runs the statistical portion of quick. Full-precision CSVs and separate
    rounded display CSVs are generated under statistics/. Display scales and
    rounding are stated in each column name. There is no typesetting output.

    python reproduce.py statistics --output-dir ../results

deterministic (alias figures)
    Regenerates the numerical figure series and the locally certified
    trajectory. The main deterministic time histories are reintegrated.
    Analytic curves are reevaluated, while the supplied graph-exponent,
    tangential-scaling, phase-fit and stochastic records are reused. The local
    certificate command recomputes its five specified solver comparisons.
    All 12 reference figure-series CSVs are compared numerically. No
    stochastic trajectories are generated by this route.

    python reproduce.py deterministic --output-dir ../results

checks
    Runs nine numerical check modules for schedule profiles, the Dirichlet
    envelope, the functional hierarchy, actuator realization, the benchmark
    certificate, schedule asymptotics, modal limits, realization gaps and
    common-snapshot comparisons. Some diagnostics use fixed random test
    points; none simulates stochastic trajectories. Finite-grid numerical
    checks are not machine-verified proofs.

    python reproduce.py checks --output-dir ../results

replay
    Replays the first 200 primary paths at all 16 caps and the corresponding
    200 nested coarse paths at three caps. It tests the recorded random stream,
    not an independent replication. Exact equality and tolerance agreement
    are recorded separately.

    python reproduce.py replay --output-dir ../results --threads 4

monte-carlo
    Regenerates the 20,000-path common-random-number cap experiment, including
    the nested 10,000-path coarse/fine refinement. This route requires explicit
    confirmation. All model constants, time steps, caps, seeds, estimators,
    batch size and path counts are fixed. It does not rerun the separate
    exploratory grid or scalar schedule studies.

    python reproduce.py monte-carlo --confirm-full --output-dir ../results

    Partial chunks preserve the same 200-path batching and NumPy stream.
    Repeat this command without --reset until the cursor reaches 20000.

    python reproduce.py monte-carlo --confirm-full --chunk-paths 400 \
        --output-dir ../results

    A partial command reports PARTIAL, not completed-experiment PASS. The
    checkpoint records the next stream position. Chunk size must be a positive
    multiple of 200. The package and environment guard prevents resuming an
    incompatible checkpoint. --reset explicitly restarts this route's state;
    it does not modify the supplied reference data. A completed run compares
    every generated terminal array and the reconstructed statistical records
    with the supplied references.

full
    Runs the complete portable numerical pipeline, starting with an empty
    numerical work directory on a fresh run. It regenerates the deterministic
    network comparisons, the 13-cap 800-path-per-cap exploratory nonlinear
    study, four 20,000-path scalar checks, the nine numerical check modules,
    the primary 20,000-path common-random-number experiment, statistical
    reporting, figure series, local-certificate calculations and the new targeted
    reference stages. The supplied
    output records are not staged into the full route's work directory.
    All 23 reference numerical CSVs, all 12 figure CSVs and the four full-
    precision statistical CSVs are compared with the regenerated products.

    python reproduce.py full --confirm-full --output-dir ../full_results

    This route is the complete numerical pipeline distributed here, not a
    claim to execute private document-build or historical-version lock tests.
    The expensive routes are never triggered by quick, statistics or verify.

INPUT DATA AND FIXED EXPERIMENT DESIGN
-------------------------------------

The experiment specification is config/experiments.json. It describes the
fixed constants in the numerical modules; validate_configuration.py checks
agreement rather than silently overriding the implementation. The code files
under code/ are internal modules. Invoke the root command, not a module file,
so that inputs, output paths, caches and error handling are set consistently.

The key path archive is

    data/records/stochastic_crn_pathwise_terminal_mse.npz

It contains caps[16], Kopt[1], terminal_squared_disagreement[20000,16],
refinement_caps[3], refinement_dt_1e3[10000,3],
refinement_dt_5e4[10000,3], and seed[1]. Arrays have ordinary numerical dtypes;
pickled objects are not used. The seed is 20260829 and the generation batch
size is 200. Changing the batch size or array ordering changes the stream.

The primary step is 0.0005 and the coarse step is 0.001. The fine refinement
array is exactly the first 10000 primary rows at its three caps. Adjacent fine
Brownian increments are summed for the coarse calculation. All caps use
common increments. There are 20000 distinct independent primary paths, not
30000; the refinement does not introduce another independent sample.

The tangent-noise network uses T=2, coupling multiplier 1.2 and sigma=0.002.
Its initial radii are one and its phases are twice the centered array
[0.08,-0.05,0.12,-0.10,0.03,-0.08]. This is not the finite-amplitude
deterministic initialization or the locally certified initialization.

The locally certified state preserves the finite-amplitude state's mean and
rescales its disagreement to 0.01. The working annulus, tube radius, exact
scalar constants, selected DOP853 settings and terminal gap are supplied.
The trajectory stops at T-1e-5. Sampled inequalities and agreement between
solvers do not establish continuous-time invariance or exact terminal contact.

INTERPRETATION OF STOCHASTIC REPORTING
-------------------------------------

Sample standard deviations use ddof=1. Marginal intervals use the 0.975
Student-t quantile. The 15 nonzero reference contrasts use the
1-0.05/(2*15) quantile and pathwise paired differences. Student-t calibration
is approximate for the non-Gaussian squared-disagreement observations. These
are sampling intervals for fixed-step expectations, not continuous-time
accuracy bounds.

Ratios 0.990, 0.995, 1.000 and 1.005 are unresolved against the analytical
reference on the specified tested grid. This set is not a confidence interval
for an unrestricted nonlinear optimizer. The seven-point least-squares fit
estimates the vertex of a fixed-design quadratic projection. Its delta-method
interval excludes quadratic-model error and time-discretization bias.

Richardson statistics evaluate 2*fine-coarse pathwise on the same 10000
paths. Their uncertainty includes cross-step covariance. Two step levels do
not establish a weak-order expansion or bound the remaining discretization
bias. Small point discrepancies do not certify 0.04 percent accuracy; the
reported marginal interval half-widths exceed one percent of the modal
prediction. The three refinement diagnostics reuse paths and are not
independent replications.

DIRECTORY CONTENTS AND OUTPUT SAFETY
-----------------------------------

code/                 Numerical implementations and comparison utilities.
config/               Fixed experiment specification, checked against code.
data/records/         Sealed numerical inputs and reference experiment records.
data/statistics/      Full-precision reference reporting products.
data/figure_series/   Reference numerical series, without rendering code.
requirements*.txt     Tested dependency pins.
SHA256SUMS.txt         SHA-256 digest for every other delivered file.

Each route writes only beneath its own external output subdirectory. Input
records are staged into work/ for reconstruction routes and are never changed
inside the package. The full route generates its work products itself. Logs,
Numba caches, checkpoint files, comparison results and run_summary.json are
kept under the selected output root. Different routes have separate work and
checkpoint directories. Existing symbolic or hard links in an output tree
are rejected to avoid redirecting writes into reference or unrelated files. An exclusive run.lock prevents simultaneous writers
to the same route. After an interrupted process, confirm that no process is
still active before removing a stale run.lock.

The manifest is checked before and after execution. A missing dependency,
incompatible checkpoint, numerical-check failure or reference-comparison
failure produces a nonzero exit status. No failed test is silently converted
into an accepted result. A partial stochastic checkpoint is deliberately not
reported as a completed experiment.

Byte equality and numerical agreement are separate comparisons. The numerical
CSV comparisons use declared relative and absolute tolerances; the pathwise
comparison records exact equality for every array in addition to its tolerance.
A small terminal deterministic difference below the declared absolute
comparison threshold is not a solver-error bound. Full-precision reference
files always remain unchanged.

Scientific data are supplied for examination and reproducibility. No public
redistribution license or release authorization is asserted by this archive.
