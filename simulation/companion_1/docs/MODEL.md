# Implemented stochastic model and its interpretation

Let X stack the six planar nodes, F stack the Stuart–Landau fields, and L be
the unit-weight ring Laplacian. The implemented nonlinear Ito equation is

`dX = [F(X) - k0*kappa_K(t)*(L tensor I2)X] dt
       - k0*sigma*kappa_K(t)*(L tensor e_theta(t)) dW_t`,

with `e_theta(t)=(-sin(omega*t),cos(omega*t))^T` and N independent standard
node Brownian motions before multiplication by L. The tangent is common,
deterministic and defined by the reference cycle, not each node or the mean.
This is node measurement noise, not independent edge noise. Diffusion is
time-dependent and state-independent, so the Ito/Stratonovich correction is zero.
The numerical rule is left-point Euler–Maruyama. The source is
`code/stochastic_network.py::_simulate_terminal_sq_batch`.

The initial radii are one and phases are `2*(a-mean(a))`, for
`a=(.08,-.05,.12,-.10,.03,-.08)`. Modal initial conditions use the centered
phase offsets projected on graph eigenvectors, not Cartesian sine projections.
Phase offsets coincide with arc length on this specified unit cycle. The
linear comparison uses cycle-tangent variational modes normal to the
synchronization manifold; it is not an exact nonlinear stochastic reduction.
The nonlinear terminal cost is `sum_i ||x_i - mean(x)||^2`, not phase variance.
The noise has zero node average because `1^T L=0`, but nonlinear node drift
can transmit disagreement effects into the mean.

The deterministic finite-amplitude, local-certified, graph-rate and displacement
states are separate. In particular the graph-rate solver assigns its state at
t=1.6, not 0. The detailed grids, initial matrices, masks and solver tolerances
are given in NUMERICAL_PROTOCOL.md. No simulator is changed by this documentation.

The 20,000 primary paths share Brownian increments across 16 caps. The first
10,000 supply a nested refinement whose coarse increments sum adjacent fine
increments. Fixed seed 20260829, batch size 200 and ordering are preserved.
The seven-point quadratic fit and Richardson analysis reuse these same paths.
Sampling intervals are for fixed-step quantities; the quadratic vertex interval
is not an unrestricted optimizer confidence interval. Same-stream replay is
not independent replication. Three unresolved nonzero contrasts plus the exact
reference yield four unresolved displayed caps.
