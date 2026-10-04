# Numerical methods, conventions, and successor records

## 1. Pinned original inputs

The unchanged original code and data are available directly under code/ and data/.
config/reference_inputs.json pins their source hashes and the previous archive hash.
The targeted directory reader verifies these inputs; it neither extracts nor executes
their code. The original primary stochastic kernel, model, seed 20260829, cap grid,
batch size, 20,000 primary paths and nested 10,000-path refinement are unchanged.
The `reference` route generates no stochastic trajectories.

The node field is the stated Stuart–Landau field with alpha=b=1 and omega=2. The network horizon is T=2, k0=1.2, N=6. The scalar power-law check below instead uses T=1, and must not be described as the network experiment.

## 2. High-precision p=3 reference

Parameters are p=3, eta0=1, rho=1.2, T=1 and sigma in {1e-5,1e-8,1e-12}. Let theta=(KT)^(1/3). The exact squared bias is

`b(theta)=eta0^2 exp[-rho(3 theta^2-1)]`.

The pre-cap and cap variances are

`v(theta)=rho^2 sigma^2/T integral_1^theta z^4 exp[rho(z^2-3 theta^2)] dz`,

`c(theta)=rho sigma^2/(2T) theta^3 [1-exp(-2rho theta^2)]`.

The exact derivative used by the solver is

`M'(theta)=-6rho theta (b+v) + 3rho sigma^2/(2T) theta^2 (1-e) + 3rho^2 sigma^2/T theta^4 e`,

where `e=exp(-2rho theta^2)`. This formula is separately compared with numerical high-precision differentiation away from the root. Pre-cap variance is evaluated by both quadrature and the independent antiderivative

`exp(rho z^2)[z^3/(2rho)-3z/(4rho^2)] + 3sqrt(pi) erfi(sqrt(rho)z)/(8rho^(5/2))`.

For each noise level, the root is found at 60 and 80 decimal digits. Residuals, explicit nearby derivative-sign brackets, finite objective probes, quadrature/antiderivative agreement, and precision refinement are recorded. Root caps agree beyond 40 significant digits. This is substantially more evidence than a double-precision bounded minimum, but it is not a new proof of global optimality.

The relative errors `abs(K_L/K_ref-1)` are approximately 7.846320869676625e-9, 6.136964529083646e-13 and 2.254136970088907e-18. These replace the numerical precision claim based on the original optimizer. They do not supersede any theorem, cap asymptotic, or existing simulation result. The original CSV remains available with its exact checksum.

## 3. Native-plant full-clock actuation

The normalization is kappa1(t)=1/(T-t), tau=log[T/(T-t)], dt/dtau=T exp(-tau). The native plant is `dot X=F(X)+u`. The full-clock diagnostic therefore requires

`u_c=-k0 kappa1 (L tensor I2)X`,

`u_w=(kappa1-1)F(X)`.

Both `J_total=J_c+J_w+J_cross`, with `J_cross=2 integral u_c dot u_w dt`, and the independent direct integral of `||u_c+u_w||^2` are accumulated. The integrators carry the correct physical-time Jacobian. The regular-clock state equation, phase accumulation and all four energies share the trajectory but have separate accumulator equations.

The cutoff is the same binary64 value tstar=1.99999 as in the archived source. Thus the floating subtraction T-tstar is 1.0000000000065512e-5. The three DOP853 tolerance pairs are (1e-10,1e-12), (1e-12,1e-13), and (2.3e-14,1e-15). The finest run gives J_c=1.2593922836984972, J_w=2399448.0144150066, J_cross=-0.10973109273849226, J_total=2399449.164076198, and phase excess 20.41208384685857 rad. Direct accumulation and component addition agree within floating-point diagnostic tolerance.

A separate cutoff series at nominal deltas 1e-3,1e-4,1e-5,1e-6 tests the trend. On the synchronized unit-cycle diagnostic, ||F(X)||^2=N omega^2=24 and the exact extra energy is

`24[1/delta-1/T-2log(T/delta)+(T-delta)]`.

Its leading coefficient is 24/delta, without an additional T^2. The finite-amplitude numbers are directly integrated and are not assumed equal to the cycle diagnostic. Existing Table I/S1 coupling cells are preserved; the total-energy figures are separately reported.

## 4. Exact deterministic protocols

The finite-amplitude initial phases are (-.75,-.42,-.15,.18,.48,.82) and radii (.75,1.25,.90,1.15,.80,1.20). The time-history and displacement solver is Radau with (rtol,atol)=(1e-10,1e-12). These values are recovered from code rather than guessed from the review.

### Panels (a) and (b)

The state is assigned at t=0. There are 4001 uniform times through tstar. The contraction fit uses log disagreement versus log s, s=(T-t)/T, and the 120 samples with s<.03. The actual plotted time series retain every fourth sample and all of the last 81, for 1061 displayed rows. Display decimation does not change the fit population. The fresh slope is 1.1996984425937631, agreeing with the archived fit under the declared tolerance.

The full-clock phase is accumulated continuously in regular time and resampled on the same 4001 times. Sampled-angle numpy.unwrap gives 14.128898539677198 rad, while continuous phase excess is 20.41208384685857 rad. The final continuous increment is 7.8636414614891805 rad; this particular grid misses one revolution. No universal sample-count threshold is inferred.

### Panel (c)

The three graphs have N=6, unit weights and lambda2 equal to 2-sqrt(3), 1 and 6 for path, ring and complete. The three targets are .6,1,1.4; each k0 is target/lambda2. The 500-point geometric s grid decreases from .2 to 2e-5. The archived solver starts at its first supplied time, T(1-.2)=1.6, with the declared finite-amplitude state. It does not first advance that state from t=0. This late-time initialization is now stated explicitly. Changing it would be a new experiment, so it has not been changed.

The mask is 2e-4<s<.03 and disagreement>1e-12, yielding 272 fitted points for each of nine runs. All nine runs are independently re-integrated using the actual initial-time convention. This panel tests near-deadline rate trends; it is not an equal-initial-time comparison against panel (a).

### Panel (d)

The base is .8(cos(.2),sin(.2)). Center the raw direction with rows (1,.2),(-.7,.5),(.3,-1.1),(-.5,-.4),(.8,.9),(-.9,-.1), then normalize by its Frobenius norm. Node initialization is the base plus epsilon times that direction, hence epsilon=||eta(0)||. The twelve geometric scales range from .015 to .6 and the first eight (ending .15688592433069348) are fitted. The time grid has 1601 uniform points from 0 to tstar. The reduced reference uses DOP853 at (1e-12,1e-13). The fresh slope is 1.9761166020081238. This is not the direction in the local-certified initialization (S14).

## 5. Phase endpoint diagnostics

The original plotted phase curve uses a 240-point geometric s grid from .2 down to 5e-4 and full-state DOP853 in tau with (2e-12,2e-13). The raw mean at tau30 is used as an estimated terminal point. There are 156 fitted samples with .001<=s<=.05 and tail>5e-13. The original recorded slope 3.4072924626261014 and plotted data are preserved. The scalar reference T s^(rho+1)/(rho+1) is evaluated analytically; it is not a separately integrated test.

To compare endpoint estimates at the same physical deadline, transport the numerical mean by the exact reduced flow for the remaining time:

`yhat_T(tau)=phi_(T exp(-tau))(y_num(T(1-exp(-tau))))`.

Compare tau25,30,35. Five runs cover the original full-state tolerance, full-state refinements at (1e-13,1e-14)/maxstep .1 and (2.3e-14,1e-15)/maxstep .05, and corresponding cancellation-free mean/scaled-normal formulations. The scaled normal coordinate is exp(rho tau) V_perp^T eta. Its exact transformed right-hand side is checked against the full-state equation at nonzero disagreements, not merely in the linearized limit.

Raw tau25/30 means differ by about5.45e-11, partly because reduced motion continues. Within each run the transported25/30/35 estimates differ by at most4.01e-16; across the five runs the largest transported-position difference is1.1969568733894625e-13. Transported-tail slopes range from 3.407363944702072 to 3.407767611106687. Thirty deterministic endpoint perturbation probes (plus/minus1e-12 along two axes and a diagonal for each run) record sensitivity; they are not random replications or error bounds.

The backward-forward closure of the reduced flow at one estimated point measures only that composition. Neither that closure nor solver agreement certifies the true terminal-estimation error. The supplement now reports the slope as3.4073 and explicitly states this limitation. The historical figure curves and fit data are not overwritten by these additional runs.

## 6. Existing stochastic records

The seven NPZ arrays are loaded with pickle disabled. The nested fine subset is checked exactly against the corresponding primary rows. No stochastic simulator is imported or executed.

The seven-point quadratic design is reconstructed from the fixed cap ratios. QR coefficients are computed per path; compensated path sums give the coefficient mean. Full coefficient covariance is propagated through the vertex gradient, retaining common-random-number covariance. The vertex is 2.536669300051517, SE .004907710096939817, and 95% interval [2.527049782830235,2.5462888172727984]. Student-t quantiles are checked against the inverse CDF and printed rounding, not an unverified hard-coded long decimal.

All16 cap means, marginal standard errors and intervals, and the 15 paired contrasts and Bonferroni intervals are reconstructed against the sealed records. Exactly three nonzero contrasts (.99,.995,1.005) remain unresolved relative to the reference cap; including the reference gives four cap entries. This result describes a fixed-grid, fixed-step comparison.

A separate modal calculation confirms the phase-offset initialization locked in the journal text and contrasts it with the different Cartesian sine projection; the alternative is diagnostic only and does not replace initial data. The vertex interval remains an interval for the fixed-design quadratic projection, not an unrestricted optimizer interval or a continuous-time error certificate.

## 7. Provenance and supersession

The historical p3 optimizer row is preserved as historical evidence and is superseded only for its stated precision interpretation. New physical-effort records supplement, rather than rewrite, the coupling-only rows. Graph initialization and all masks are disclosed as implemented. Original phase curves are retained with distinct higher-accuracy diagnostics. Statistical numbers are reconstructed from the existing paths. No author identity, manuscript source, or theorem proof is included in this numerical addendum.
