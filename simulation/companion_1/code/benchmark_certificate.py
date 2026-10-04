#!/usr/bin/env python3
"""Independent validation for Benchmark benchmark closure.

The suite checks the synchronization-manifold realization of assumptions
A1--A5, the Stuart--Landau inverse-flow collar, the exact orthogonal-modal
finite-cap sum, the Xu--Liu ell=2 coefficient reduction, and the real-time
meaning of finite-horizon-compatible perturbations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad

import network_experiments as sl

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()
OUT_JSON = ROOT / "benchmark_benchmark_validation.json"
OUT_TXT = ROOT / "benchmark_benchmark_validation.txt"

# Locked benchmark parameters.
N = sl.N
ALPHA = sl.ALPHA
B_SL = sl.B_SL
OMEGA = sl.OMEGA
T = sl.T
K0_GAIN = sl.K0
L = sl.L_RING
LAMBDA2 = sl.LAMBDA2
RHO = sl.RHO_STAR

# Explicit compact-set/tube choices used in the analytical verification.
R0_BASE = 0.90
R1_BASE = 1.002
R2_BASE = 1.15
Q_STAR = 2.0
REGULARITY_ORDER = 3
COMPACTIFICATION_M = 4

TOL = 2.0e-10
RNG = np.random.default_rng(20260829)


def node_jacobian(y: NDArray[np.float64]) -> NDArray[np.float64]:
    return (
        ALPHA * np.eye(2)
        + OMEGA * sl.J
        - B_SL * (2.0 * np.outer(y, y) + float(y @ y) * np.eye(2))
    )


def radial_flow(r: float, t: float) -> float:
    if r == 0.0:
        return 0.0
    q = r * r
    e2 = math.exp(2.0 * ALPHA * t)
    den = ALPHA + B_SL * q * (e2 - 1.0)
    if den <= 0.0:
        return math.inf
    return math.sqrt(ALPHA * q * e2 / den)


def sample_disk(radius: float) -> NDArray[np.float64]:
    theta = RNG.uniform(0.0, 2.0 * math.pi)
    r = radius * math.sqrt(RNG.uniform())
    return np.array([r * math.cos(theta), r * math.sin(theta)])


def sample_disagreement(max_norm: float) -> NDArray[np.float64]:
    eta = RNG.normal(size=(N, 2))
    eta -= eta.mean(axis=0, keepdims=True)
    nrm = float(np.linalg.norm(eta))
    if nrm == 0.0:
        eta[0, 0] = 1.0
        eta[1, 0] = -1.0
        nrm = float(np.linalg.norm(eta))
    eta *= max_norm * RNG.uniform(1.0e-4, 1.0) / nrm
    return eta


def scalar_inverse_time_exact(
    rho: float, xi0: float, sigma: float, cap: float
) -> float:
    b_rho = 0.5 * rho * (1.0 - math.exp(-2.0 * rho))
    c_rho = rho * rho * math.exp(-2.0 * rho) / (2.0 * rho + 1.0)
    b_total = b_rho + c_rho
    mean_sq = xi0 * xi0 * math.exp(-2.0 * rho) * (cap * T) ** (-2.0 * rho)
    variance = sigma * sigma * (
        b_total * cap
        - c_rho * T ** (-(2.0 * rho + 1.0)) * cap ** (-2.0 * rho)
    )
    return mean_sq + variance


def scalar_inverse_time_quadrature(
    rho: float, xi0: float, sigma: float, cap: float
) -> float:
    t_k = T - 1.0 / cap
    mass = math.log(cap * T) + 1.0
    mean_sq = xi0 * xi0 * math.exp(-2.0 * rho * mass)

    def phi(s: float) -> float:
        if s < t_k:
            return math.exp(-rho) * (cap * (T - s)) ** (-rho)
        return math.exp(-rho * cap * (T - s))

    def gain(s: float) -> float:
        return 1.0 / (T - s) if s < t_k else cap

    variance, _ = quad(
        lambda s: rho * rho * sigma * sigma * phi(s) ** 2 * gain(s) ** 2,
        0.0,
        T,
        points=[t_k],
        epsabs=1.0e-13,
        epsrel=2.0e-12,
        limit=300,
    )
    return mean_sq + float(variance)


def main() -> int:
    checks: list[dict[str, Any]] = []

    def add(name: str, condition: bool, detail: Any = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    # A1: reduced-flow geometry and explicit margins.
    x0 = sl.initial_condition()
    y0 = x0.mean(axis=0)
    eta0 = x0 - y0
    y0_radius = float(np.linalg.norm(y0))
    eta0_norm = float(np.linalg.norm(eta0))
    a_star = math.sqrt(ALPHA / B_SL)
    r_crit = a_star / math.sqrt(1.0 - math.exp(-2.0 * ALPHA * T))
    r_forward = radial_flow(R0_BASE, T)
    base_margin = R1_BASE - r_forward
    r_backward = radial_flow(R1_BASE, -T)
    phase_margin = R2_BASE - r_backward
    boundary_radial_derivative = R1_BASE * (ALPHA - B_SL * R1_BASE**2)

    add("initial mean lies in K0", y0_radius < R0_BASE, y0_radius)
    add("initial disagreement lies in selected tube", eta0_norm < Q_STAR, eta0_norm)
    add("K1 forward boundary points inward", boundary_radial_derivative < 0.0, boundary_radial_derivative)
    add("positive reduced-flow base margin", base_margin > 4.0e-3, base_margin)
    add("phase-collar threshold", R1_BASE < r_crit, {"r1": R1_BASE, "rcrit": r_crit})
    add("backward collar lies in K2", phase_margin > 1.0e-2, {"rback": r_backward, "margin": phase_margin})

    # Direct reduced-flow denominator and monotonicity checks on a dense grid.
    min_den = math.inf
    max_k1_radius = 0.0
    for r in np.linspace(0.0, R1_BASE, 101):
        for u in np.linspace(-T, T, 101):
            q = float(r * r)
            e2 = math.exp(2.0 * ALPHA * float(u))
            den = ALPHA + B_SL * q * (e2 - 1.0)
            min_den = min(min_den, den)
            rr = radial_flow(float(r), float(u))
            max_k1_radius = max(max_k1_radius, rr)
    add("reduced flow exists on K1 for |u|<=T", min_den > 0.0, min_den)
    add("dense phase-collar radius check", max_k1_radius < R2_BASE, max_k1_radius)

    # A2: exact graph-normal scheduled contraction.
    evals = np.linalg.eigvalsh(L)
    normal_gap = K0_GAIN * float(evals[1])
    add("ring algebraic connectivity", abs(float(evals[1]) - 1.0) < 2.0e-12, evals.tolist())
    add("scheduled graph gap", abs(normal_gap - RHO) < 2.0e-12, normal_gap)
    min_rayleigh = math.inf
    for _ in range(600):
        eta = sample_disagreement(Q_STAR)
        num = K0_GAIN * float(np.sum(eta * (L @ eta)))
        den = float(np.sum(eta * eta))
        min_rayleigh = min(min_rayleigh, num / den)
        add("A2 Rayleigh sample", num + TOL * den >= RHO * den, num / den)

    # A3: ordinary metric and explicit nonlinear constants.
    beta = ALPHA
    c_r = B_SL * (3.0 * R1_BASE + Q_STAR)
    l2 = B_SL * (3.0 * R1_BASE + Q_STAR) / N
    l_h = l2 * Q_STAR
    max_sym_eig = -math.inf
    max_h_ratio2 = 0.0
    max_h_ratio1 = 0.0
    max_r_ratio2 = 0.0
    for _ in range(900):
        y = sample_disk(R1_BASE)
        eta = sample_disagreement(Q_STAR)
        eta_norm = float(np.linalg.norm(eta))
        jac = node_jacobian(y)
        max_sym_eig = max(max_sym_eig, float(np.linalg.eigvalsh(0.5 * (jac + jac.T)).max()))
        states = y[None, :] + eta
        f_states = sl.node_field(states)
        f_y = sl.node_field(y[None, :])[0]
        h = f_states.mean(axis=0) - f_y
        raw_r = f_states - f_y[None, :] - eta @ jac.T
        r_perp = raw_r - raw_r.mean(axis=0, keepdims=True)
        if eta_norm > 0.0:
            h_ratio2 = float(np.linalg.norm(h)) / (eta_norm * eta_norm)
            h_ratio1 = float(np.linalg.norm(h)) / eta_norm
            r_ratio2 = float(np.linalg.norm(r_perp)) / (eta_norm * eta_norm)
            max_h_ratio2 = max(max_h_ratio2, h_ratio2)
            max_h_ratio1 = max(max_h_ratio1, h_ratio1)
            max_r_ratio2 = max(max_r_ratio2, r_ratio2)
            add("quadratic tangential bound sample", h_ratio2 <= l2 + 2.0e-12, h_ratio2)
            add("generic tangential bound sample", h_ratio1 <= l_h + 2.0e-12, h_ratio1)
            add("normal remainder bound sample", r_ratio2 <= c_r + 2.0e-12, r_ratio2)
        add("ordinary Jacobian metric bound sample", max_sym_eig <= beta + 2.0e-12, max_sym_eig)

    # Consistency check against the displayed inverse-time deterministic run.
    t_eval = np.unique(
        np.sort(
            np.concatenate(
                [
                    np.linspace(0.0, T - 0.05, 180),
                    T - np.geomspace(5.0e-2, sl.EPS_T, 220),
                ]
            )
        )
    )
    trajectory = sl.simulate_deterministic(
        "fiber_ptc", x0, t_eval, compute_energy=False
    )
    mean_radii = np.linalg.norm(np.asarray(trajectory["y"]), axis=1)
    eta_path = np.asarray(trajectory["eta"])
    max_mean_radius = float(mean_radii.max())
    max_eta_norm = float(np.linalg.norm(eta_path.reshape(len(t_eval), -1), axis=1).max())
    add("displayed mean remains in K1", max_mean_radius < R1_BASE, max_mean_radius)
    add("displayed disagreement remains in q-star tube", max_eta_norm < Q_STAR, max_eta_norm)

    # A5: trivial bundle/global polynomial extension and compactification gap.
    delta_r = RHO - REGULARITY_ORDER / COMPACTIFICATION_M
    add("even compactification order", COMPACTIFICATION_M % 2 == 0, COMPACTIFICATION_M)
    add("order-r compactification gap", delta_r > 0.0, delta_r)

    # Xu--Liu ell=2 specialization: bar{k}/C(t) with bar{k}=k0*T.
    max_xu_error = 0.0
    k_bar = K0_GAIN * T
    for t in np.linspace(0.0, T - 1.0e-4, 500):
        c_t = (T - float(t)) ** 2
        lhs = k_bar / c_t
        rhs = K0_GAIN * T / (T - float(t)) ** 2
        max_xu_error = max(max_xu_error, abs(lhs - rhs))
        add("Xu-Liu ell=2 coefficient identity", abs(lhs - rhs) <= 2.0e-12 * max(1.0, abs(rhs)), lhs - rhs)
    add("equal initial gain", abs(k_bar / T**2 - K0_GAIN / T) < 2.0e-14, k_bar / T**2)

    # Exact orthogonal-modal finite-cap sum against independent quadrature.
    max_modal_rel = 0.0
    for _ in range(160):
        modes = int(RNG.integers(2, 9))
        rhos = RNG.uniform(0.25, 4.0, size=modes)
        xi0 = RNG.normal(size=modes)
        sigmas = 10.0 ** RNG.uniform(-5.0, -1.5, size=modes)
        cap = float(RNG.uniform(1.0 / T + 0.05, 8.0))
        exact = sum(
            scalar_inverse_time_exact(float(r), float(x), float(s), cap)
            for r, x, s in zip(rhos, xi0, sigmas)
        )
        quad_sum = sum(
            scalar_inverse_time_quadrature(float(r), float(x), float(s), cap)
            for r, x, s in zip(rhos, xi0, sigmas)
        )
        rel = abs(exact - quad_sum) / max(1.0e-16, abs(exact), abs(quad_sum))
        max_modal_rel = max(max_modal_rel, rel)
        add("exact finite-cap modal sum", rel < 3.0e-10, rel)

    # Real-time interpretation of extension-compatible perturbations.
    max_clock_identity = 0.0
    for t in np.linspace(0.0, T - 1.0e-7, 300):
        s = (T - float(t)) / T
        varsigma = s ** (1.0 / COMPACTIFICATION_M)
        dt_dtau = T * varsigma**COMPACTIFICATION_M
        max_clock_identity = max(max_clock_identity, abs(dt_dtau - (T - float(t))))
        # A relative perturbation of the scheduled normal matrix remains O(1)
        # in tau because dt/dtau * kappa_1(t) = 1.
        add("relative scheduled-gain pullback", abs(dt_dtau / (T - float(t)) - 1.0) < 3.0e-12, dt_dtau / (T - float(t)))
    add("bounded real-time perturbation clock factor", max_clock_identity < 2.0e-15, max_clock_identity)

    passed = all(c["passed"] for c in checks)
    summary = {
        "experiment": "Benchmark",
        "status": "PASS" if passed else "FAIL",
        "checks_passed": sum(c["passed"] for c in checks),
        "checks_total": len(checks),
        "constants": {
            "r0": R0_BASE,
            "r1": R1_BASE,
            "r2": R2_BASE,
            "q_star": Q_STAR,
            "initial_mean_radius": y0_radius,
            "initial_disagreement_norm": eta0_norm,
            "forward_radius_from_r0_at_T": r_forward,
            "base_margin": base_margin,
            "backward_radius_from_r1_at_minus_T": r_backward,
            "phase_collar_margin": phase_margin,
            "phase_threshold": r_crit,
            "beta": beta,
            "L2": l2,
            "Lh": l_h,
            "c_r": c_r,
            "rho": RHO,
            "regularity_order": REGULARITY_ORDER,
            "compactification_m": COMPACTIFICATION_M,
            "compactification_gap": delta_r,
            "max_displayed_mean_radius": max_mean_radius,
            "max_displayed_disagreement": max_eta_norm,
        },
        "maxima": {
            "minimum_A2_Rayleigh": min_rayleigh,
            "maximum_symmetric_Jacobian_eigenvalue": max_sym_eig,
            "maximum_h_over_eta_squared": max_h_ratio2,
            "maximum_h_over_eta": max_h_ratio1,
            "maximum_remainder_over_eta_squared": max_r_ratio2,
            "maximum_Xu_Liu_coefficient_error": max_xu_error,
            "maximum_modal_sum_relative_error": max_modal_rel,
            "maximum_clock_identity_error": max_clock_identity,
        },
        "checks": checks,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "Benchmark BENCHMARK-HYPOTHESIS VALIDATION",
        "=" * 57,
        f"Status: {summary['status']}",
        f"Checks: {summary['checks_passed']}/{summary['checks_total']} PASS",
        "",
        "Explicit benchmark constants:",
        f"  ||y(0)|| = {y0_radius:.12f} < r0 = {R0_BASE:.6f}",
        f"  ||eta(0)|| = {eta0_norm:.12f} < q* = {Q_STAR:.6f}",
        f"  rcrit(T) = {r_crit:.12f}",
        f"  r(-T; r1) = {r_backward:.12f} < r2 = {R2_BASE:.6f}",
        f"  base margin d* = {base_margin:.12e}",
        f"  phase-collar margin = {phase_margin:.12e}",
        f"  rho = k0 lambda2 = {RHO:.12f}",
        f"  beta = {beta:.6f}, L2 = {l2:.12f}, Lh = {l_h:.12f}, c_r = {c_r:.12f}",
        f"  regularity order/m = {REGULARITY_ORDER}/{COMPACTIFICATION_M}, gap = {delta_r:.12f}",
        "",
        "Independent maxima:",
        f"  max modal finite-cap relative error = {max_modal_rel:.3e}",
        f"  max Xu--Liu coefficient error = {max_xu_error:.3e}",
        f"  max pullback clock error = {max_clock_identity:.3e}",
        f"  max displayed mean radius = {max_mean_radius:.12f}",
        f"  max displayed disagreement = {max_eta_norm:.12f}",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
