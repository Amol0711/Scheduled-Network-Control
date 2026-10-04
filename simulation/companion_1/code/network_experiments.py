#!/usr/bin/env python3
"""Deterministic and stochastic six-node Stuart--Landau experiments.

Checks schedule comparisons, continuous phase accumulation, phase projection,
modal cap predictions, and nonlinear stochastic terminal disagreement.
Writes numerical CSV, JSON and text records only."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize_scalar
from scipy.special import expn

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()

# Locked application parameters.
N = 6
Q = 2
ALPHA = 1.0
B_SL = 1.0
OMEGA = 2.0
T = 2.0
K0 = 1.2
EPS_T = 1.0e-5
RESOLUTION_FLOOR = 1.0e-14

J = np.array([[0.0, -1.0], [1.0, 0.0]])


def graph_laplacian(kind: str, n: int = N) -> NDArray[np.float64]:
    a = np.zeros((n, n), dtype=float)
    if kind == "path":
        for i in range(n - 1):
            a[i, i + 1] = a[i + 1, i] = 1.0
    elif kind == "ring":
        for i in range(n):
            a[i, (i - 1) % n] = 1.0
            a[i, (i + 1) % n] = 1.0
    elif kind == "complete":
        a = np.ones((n, n), dtype=float) - np.eye(n)
    else:
        raise ValueError(f"Unknown graph kind: {kind}")
    return np.diag(a.sum(axis=1)) - a


L_RING = graph_laplacian("ring")
LAM_RING, V_RING = np.linalg.eigh(L_RING)
LAMBDA2 = float(LAM_RING[1])
RHO_STAR = K0 * LAMBDA2


def node_field(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stuart--Landau field in real coordinates, vectorized over leading axes."""
    jx = x @ J.T
    return ALPHA * x + OMEGA * jx - B_SL * np.sum(x * x, axis=-1)[..., None] * x


def schedule(t: float, method: str, cap: float | None = None) -> float:
    if method == "constant":
        return 1.0 / T
    if method == "fiber_ptc":
        return 1.0 / (T - t)
    if method == "xu_liu_l2":
        # Equal-initial-gain inverse-square profile.  It is also the
        # single-weight undirected specialization of Xu--Liu's regulatory
        # function C(t)=(T-t)^2.
        return T / (T - t) ** 2
    if method == "capped":
        if cap is None:
            raise ValueError("cap is required")
        if t >= T - 1.0 / cap:
            return cap
        return 1.0 / (T - t)
    if method == "full_clock":
        return 1.0 / (T - t)
    raise ValueError(method)


def deterministic_rhs(
    t: float,
    xflat: NDArray[np.float64],
    lap: NDArray[np.float64],
    k0: float,
    method: str,
    cap: float | None = None,
) -> NDArray[np.float64]:
    x = xflat.reshape(N, Q)
    kappa = schedule(t, method, cap)
    coupling = lap @ x
    if method == "full_clock":
        dx = kappa * (node_field(x) - k0 * coupling)
    else:
        dx = node_field(x) - kappa * k0 * coupling
    return dx.ravel()


def initial_condition() -> NDArray[np.float64]:
    phases = np.array([-0.75, -0.42, -0.15, 0.18, 0.48, 0.82])
    radii = np.array([0.75, 1.25, 0.90, 1.15, 0.80, 1.20])
    return np.column_stack((radii * np.cos(phases), radii * np.sin(phases)))


def solve_reduced(y0: NDArray[np.float64], t_eval: NDArray[np.float64]) -> NDArray[np.float64]:
    sol = solve_ivp(
        lambda _t, y: node_field(y[None, :])[0],
        (float(t_eval[0]), float(t_eval[-1])),
        y0,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def stuart_landau_flow(
    y0: NDArray[np.float64], delta_t: NDArray[np.float64] | float
) -> NDArray[np.float64]:
    """Closed-form reduced flow, including negative times in its flow collar."""
    dt = np.asarray(delta_t, dtype=float)
    q0 = float(y0 @ y0)
    if q0 == 0.0:
        return np.zeros(dt.shape + (2,), dtype=float)
    e2 = np.exp(2.0 * ALPHA * dt)
    den = ALPHA + B_SL * q0 * (e2 - 1.0)
    if np.any(den <= 0.0):
        raise ValueError("Requested backward reduced flow leaves the selected collar")
    q = ALPHA * q0 * e2 / den
    radius = np.sqrt(q)
    theta = math.atan2(float(y0[1]), float(y0[0])) + OMEGA * dt
    return np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=-1)


def _phase_rate(y: NDArray[np.float64], dy: NDArray[np.float64]) -> float:
    den = float(y @ y)
    if den <= 1.0e-24:
        raise RuntimeError("Mean state approached the phase singularity")
    return float((y[0] * dy[1] - y[1] * dy[0]) / den)


def _control_energy_from_dense_solution(
    sol: Any,
    method: str,
    lap: NDArray[np.float64],
    k0: float,
    t_end: float,
    cap: float | None,
) -> float:
    """Adaptive physical-time quadrature of the coupling-energy integral."""

    def integrand(t: float) -> float:
        x = np.asarray(sol.sol(t)).reshape(N, Q)
        u = -k0 * schedule(t, method, cap) * (lap @ x)
        return float(np.sum(u * u))

    # The geometric terminal partition resolves every singular schedule while
    # retaining a deterministic, high-accuracy quadrature path.
    fractions = [0.0, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995,
                 0.999, 0.9995, 0.9999]
    points = sorted(set([T * f for f in fractions] + [t_end]))
    points = [p for p in points if 0.0 <= p <= t_end]
    if points[0] != 0.0:
        points.insert(0, 0.0)
    if points[-1] != t_end:
        points.append(t_end)
    value = 0.0
    for a, b in zip(points[:-1], points[1:]):
        part, _ = quad(integrand, a, b, epsabs=1e-11, epsrel=1e-10, limit=250)
        value += part
    return float(value)


def simulate_full_clock_tau(
    x0: NDArray[np.float64],
    t_eval: NDArray[np.float64],
    lap: NDArray[np.float64] = L_RING,
    k0: float = K0,
) -> Dict[str, NDArray[np.float64] | float]:
    """Regular full-clock integration in tau=log[T/(T-t)].

    A continuous mean-phase accumulator removes the 2*pi alias caused by
    unwrapping a terminal physical-time grid whose final phase increment can
    exceed pi.  The energy accumulator integrates the physical-time coupling
    effort exactly under the same transformation.
    """
    tau_eval = np.log(T / (T - t_eval))
    y0 = x0.mean(axis=0)
    z0 = np.concatenate(
        (x0.ravel(), np.array([math.atan2(float(y0[1]), float(y0[0])), 0.0]))
    )

    def rhs_tau(tau: float, z: NDArray[np.float64]) -> NDArray[np.float64]:
        x = z[: N * Q].reshape(N, Q)
        coupling = lap @ x
        dx = node_field(x) - k0 * coupling
        y = x.mean(axis=0)
        dy = dx.mean(axis=0)
        phase_dot = _phase_rate(y, dy)
        physical_gain = math.exp(tau) / T
        u_sq = (k0 * physical_gain) ** 2 * float(np.sum(coupling * coupling))
        dt_dtau = T * math.exp(-tau)
        energy_dot = u_sq * dt_dtau
        return np.concatenate((dx.ravel(), np.array([phase_dot, energy_dot])))

    sol = solve_ivp(
        rhs_tau,
        (float(tau_eval[0]), float(tau_eval[-1])),
        z0,
        t_eval=tau_eval,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )
    if not sol.success:
        raise RuntimeError(f"full_clock_tau: {sol.message}")

    x = sol.y[: N * Q].T.reshape(-1, N, Q)
    phase = sol.y[N * Q]
    energy_path = sol.y[N * Q + 1]
    y = x.mean(axis=1)
    eta = x - y[:, None, :]
    disagreement = np.linalg.norm(eta.reshape(len(t_eval), -1), axis=1)
    u_norm = np.empty(len(t_eval))
    for k, (tau, xx) in enumerate(zip(tau_eval, x)):
        u_norm[k] = k0 * math.exp(float(tau)) / T * np.linalg.norm(lap @ xx)
    legacy_phase = np.unwrap(np.arctan2(y[:, 1], y[:, 0]))
    return {
        "x": x,
        "y": y,
        "eta": eta,
        "disagreement": disagreement,
        "u_norm": u_norm,
        "energy": float(energy_path[-1]),
        "phase": phase,
        "legacy_phase": legacy_phase,
        "achieved_final_time": float(t_eval[-1]),
        "achieved_final_tau": float(tau_eval[-1]),
    }


def simulate_deterministic(
    method: str,
    x0: NDArray[np.float64],
    t_eval: NDArray[np.float64],
    lap: NDArray[np.float64] = L_RING,
    k0: float = K0,
    cap: float | None = None,
    compute_energy: bool = True,
) -> Dict[str, NDArray[np.float64] | float]:
    if method == "full_clock":
        return simulate_full_clock_tau(x0, t_eval, lap=lap, k0=k0)

    sol = solve_ivp(
        lambda t, x: deterministic_rhs(t, x, lap, k0, method, cap),
        (float(t_eval[0]), float(t_eval[-1])),
        x0.ravel(),
        t_eval=t_eval,
        method="Radau",
        rtol=1e-10,
        atol=1e-12,
        dense_output=compute_energy,
    )
    if not sol.success:
        raise RuntimeError(f"{method}: {sol.message}")
    x = sol.y.T.reshape(-1, N, Q)
    y = x.mean(axis=1)
    eta = x - y[:, None, :]
    disagreement = np.linalg.norm(eta.reshape(len(t_eval), -1), axis=1)
    phase = np.unwrap(np.arctan2(y[:, 1], y[:, 0]))
    u_norm = np.zeros(len(t_eval))
    for k, (t, xx) in enumerate(zip(t_eval, x)):
        gain = schedule(float(t), method, cap)
        u = -k0 * gain * (lap @ xx)
        u_norm[k] = np.linalg.norm(u)
    energy = (
        _control_energy_from_dense_solution(sol, method, lap, k0, float(t_eval[-1]), cap)
        if compute_energy
        else float("nan")
    )
    return {
        "x": x,
        "y": y,
        "eta": eta,
        "disagreement": disagreement,
        "u_norm": u_norm,
        "energy": energy,
        "phase": phase,
        "legacy_phase": phase,
        "achieved_final_time": float(t_eval[-1]),
    }


def normalized_offset_functional(method: str, cap: float | None = None) -> float:
    """Return I_kappa/T for the fiber-only profile used by each method."""
    rho = RHO_STAR
    if method == "constant":
        return (1.0 - math.exp(-rho)) / rho
    if method == "fiber_ptc":
        return 1.0 / (rho + 1.0)
    if method == "xu_liu_l2":
        return math.exp(rho) * float(expn(2, rho))
    if method == "capped":
        if cap is None:
            raise ValueError("cap is required")
        a = cap * T
        return (
            (1.0 - a ** (-(rho + 1.0))) / (rho + 1.0)
            + a ** (-(rho + 1.0)) * (1.0 - math.exp(-rho)) / rho
        )
    if method == "full_clock":
        return float("nan")
    raise ValueError(method)


def modal_terminal_components(
    cap: float,
    sigma: float,
    lam: NDArray[np.float64],
    modal_initial: NDArray[np.float64],
) -> Tuple[float, float, float]:
    """Return exact deterministic bias, stochastic variance, and total MSE."""
    bias = 0.0
    variance = 0.0
    total = 0.0
    for j in range(1, len(lam)):
        rho = K0 * float(lam[j])
        xi0 = float(modal_initial[j])
        b_rho = 0.5 * rho * (1.0 - math.exp(-2.0 * rho))
        c_rho = rho * rho * math.exp(-2.0 * rho) / (2.0 * rho + 1.0)
        b_big = b_rho + c_rho
        a_j = abs(xi0) * math.exp(-rho) * T ** (-rho)
        bias_j = a_j * a_j * cap ** (-2.0 * rho)
        variance_j = sigma * sigma * (
            b_big * cap
            - c_rho * T ** (-(2.0 * rho + 1.0)) * cap ** (-2.0 * rho)
        )
        bias += bias_j
        variance += variance_j
        total += bias_j
        total += variance_j
    return bias, variance, total


def modal_terminal_mse(
    cap: float,
    sigma: float,
    lam: NDArray[np.float64],
    modal_initial: NDArray[np.float64],
) -> float:
    return modal_terminal_components(cap, sigma, lam, modal_initial)[2]


def nonlinear_noise_mc(
    cap: float,
    sigma: float,
    phase_offsets: NDArray[np.float64],
    paths: int,
    dt: float,
    seed: int,
) -> Tuple[float, float, float, float]:
    """Full nonlinear network under tangent-channel measurement noise."""
    rng = np.random.default_rng(seed)
    steps = int(round(T / dt))
    dt = T / steps
    x0 = np.column_stack((np.cos(phase_offsets), np.sin(phase_offsets)))
    x = np.broadcast_to(x0, (paths, N, Q)).copy()

    for n in range(steps):
        t = n * dt
        gain = cap if t >= T - 1.0 / cap else 1.0 / (T - t)
        coupling = np.einsum("ij,pjk->pik", L_RING, x)
        drift = node_field(x) - gain * K0 * coupling
        dw = rng.normal(size=(paths, N)) * math.sqrt(dt)
        ldw = dw @ L_RING.T
        theta = OMEGA * t
        tangent = np.array([-math.sin(theta), math.cos(theta)])
        x += drift * dt - gain * K0 * sigma * ldw[:, :, None] * tangent[None, None, :]

    y = x.mean(axis=1)
    eta = x - y[:, None, :]
    sq = np.sum(eta * eta, axis=(1, 2))
    mean = float(np.mean(sq))
    se = float(np.std(sq, ddof=1) / math.sqrt(paths))
    return mean, se, mean - 1.96 * se, mean + 1.96 * se


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def phase_map_validation(x0: NDArray[np.float64]) -> dict[str, Any]:
    """Directly validate the generic and Stuart--Landau phase-tail exponents."""
    s_values = np.geomspace(0.2, 5.0e-4, 240)
    tau_values = -np.log(s_values)
    tau_terminal_1 = 25.0
    tau_terminal_2 = 30.0
    tau_eval = np.unique(
        np.concatenate(([0.0], tau_values, [tau_terminal_1, tau_terminal_2]))
    )

    def rhs_tau(tau: float, xflat: NDArray[np.float64]) -> NDArray[np.float64]:
        x = xflat.reshape(N, Q)
        dx = T * math.exp(-tau) * node_field(x) - K0 * (L_RING @ x)
        return dx.ravel()

    sol = solve_ivp(
        rhs_tau,
        (0.0, tau_terminal_2),
        x0.ravel(),
        t_eval=tau_eval,
        method="DOP853",
        rtol=2e-12,
        atol=2e-13,
    )
    if not sol.success:
        raise RuntimeError(f"phase-map tau integration: {sol.message}")
    x_all = sol.y.T.reshape(-1, N, Q)
    y_all = x_all.mean(axis=1)
    # tau_eval is [0, tau_values..., 25, 30].
    y_samples = y_all[1 : 1 + len(tau_values)]
    y_terminal_25 = y_all[-2]
    y_terminal = y_all[-1]
    terminal_convergence_error = float(np.linalg.norm(y_terminal - y_terminal_25))

    t_values = T * (1.0 - s_values)
    y_phase = stuart_landau_flow(y_terminal, t_values - T)
    sl_tail = np.linalg.norm(y_samples - y_phase, axis=1)
    fit_mask = (
        (s_values >= 1.0e-3)
        & (s_values <= 5.0e-2)
        & (sl_tail > 5.0e-13)
    )
    sl_slope = float(
        np.polyfit(np.log(s_values[fit_mask]), np.log(sl_tail[fit_mask]), 1)[0]
    )

    scalar_tail = T / (RHO_STAR + 1.0) * s_values ** (RHO_STAR + 1.0)
    scalar_slope = float(np.polyfit(np.log(s_values), np.log(scalar_tail), 1)[0])
    anchor_idx = int(np.argmin(np.abs(s_values - 2.0e-2)))
    sl_reference = sl_tail[anchor_idx] * (
        s_values / s_values[anchor_idx]
    ) ** (2.0 * RHO_STAR + 1.0)

    theta_T = np.asarray(stuart_landau_flow(y_terminal, -T))
    terminal_reconstruction = np.asarray(stuart_landau_flow(theta_T, T))
    phase_map_closure_error = float(np.linalg.norm(terminal_reconstruction - y_terminal))
    phase_shift = float(np.linalg.norm(theta_T - x0.mean(axis=0)))

    rows = [
        [s, scalar, sl, ref]
        for s, scalar, sl, ref in zip(
            s_values, scalar_tail, sl_tail, sl_reference
        )
    ]
    write_csv(
        ROOT / "phase_map_validation.csv",
        [
            "s",
            "scalar_phase_tail",
            "stuart_landau_phase_tail",
            "stuart_landau_s3p4_reference",
        ],
        rows,
    )

    return {
        "generic_predicted_slope": RHO_STAR + 1.0,
        "scalar_fitted_slope": scalar_slope,
        "stuart_landau_predicted_slope": 2.0 * RHO_STAR + 1.0,
        "stuart_landau_fitted_slope": sl_slope,
        "fit_window": [1.0e-3, 5.0e-2],
        "fit_points": int(np.count_nonzero(fit_mask)),
        "terminal_base_state": y_terminal.tolist(),
        "terminal_tau": tau_terminal_2,
        "terminal_convergence_error_tau25_to_tau30": terminal_convergence_error,
        "phase_point_at_zero": theta_T.tolist(),
        "phase_shift_norm": phase_shift,
        "phase_map_closure_error": phase_map_closure_error,
        "minimum_reported_tail": float(np.min(sl_tail)),
        "maximum_reported_tail": float(np.max(sl_tail)),
    }


def main() -> None:
    x0 = initial_condition()
    t_end = T - EPS_T
    t_eval = np.linspace(0.0, t_end, 4001)
    ybar = solve_reduced(x0.mean(axis=0), t_eval)
    ybar_phase = np.unwrap(np.arctan2(ybar[:, 1], ybar[:, 0]))

    # Noise experiment determines the application cap.
    phase_offsets = 2.0 * np.array([0.08, -0.05, 0.12, -0.10, 0.03, -0.08])
    phase_offsets -= phase_offsets.mean()
    modal_initial = V_RING.T @ phase_offsets
    sigma = 0.002
    opt = minimize_scalar(
        lambda k: modal_terminal_mse(k, sigma, LAM_RING, modal_initial),
        bounds=(1.0 / T, 20.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    k_opt = float(opt.x)

    methods = ["constant", "fiber_ptc", "xu_liu_l2", "capped", "full_clock"]
    labels = {
        "constant": "constant profile",
        "fiber_ptc": "inverse-time profile",
        "xu_liu_l2": "inverse-square / Xu--Liu (ell=2)",
        "capped": f"capped inverse-time (K={k_opt:.3f})",
        "full_clock": "full-clock ablation",
    }
    classes = {
        "constant": "bounded nonadmissible profile",
        "fiber_ptc": "admissible fiber-only profile",
        "xu_liu_l2": "admissible fiber-only profile",
        "capped": "finite capped profile",
        "full_clock": "tangential-clock ablation",
    }
    sims: Dict[str, Dict[str, NDArray[np.float64] | float]] = {}
    metrics_rows: list[list[object]] = []
    for method in methods:
        cap = k_opt if method == "capped" else None
        sim = simulate_deterministic(method, x0, t_eval, cap=cap, compute_energy=True)
        sims[method] = sim
        y = np.asarray(sim["y"])
        phase = np.asarray(sim["phase"])
        base_error = np.linalg.norm(y - ybar, axis=1)
        offset_ratio = normalized_offset_functional(method, cap)
        metrics_rows.append(
            [
                method,
                labels[method],
                classes[method],
                float(sim["achieved_final_time"]),
                offset_ratio,
                float(np.asarray(sim["disagreement"])[-1]),
                float(np.max(base_error)),
                float(base_error[-1]),
                float(phase[-1] - ybar_phase[-1]),
                float(np.max(np.asarray(sim["u_norm"]))),
                float(sim["energy"]),
            ]
        )
    write_csv(
        ROOT / "method_metrics.csv",
        [
            "method",
            "label",
            "profile_class",
            "achieved_final_time",
            "normalized_offset_functional_I_over_T",
            "terminal_disagreement_at_common_snapshot",
            "max_reduced_trajectory_error",
            "terminal_reduced_trajectory_error",
            "terminal_continuous_phase_error",
            "peak_control_norm",
            "control_energy_to_common_snapshot",
        ],
        metrics_rows,
    )

    # Full-clock regression: continuous phase, closed-form excess, and the
    # one-revolution alias produced by the legacy uniform-t unwrap.
    full = sims["full_clock"]
    continuous_error = float(np.asarray(full["phase"])[-1] - ybar_phase[-1])
    legacy_error = float(np.asarray(full["legacy_phase"])[-1] - ybar_phase[-1])
    tau_star = float(full["achieved_final_tau"])
    closed_form_excess = OMEGA * (tau_star - t_end)
    full_clock_regression = {
        "common_snapshot": t_end,
        "T_minus_snapshot": T - t_end,
        "tau_at_snapshot": tau_star,
        "continuous_phase_error": continuous_error,
        "legacy_uniform_t_unwrap_error": legacy_error,
        "alias_gap": continuous_error - legacy_error,
        "alias_gap_over_2pi": (continuous_error - legacy_error) / (2.0 * math.pi),
        "limit_cycle_closed_form_excess": closed_form_excess,
        "continuous_minus_closed_form": continuous_error - closed_form_excess,
    }

    # Exponent validation across graph eigenstructures.
    exponent_rows: list[list[object]] = []
    target_rhos = [0.6, 1.0, 1.4]
    s_grid = np.geomspace(2e-5, 0.2, 500)[::-1]
    t_slope = T * (1.0 - s_grid)
    for graph_kind in ["path", "ring", "complete"]:
        lap = graph_laplacian(graph_kind)
        lambda2 = float(np.linalg.eigvalsh(lap)[1])
        for target_rho in target_rhos:
            k0 = target_rho / lambda2
            sim = simulate_deterministic(
                "fiber_ptc", x0, t_slope, lap=lap, k0=k0, compute_energy=False
            )
            dis = np.asarray(sim["disagreement"])
            mask = (s_grid < 0.03) & (s_grid > 2e-4) & (dis > 1e-12)
            fitted = float(np.polyfit(np.log(s_grid[mask]), np.log(dis[mask]), 1)[0])
            exponent_rows.append(
                [graph_kind, lambda2, k0, target_rho, fitted, fitted - target_rho]
            )
    write_csv(
        ROOT / "exponent_validation.csv",
        ["graph", "lambda2", "k0", "predicted_rho", "fitted_rho", "error"],
        exponent_rows,
    )

    # Quadratic projection-based tangential offset.
    base = np.array([0.8 * math.cos(0.2), 0.8 * math.sin(0.2)])
    direction = np.array(
        [[1.0, 0.2], [-0.7, 0.5], [0.3, -1.1], [-0.5, -0.4], [0.8, 0.9], [-0.9, -0.1]]
    )
    direction -= direction.mean(axis=0)
    direction /= np.linalg.norm(direction)
    eps_values = np.geomspace(0.015, 0.6, 12)
    tangent_rows: list[list[object]] = []
    t_short = np.linspace(0.0, t_end, 1601)
    base_reduced = solve_reduced(base, t_short)[-1]
    tangent_errors = []
    for eps in eps_values:
        x_eps = base[None, :] + eps * direction
        sim = simulate_deterministic(
            "fiber_ptc", x_eps, t_short, compute_energy=False
        )
        err = float(np.linalg.norm(np.asarray(sim["y"])[-1] - base_reduced))
        tangent_errors.append(err)
        tangent_rows.append([eps, err])
    fit_count = 8
    tangent_slope = float(
        np.polyfit(np.log(eps_values[:fit_count]), np.log(tangent_errors[:fit_count]), 1)[0]
    )
    write_csv(
        ROOT / "tangential_scaling.csv",
        ["initial_disagreement_scale", "terminal_projection_orbit_error"],
        tangent_rows,
    )

    phase_map = phase_map_validation(x0)

    # Nonlinear Monte Carlo validation of the modal cap formula.
    caps = np.geomspace(0.5, 10.0, 13)
    noise_rows: list[list[object]] = []
    for idx, cap in enumerate(caps):
        exact = modal_terminal_mse(float(cap), sigma, LAM_RING, modal_initial)
        mc_mean, mc_se, ci_low, ci_high = nonlinear_noise_mc(
            float(cap), sigma, phase_offsets, paths=800, dt=0.001, seed=7100 + idx
        )
        noise_rows.append(
            [cap, exact, mc_mean, mc_se, ci_low, ci_high, (mc_mean - exact) / exact]
        )
    write_csv(
        ROOT / "network_noise_validation.csv",
        [
            "K",
            "modal_exact_mse",
            "nonlinear_mc_mse",
            "mc_se",
            "ci95_low",
            "ci95_high",
            "relative_error",
        ],
        noise_rows,
    )

    method_metrics = {row[0]: row for row in metrics_rows}
    max_exp_error = max(abs(float(r[5])) for r in exponent_rows)
    max_noise_rel = max(abs(float(r[6])) for r in noise_rows)
    main_mask = (T - t_eval) / T < 0.03
    fitted_main_exponent = float(
        np.polyfit(
            np.log((T - t_eval[main_mask]) / T),
            np.log(np.asarray(sims["fiber_ptc"]["disagreement"])[main_mask]),
            1,
        )[0]
    )

    summary = {
        "experiment": "Network",
        "N": N,
        "T": T,
        "common_snapshot": t_end,
        "T_minus_snapshot": EPS_T,
        "resolution_floor": RESOLUTION_FLOOR,
        "alpha": ALPHA,
        "b_sl": B_SL,
        "omega": OMEGA,
        "k0": K0,
        "lambda2_ring": LAMBDA2,
        "rho_star": RHO_STAR,
        "fitted_main_exponent": fitted_main_exponent,
        "max_graph_exponent_absolute_error": max_exp_error,
        "quadratic_tangential_offset_slope": tangent_slope,
        "phase_map_validation": phase_map,
        "full_clock_phase_regression": full_clock_regression,
        "sigma_network": sigma,
        "Kopt_network_modal": k_opt,
        "max_nonlinear_modal_relative_mse_error": max_noise_rel,
        "method_metrics": {
            k: {
                "label": str(v[1]),
                "profile_class": str(v[2]),
                "achieved_final_time": float(v[3]),
                "normalized_offset_I_over_T": float(v[4]),
                "terminal_disagreement": float(v[5]),
                "terminal_disagreement_report": (
                    f"<{RESOLUTION_FLOOR:.0e}"
                    if k == "xu_liu_l2" and float(v[5]) < RESOLUTION_FLOOR
                    else f"{float(v[5]):.9g}"
                ),
                "max_reduced_error": float(v[6]),
                "terminal_reduced_error": float(v[7]),
                "phase_error": float(v[8]),
                "peak_control": float(v[9]),
                "energy": float(v[10]),
            }
            for k, v in method_metrics.items()
        },
    }

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "common achieved snapshot",
        all(abs(float(v[3]) - t_end) < 1e-14 for v in metrics_rows),
        {str(v[0]): float(v[3]) for v in metrics_rows},
    )
    check(
        "full-clock continuous phase matches closed-form excess",
        abs(full_clock_regression["continuous_minus_closed_form"]) < 2e-4,
        full_clock_regression,
    )
    check(
        "legacy unwrap misses exactly one revolution",
        abs(full_clock_regression["alias_gap"] - 2.0 * math.pi) < 1e-8,
        full_clock_regression["alias_gap"],
    )
    check(
        "corrected phase divergence exceeds legacy value",
        continuous_error > legacy_error + 6.0,
        {"continuous": continuous_error, "legacy": legacy_error},
    )
    check(
        "inverse-square disagreement is below reporting resolution",
        float(method_metrics["xu_liu_l2"][5]) < RESOLUTION_FLOOR,
        float(method_metrics["xu_liu_l2"][5]),
    )
    check(
        "inverse-time normalized offset exact",
        abs(float(method_metrics["fiber_ptc"][4]) - 1.0 / (RHO_STAR + 1.0)) < 1e-13,
        float(method_metrics["fiber_ptc"][4]),
    )
    check(
        "inverse-square normalized offset exact",
        abs(float(method_metrics["xu_liu_l2"][4]) - 0.368878561726258) < 1e-13,
        float(method_metrics["xu_liu_l2"][4]),
    )
    check(
        "offset ordering predicted by schedule profile",
        float(method_metrics["xu_liu_l2"][4])
        < float(method_metrics["fiber_ptc"][4])
        < float(method_metrics["constant"][4]),
        {k: float(method_metrics[k][4]) for k in ["constant", "fiber_ptc", "xu_liu_l2"]},
    )
    check(
        "scalar phase-tail exponent",
        abs(float(phase_map["scalar_fitted_slope"]) - (RHO_STAR + 1.0)) < 1e-10,
        phase_map["scalar_fitted_slope"],
    )
    check(
        "Stuart--Landau phase-tail exponent",
        abs(float(phase_map["stuart_landau_fitted_slope"]) - (2.0 * RHO_STAR + 1.0)) < 0.03,
        phase_map["stuart_landau_fitted_slope"],
    )
    check(
        "quadratic cancellation selected over generic exponent",
        abs(float(phase_map["stuart_landau_fitted_slope"]) - (2.0 * RHO_STAR + 1.0))
        < abs(float(phase_map["stuart_landau_fitted_slope"]) - (RHO_STAR + 1.0)),
        phase_map["stuart_landau_fitted_slope"],
    )
    check(
        "phase-map backward-forward closure",
        float(phase_map["phase_map_closure_error"]) < 1e-12,
        phase_map["phase_map_closure_error"],
    )
    check(
        "terminal base convergence in regular time",
        float(phase_map["terminal_convergence_error_tau25_to_tau30"]) < 1e-10,
        phase_map["terminal_convergence_error_tau25_to_tau30"],
    )
    check(
        "main fiber exponent preserved",
        abs(fitted_main_exponent - 1.1996984425895174) < 2e-10,
        fitted_main_exponent,
    )
    check(
        "graph exponent sweep preserved",
        abs(max_exp_error - 0.001450911842219904) < 2e-10,
        max_exp_error,
    )
    check(
        "quadratic projection-orbit slope preserved",
        abs(tangent_slope - 1.9761166019943055) < 2e-10,
        tangent_slope,
    )
    check(
        "network cap optimum preserved",
        abs(k_opt - 2.53979193583334) < 2e-10,
        k_opt,
    )
    check(
        "nonlinear/modal maximum discrepancy preserved",
        abs(max_noise_rel - 0.044480096385107616) < 2e-10,
        max_noise_rel,
    )
    peak_values = [float(method_metrics[k][9]) for k in methods]
    check(
        "equal initial and peak coupling norm",
        max(peak_values) - min(peak_values) < 2e-8,
        peak_values,
    )
    check(
        "finite verified coupling energies",
        all(math.isfinite(float(method_metrics[k][10])) and float(method_metrics[k][10]) > 0.0 for k in methods),
        {k: float(method_metrics[k][10]) for k in methods},
    )

    validation = {
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    summary["network_numerical_validation"] = {
        "status": validation["status"],
        "checks_passed": validation["checks_passed"],
        "checks_total": validation["checks_total"],
    }

    (ROOT / "numerical_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "network_numerical_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    validation_lines = [
        "Network NUMERICAL COMPARISON CLOSURE",
        "=" * 49,
        f"Status: {validation['status']}",
        f"Checks passed: {validation['checks_passed']}/{validation['checks_total']}",
        f"Full-clock continuous phase error: {continuous_error:.12f} rad",
        f"Legacy unwrap value: {legacy_error:.12f} rad",
        f"Alias gap: {full_clock_regression['alias_gap']:.12f} rad",
        f"Stuart--Landau phase-tail slope: {phase_map['stuart_landau_fitted_slope']:.9f}",
        f"Scalar phase-tail slope: {phase_map['scalar_fitted_slope']:.9f}",
        f"Inverse-square terminal disagreement: {float(method_metrics['xu_liu_l2'][5]):.6e}",
        f"Reporting rule: <{RESOLUTION_FLOOR:.0e} below numerical resolution",
    ]
    (ROOT / "network_numerical_validation.txt").write_text("\n".join(validation_lines) + "\n")

    (ROOT / "numerical_summary.txt").write_text(
        "\n".join(
            [
                f"ring lambda2 = {LAMBDA2:.12g}",
                f"rho_star = {RHO_STAR:.12g}",
                f"common snapshot = T - {EPS_T:.1e} = {t_end:.12g}",
                f"main fitted exponent = {fitted_main_exponent:.9f}",
                f"maximum graph-exponent absolute error = {max_exp_error:.3e}",
                f"quadratic tangential-offset fitted slope = {tangent_slope:.9f}",
                f"scalar phase-tail fitted slope = {phase_map['scalar_fitted_slope']:.9f}",
                f"Stuart-Landau phase-tail fitted slope = {phase_map['stuart_landau_fitted_slope']:.9f}",
                f"full-clock continuous phase error = {continuous_error:.9f} rad",
                f"legacy phase-unwrapping value = {legacy_error:.9f} rad",
                f"network modal Kopt = {k_opt:.9f}",
                f"maximum nonlinear/modal relative MSE error = {max_noise_rel:.6%}",
                f"Network numerical closure = {validation['status']} ({validation['checks_passed']}/{validation['checks_total']})",
            ]
        )
        + "\n"
    )
    print((ROOT / "numerical_summary.txt").read_text(), end="")
    print((ROOT / "network_numerical_validation.txt").read_text(), end="")
    if validation["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
