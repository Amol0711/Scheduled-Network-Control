#!/usr/bin/env python3
"""Independent validation of the schedule noise schedule-shaped noise laws.

The script checks the exact terminal-moment identity for a capped deterministic
schedule, the named inverse-time and inverse-square specializations, the
normalized p-power family asymptotics, and the optimized small-noise laws.
Validation layers are deliberately independent:

1. symbolic algebra (SymPy),
2. direct quadrature of the Itô-isometry integral,
3. integration of the deterministic mean/variance ODEs,
4. piecewise-exact Monte Carlo simulation of the scalar SDE.

Only CSV/JSON/text data are written.  No figures are generated in Python.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize_scalar
from scipy.special import lambertw

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()
OUT_JSON = ROOT / "schedule_noise_validation.json"
OUT_TXT = ROOT / "schedule_noise_validation.txt"
OUT_OPTIMA = ROOT / "schedule_noise_optima.csv"
OUT_COMPONENTS = ROOT / "schedule_noise_components.csv"


@dataclass
class Check:
    name: str
    value: float
    reference: float
    absolute_error: float
    tolerance: float
    passed: bool
    layer: str


def add_check(
    checks: list[Check],
    name: str,
    value: float,
    reference: float,
    tolerance: float,
    layer: str,
) -> None:
    err = abs(float(value) - float(reference))
    checks.append(Check(name, float(value), float(reference), err,
                        float(tolerance), bool(err <= tolerance), layer))


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def kappa_power(t: float, p: float, T: float) -> float:
    return T ** (p - 1.0) / (T - t) ** p


def capped_power(t: float, p: float, T: float, K: float) -> float:
    if t >= T:
        return K
    return min(kappa_power(t, p, T), K)


def activation_power(p: float, T: float, K: float) -> tuple[float, float]:
    if K < 1.0 / T:
        raise ValueError("K must satisfy K >= 1/T")
    a = (K * T) ** (1.0 / p)
    return T * (1.0 - 1.0 / a), a


def j_power(p: float, T: float, K: float) -> float:
    _, a = activation_power(p, T, K)
    if abs(p - 1.0) <= 1e-14:
        return math.log(K * T) + 1.0
    return (p * a ** (p - 1.0) - 1.0) / (p - 1.0)


def general_terminal_moments(
    kappa_cap: Callable[[float], float],
    rho: float,
    T: float,
    eta0: float,
    sigma: float,
    breakpoints: Iterable[float] = (),
) -> tuple[float, float, float, float]:
    """Direct quadrature of J, mean, variance, and MSE."""
    points = sorted(x for x in breakpoints if 0.0 < x < T)
    intervals = [0.0, *points, T]
    J = 0.0
    for left, right in zip(intervals[:-1], intervals[1:]):
        J += quad(kappa_cap, left, right, epsabs=2e-12, epsrel=2e-12,
                  limit=400)[0]

    # Compute the terminal transition by an inner quadrature.  This is used
    # only for a modest number of validation points; the named closed forms
    # and moment ODE provide the production evaluation.
    def terminal_transition(s: float) -> float:
        local_points = [x for x in points if s < x < T]
        local_intervals = [s, *local_points, T]
        tail = 0.0
        for left, right in zip(local_intervals[:-1], local_intervals[1:]):
            tail += quad(kappa_cap, left, right, epsabs=2e-11,
                         epsrel=2e-11, limit=300)[0]
        return math.exp(-rho * tail)

    variance = 0.0
    for left, right in zip(intervals[:-1], intervals[1:]):
        variance += quad(
            lambda s: rho * rho * sigma * sigma
            * terminal_transition(s) ** 2 * kappa_cap(s) ** 2,
            left,
            right,
            epsabs=2e-11,
            epsrel=2e-11,
            limit=350,
        )[0]
    mean = eta0 * math.exp(-rho * J)
    return J, mean, variance, mean * mean + variance


def moment_ode_terminal(
    kappa_cap: Callable[[float], float],
    rho: float,
    T: float,
    eta0: float,
    sigma: float,
    breakpoints: Iterable[float] = (),
) -> tuple[float, float, float]:
    """Independent integration of m' and v'."""
    z = np.array([eta0, 0.0], dtype=float)
    points = sorted(x for x in breakpoints if 0.0 < x < T)
    intervals = [0.0, *points, T]

    def rhs(t: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        k = kappa_cap(t)
        return np.array([
            -rho * k * state[0],
            -2.0 * rho * k * state[1] + rho * rho * sigma * sigma * k * k,
        ])

    for left, right in zip(intervals[:-1], intervals[1:]):
        sol = solve_ivp(
            rhs,
            (left, right),
            z,
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
            max_step=T / 1500.0,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        z = sol.y[:, -1]
    mean, variance = map(float, z)
    return mean, variance, mean * mean + variance


def p_poly(z: float, rho: float) -> float:
    lam = 2.0 * rho
    return z * z / lam - 2.0 * z / (lam * lam) + 2.0 / (lam ** 3)


def inverse_time_components(
    K: float, rho: float, T: float, eta0: float, sigma: float
) -> dict[str, float]:
    if K < 1.0 / T:
        raise ValueError("K must satisfy K >= 1/T")
    J = math.log(K * T) + 1.0
    mean = eta0 * math.exp(-rho) * (K * T) ** (-rho)
    v_pre = (
        rho * rho * sigma * sigma * math.exp(-2.0 * rho) / (2.0 * rho + 1.0)
        * (K - K ** (-2.0 * rho) * T ** (-(2.0 * rho + 1.0)))
    )
    v_cap = 0.5 * rho * sigma * sigma * K * (1.0 - math.exp(-2.0 * rho))
    variance = v_pre + v_cap
    return {
        "J": J,
        "mean": mean,
        "V_pre": v_pre,
        "V_cap": v_cap,
        "variance": variance,
        "mse": mean * mean + variance,
    }


def inverse_square_components(
    K: float, rho: float, T: float, eta0: float, sigma: float
) -> dict[str, float]:
    if K < 1.0 / T:
        raise ValueError("K must satisfy K >= 1/T")
    a = math.sqrt(K * T)
    J = 2.0 * a - 1.0
    mean = eta0 * math.exp(-rho * J)
    v_pre = rho * rho * sigma * sigma / T * (
        math.exp(-2.0 * rho * a) * p_poly(a, rho)
        - math.exp(-4.0 * rho * a + 2.0 * rho) * p_poly(1.0, rho)
    )
    v_cap = 0.5 * rho * sigma * sigma * K * (
        1.0 - math.exp(-2.0 * rho * a)
    )
    variance = v_pre + v_cap
    d_rho = (2.0 * rho * rho - 2.0 * rho + 1.0) / (4.0 * rho)
    compact_mse = (
        sigma * sigma / T
        * (0.5 * rho * a * a
           + (1.0 / (4.0 * rho) - 0.5 * a) * math.exp(-2.0 * rho * a))
        + (eta0 * eta0 - sigma * sigma * d_rho / T)
        * math.exp(-4.0 * rho * a + 2.0 * rho)
    )
    return {
        "a": a,
        "J": J,
        "mean": mean,
        "V_pre": v_pre,
        "V_cap": v_cap,
        "variance": variance,
        "mse": mean * mean + variance,
        "compact_mse": compact_mse,
        "D_rho": d_rho,
    }


def power_components_numeric(
    p: float, K: float, rho: float, T: float, eta0: float, sigma: float
) -> dict[str, float]:
    if p <= 1.0:
        raise ValueError("This transformed formula is for p>1")
    _, a = activation_power(p, T, K)
    ap = a ** (p - 1.0)
    J = (p * ap - 1.0) / (p - 1.0)
    mean = eta0 * math.exp(-rho * J)
    v_pre = rho * rho * sigma * sigma / T * quad(
        lambda z: z ** (2.0 * p - 2.0)
        * math.exp(
            -2.0 * rho * (p * ap - z ** (p - 1.0)) / (p - 1.0)
        ),
        1.0,
        a,
        epsabs=2e-13,
        epsrel=2e-12,
        limit=400,
    )[0]
    v_cap = 0.5 * rho * sigma * sigma * K * (
        1.0 - math.exp(-2.0 * rho * ap)
    )
    return {
        "a": a,
        "J": J,
        "mean": mean,
        "V_pre": v_pre,
        "V_cap": v_cap,
        "variance": v_pre + v_cap,
        "mse": mean * mean + v_pre + v_cap,
    }


def inverse_time_exact_optimum(
    rho: float, T: float, eta0: float, sigma: float
) -> tuple[float, float]:
    b_rho = 0.5 * rho * (1.0 - math.exp(-2.0 * rho))
    c_rho = rho * rho * math.exp(-2.0 * rho) / (2.0 * rho + 1.0)
    b_big = b_rho + c_rho
    A = abs(eta0) * math.exp(-rho) * T ** (-rho)
    delta = A * A - c_rho * sigma * sigma * T ** (-(2.0 * rho + 1.0))
    interior = (2.0 * rho * max(delta, 0.0) / (b_big * sigma * sigma)) ** (
        1.0 / (2.0 * rho + 1.0)
    ) if sigma > 0.0 else math.inf
    K = max(1.0 / T, interior)
    return K, inverse_time_components(K, rho, T, eta0, sigma)["mse"]


def minimize_inverse_square(
    rho: float, T: float, eta0: float, sigma: float
) -> tuple[float, float, float]:
    zarg = 16.0 * rho * T * eta0 * eta0 * math.exp(2.0 * rho) / (sigma * sigma)
    a_asym = float(lambertw(zarg).real) / (4.0 * rho)
    upper = max(8.0, 3.0 * a_asym + 5.0)
    result = minimize_scalar(
        lambda a: inverse_square_components(a * a / T, rho, T, eta0, sigma)["mse"],
        bounds=(1.0, upper),
        method="bounded",
        options={"xatol": 2e-13, "maxiter": 1200},
    )
    if not result.success:
        raise RuntimeError(result.message)
    a_opt = float(result.x)
    return a_opt * a_opt / T, float(result.fun), a_asym


def minimize_power(
    p: float, rho: float, T: float, eta0: float, sigma: float
) -> tuple[float, float, float]:
    H = eta0 * eta0 * math.exp(2.0 * rho / (p - 1.0))
    D = 4.0 * T * H / (sigma * sigma)
    warg = 2.0 * rho * p * D ** (p - 1.0)
    a_asym = (float(lambertw(warg).real) / (2.0 * rho * p)) ** (
        1.0 / (p - 1.0)
    )
    upper = max(6.0, 2.5 * a_asym + 4.0)
    result = minimize_scalar(
        lambda a: power_components_numeric(
            p, a ** p / T, rho, T, eta0, sigma
        )["mse"],
        bounds=(1.0, upper),
        method="bounded",
        options={"xatol": 2e-12, "maxiter": 1200},
    )
    if not result.success:
        raise RuntimeError(result.message)
    a_opt = float(result.x)
    return a_opt ** p / T, float(result.fun), a_asym


def piecewise_exact_mc(
    p: float,
    K: float,
    rho: float,
    T: float,
    eta0: float,
    sigma: float,
    paths: int,
    steps: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Monte Carlo with exact OU updates for midpoint-frozen gain."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    eta = np.full(paths, eta0, dtype=float)
    for n in range(steps):
        tmid = (n + 0.5) * dt
        k = capped_power(tmid, p, T, K)
        decay = math.exp(-rho * k * dt)
        q = 0.5 * rho * sigma * sigma * k * (1.0 - decay * decay)
        eta = decay * eta + math.sqrt(max(q, 0.0)) * rng.standard_normal(paths)
    sq = eta * eta
    sample = float(np.mean(sq))
    se = float(np.std(sq, ddof=1) / math.sqrt(paths))
    return sample, se, sample - 2.576 * se, sample + 2.576 * se


def main() -> int:
    checks: list[Check] = []
    component_rows: list[list[object]] = []

    # ------------------------------------------------------------------
    # 1. Symbolic identities.
    # ------------------------------------------------------------------
    z, rr = sp.symbols("z rr", positive=True)
    lam = 2 * rr
    P = z**2 / lam - 2 * z / lam**2 + 2 / lam**3
    residual = sp.simplify(sp.diff(sp.exp(lam * z) * P, z) - z**2 * sp.exp(lam * z))
    add_check(checks, "symbolic_inverse_square_antiderivative",
              float(residual), 0.0, 0.0, "symbolic")

    aa, TT = sp.symbols("aa TT", positive=True)
    tt = sp.symbols("tt", real=True)
    t2 = TT * (1 - 1 / aa)
    k2 = aa**2 / TT
    J2 = sp.integrate(TT / (TT - tt)**2, (tt, 0, t2)) + k2 * (TT - t2)
    add_check(checks, "symbolic_inverse_square_cap_integral",
              float(sp.N(sp.simplify(J2 - (2 * aa - 1)).subs({aa: 2, TT: 3}))),
              0.0, 0.0, "symbolic")

    kk = sp.symbols("kk", positive=True)
    t1 = TT - 1 / kk
    J1 = sp.integrate(1 / (TT - tt), (tt, 0, t1)) + kk * (TT - t1)
    j1_res = sp.simplify(J1 - (sp.log(kk * TT) + 1))
    add_check(checks, "symbolic_inverse_time_cap_integral",
              float(sp.N(j1_res.subs({kk: 2, TT: 1}))), 0.0, 0.0, "symbolic")

    # ------------------------------------------------------------------
    # 2. Closed forms against direct quadrature and moment ODEs.
    # ------------------------------------------------------------------
    parameter_sets = [
        (0.6, 0.7, 0.8, 0.03),
        (1.2, 1.0, 1.0, 0.02),
        (2.1, 1.8, -0.7, 0.015),
    ]
    multipliers = (1.0, 1.7, 5.0, 24.0)
    for rho, T, eta0, sigma in parameter_sets:
        for mult in multipliers:
            K = mult / T
            for p, name, exact_fun in (
                (1.0, "inverse_time", inverse_time_components),
                (2.0, "inverse_square", inverse_square_components),
            ):
                tK, _ = activation_power(p, T, K)
                exact = exact_fun(K, rho, T, eta0, sigma)
                kcap = lambda t, pp=p, TT_=T, KK=K: capped_power(t, pp, TT_, KK)
                Jq, mq, vq, eq = general_terminal_moments(
                    kcap, rho, T, eta0, sigma, (tK,)
                )
                mo, vo, eo = moment_ode_terminal(
                    kcap, rho, T, eta0, sigma, (tK,)
                )
                tag = f"{name}_rho{rho}_T{T}_m{mult}"
                add_check(checks, tag + "_J_quadrature", Jq, exact["J"], 2e-9,
                          "quadrature")
                add_check(checks, tag + "_mean_quadrature", mq, exact["mean"], 2e-10,
                          "quadrature")
                add_check(checks, tag + "_variance_quadrature", vq, exact["variance"],
                          2e-9 * max(1.0, abs(exact["variance"])), "quadrature")
                add_check(checks, tag + "_mse_quadrature", eq, exact["mse"],
                          2e-9 * max(1.0, abs(exact["mse"])), "quadrature")
                add_check(checks, tag + "_mean_moment_ode", mo, exact["mean"], 2e-9,
                          "moment_ode")
                add_check(checks, tag + "_variance_moment_ode", vo, exact["variance"],
                          2e-8 * max(1.0, abs(exact["variance"])), "moment_ode")
                add_check(checks, tag + "_mse_moment_ode", eo, exact["mse"],
                          2e-8 * max(1.0, abs(exact["mse"])), "moment_ode")
                component_rows.append([
                    name, rho, T, eta0, sigma, K, exact["J"], exact["mean"],
                    exact["V_pre"], exact["V_cap"], exact["variance"], exact["mse"]
                ])
                if p == 2.0:
                    add_check(checks, tag + "_compact_mse_identity", exact["compact_mse"],
                              exact["mse"], 5e-14 * max(1.0, abs(exact["mse"])),
                              "algebraic")

    # General identity for a non-power schedule with multiple smooth features.
    rho, T, eta0, sigma, K = 1.35, 1.4, 0.9, 0.025, 4.0
    def irregular_profile(t: float) -> float:
        s = (T - t) / T
        return 1.0 / (T - t) + 0.28 / T * (1.0 + math.sin(7.0 * math.pi * t / T)) ** 2 + 0.06 / (T * math.sqrt(s))
    kcap_irregular = lambda t: K if t >= T else min(irregular_profile(t), K)
    Jq, mq, vq, eq = general_terminal_moments(
        kcap_irregular, rho, T, eta0, sigma
    )
    mo, vo, eo = moment_ode_terminal(
        kcap_irregular, rho, T, eta0, sigma
    )
    add_check(checks, "general_profile_mean_identity", mo, mq, 2e-8, "general_identity")
    add_check(checks, "general_profile_variance_identity", vo, vq,
              3e-8 * max(1.0, abs(vq)), "general_identity")
    add_check(checks, "general_profile_mse_identity", eo, eq,
              3e-8 * max(1.0, abs(eq)), "general_identity")

    # ------------------------------------------------------------------
    # 3. Cap-layer factor and p-family asymptotics.
    # ------------------------------------------------------------------
    rho, T, eta0, sigma = 1.2, 1.0, 1.0, 0.02
    for p in (1.0, 2.0, 3.0, 5.0):
        for K in (2.0, 16.0, 128.0):
            tK, a = activation_power(p, T, K)
            expected_mass = 1.0 if p == 1.0 else a ** (p - 1.0)
            add_check(checks, f"cap_mass_p{p}_K{K}", K * (T - tK),
                      expected_mass, 2e-13, "cap_layer")
            vcap_direct = rho * rho * sigma * sigma * quad(
                lambda s: math.exp(-2.0 * rho * K * (T - s)) * K * K,
                tK, T, epsabs=2e-13, epsrel=2e-13, limit=250
            )[0]
            vcap_formula = 0.5 * rho * sigma * sigma * K * (
                1.0 - math.exp(-2.0 * rho * K * (T - tK))
            )
            add_check(checks, f"cap_variance_factor_p{p}_K{K}", vcap_direct,
                      vcap_formula, 2e-13, "cap_layer")

    for a in (4.0, 8.0, 16.0):
        K = a * a / T
        comp = inverse_square_components(K, rho, T, eta0, sigma)
        leading_pre = 0.5 * rho * sigma * sigma * K * math.exp(-2.0 * rho * a)
        ratio = comp["V_pre"] / leading_pre
        tolerance = 0.90 / a + 0.005
        add_check(checks, f"inverse_square_precap_laplace_a{a}", ratio, 1.0,
                  tolerance, "asymptotic")

    # p=3 exact cap integral and leading variance ratio.
    p = 3.0
    for K in (8.0, 64.0, 512.0):
        tK, _ = activation_power(p, T, K)
        Jq = quad(lambda t: capped_power(t, p, T, K), 0.0, tK,
                  epsabs=2e-13, epsrel=2e-13)[0] + K * (T - tK)
        comp = power_components_numeric(p, K, rho, T, eta0, sigma)
        add_check(checks, f"p3_cap_integral_K{K}", Jq, comp["J"], 2e-12,
                  "power_family")
        variance_ratio = comp["variance"] / (0.5 * rho * sigma * sigma * K)
        add_check(checks, f"p3_leading_variance_K{K}", variance_ratio, 1.0,
                  2.5 * math.exp(-2.0 * rho * (K * T) ** ((p - 1.0) / p)) + 2e-4,
                  "power_family")

    # ------------------------------------------------------------------
    # 4. Optimizers and small-noise exponents.
    # ------------------------------------------------------------------
    rho, T, eta0 = 1.2, 1.0, 1.0
    sigmas = np.geomspace(1e-9, 1e-3, 31)
    optimum_rows: list[list[object]] = []
    rms1: list[float] = []
    rms2: list[float] = []
    for sigma in sigmas:
        K1, E1 = inverse_time_exact_optimum(rho, T, eta0, float(sigma))
        K2, E2, a2_asym = minimize_inverse_square(rho, T, eta0, float(sigma))
        W2 = float(lambertw(
            16.0 * rho * T * eta0 * eta0 * math.exp(2.0 * rho) / (sigma * sigma)
        ).real)
        K2_asym = W2 * W2 / (16.0 * rho * rho * T)
        R2_asym = float(sigma) * W2 / (4.0 * math.sqrt(2.0 * rho * T))
        R1 = math.sqrt(E1)
        R2 = math.sqrt(E2)
        rms1.append(R1)
        rms2.append(R2)
        optimum_rows.append([
            float(sigma), "inverse_time", K1, math.sqrt(K1 * T), R1,
            "exact", 2.0 * rho / (2.0 * rho + 1.0)
        ])
        optimum_rows.append([
            float(sigma), "inverse_square", K2, math.sqrt(K2 * T), R2,
            K2_asym, R2_asym
        ])
        add_check(checks, f"inverse_square_K_lambert_sigma{sigma:.3e}",
                  K2 / K2_asym, 1.0, 0.004, "optimizer")
        add_check(checks, f"inverse_square_RMS_lambert_sigma{sigma:.3e}",
                  R2 / R2_asym, 1.0, 0.25 / math.sqrt(K2 * T) + 0.01, "optimizer")
        add_check(checks, f"inverse_square_a_internal_sigma{sigma:.3e}",
                  math.sqrt(K2 * T), a2_asym, 0.01, "optimizer")

    slope1 = float(np.polyfit(np.log(sigmas), np.log(rms1), 1)[0])
    slope2 = float(np.polyfit(np.log(sigmas), np.log(rms2), 1)[0])
    predicted1 = 2.0 * rho / (2.0 * rho + 1.0)
    add_check(checks, "inverse_time_fitted_rms_slope", slope1, predicted1,
              2e-4, "scaling_fit")
    add_check(checks, "inverse_square_finite_range_rms_slope", slope2,
              0.9341354322738368, 3e-4, "scaling_fit")

    # Power-family correction: the log exponent tends to 1/2, not zero.
    for p in (2.0, 3.0, 5.0, 10.0, 100.0):
        exponent = p / (2.0 * (p - 1.0))
        expected = 0.5 + 1.0 / (2.0 * (p - 1.0))
        add_check(checks, f"power_log_exponent_p{p}", exponent, expected,
                  1e-15, "power_family")
    add_check(checks, "power_log_exponent_limit", 10000.0 / (2.0 * 9999.0),
              0.5, 6e-5, "power_family")

    # A direct p=3 optimizer check at very small noise.
    for sigma in (1e-5, 1e-8, 1e-12):
        K3, E3, a3_asym = minimize_power(3.0, rho, T, eta0, sigma)
        add_check(checks, f"p3_K_lambert_sigma{sigma:.0e}",
                  (K3 * T) ** (1.0 / 3.0) / a3_asym, 1.0, 0.018,
                  "power_family")
        optimum_rows.append([
            sigma, "power_p3", K3, (K3 * T) ** (1.0 / 3.0), math.sqrt(E3),
            a3_asym ** 3 / T, 3.0 / 4.0
        ])

    # ------------------------------------------------------------------
    # 5. Monte Carlo checks, independent of all quadrature formulas.
    # ------------------------------------------------------------------
    mc_cases = [
        (1.0, 4.0, 1001),
        (1.0, 16.0, 1002),
        (2.0, 4.0, 2001),
        (2.0, 16.0, 2002),
    ]
    rho, T, eta0, sigma = 1.2, 1.0, 1.0, 0.04
    mc_records: list[dict[str, float | str]] = []
    for p, K, seed in mc_cases:
        exact = (inverse_time_components if p == 1.0 else inverse_square_components)(
            K, rho, T, eta0, sigma
        )["mse"]
        sample, se, lo, hi = piecewise_exact_mc(
            p, K, rho, T, eta0, sigma, paths=20000, steps=8000, seed=seed
        )
        zscore = abs(sample - exact) / se
        checks.append(Check(
            f"monte_carlo_p{p}_K{K}", sample, exact, abs(sample - exact),
            3.2 * se, bool(zscore <= 3.2), "monte_carlo"
        ))
        mc_records.append({
            "schedule": "inverse_time" if p == 1.0 else "inverse_square",
            "K": K,
            "exact_mse": exact,
            "sample_mse": sample,
            "sample_se": se,
            "confidence99_lower": lo,
            "confidence99_upper": hi,
            "zscore": zscore,
        })

    write_csv(
        OUT_COMPONENTS,
        ["schedule", "rho", "T", "eta0", "sigma", "K", "J", "mean",
         "V_pre", "V_cap", "variance", "mse"],
        component_rows,
    )
    write_csv(
        OUT_OPTIMA,
        ["sigma", "schedule", "K_opt", "a_opt", "optimized_rms",
         "asymptotic_K_or_status", "asymptotic_RMS_or_exponent"],
        optimum_rows,
    )

    status = "PASS" if all(item.passed for item in checks) else "FAIL"
    by_layer: dict[str, dict[str, int]] = {}
    for item in checks:
        layer = by_layer.setdefault(item.layer, {"passed": 0, "total": 0})
        layer["total"] += 1
        layer["passed"] += int(item.passed)

    exact_checks = [item for item in checks if item.layer not in {
        "asymptotic", "optimizer", "scaling_fit", "monte_carlo", "power_family"
    }]
    result = {
        "status": status,
        "checks_total": len(checks),
        "checks_passed": sum(item.passed for item in checks),
        "checks_by_layer": by_layer,
        "maximum_absolute_error_exact_layers": max(item.absolute_error for item in exact_checks),
        "reported_scaling": {
            "rho": rho,
            "sigma_range": [float(sigmas[0]), float(sigmas[-1])],
            "inverse_time_fitted_rms_slope": slope1,
            "inverse_time_predicted_rms_slope": predicted1,
            "inverse_square_fitted_rms_slope": slope2,
            "inverse_square_asymptotic_law": "Theta(sigma log(1/sigma))",
            "inverse_square_K_law": "Theta(log(1/sigma)^2)",
            "power_family_log_exponent": "p/[2(p-1)]",
            "power_family_limit_log_exponent": 0.5,
        },
        "monte_carlo": mc_records,
        "checks": [asdict(item) for item in checks],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "schedule noise SCHEDULE-SHAPED NOISE VALIDATION",
        "=" * 53,
        f"Status: {status}",
        f"Checks passed: {result['checks_passed']}/{result['checks_total']}",
        f"Maximum absolute error in exact layers: {result['maximum_absolute_error_exact_layers']:.3e}",
        f"Inverse-time fitted RMS slope: {slope1:.9f} (predicted {predicted1:.9f})",
        f"Inverse-square fitted RMS slope on [1e-9,1e-3]: {slope2:.9f}",
        "Inverse-square asymptotic RMS law: Theta(sigma log(1/sigma))",
        "Power-family p->infinity limit: Theta(sigma sqrt(log(1/sigma)))",
        "",
    ]
    for layer, counts in sorted(by_layer.items()):
        lines.append(f"{layer}: {counts['passed']}/{counts['total']} PASS")
    lines.append("")
    for item in checks:
        lines.append(
            f"[{'PASS' if item.passed else 'FAIL'}] {item.layer} :: {item.name}: "
            f"value={item.value:.16g}, reference={item.reference:.16g}, "
            f"abs_err={item.absolute_error:.3e}, tol={item.tolerance:.3e}"
        )
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))
    if status != "PASS":
        for item in checks:
            if not item.passed:
                print(f"FAILED: {item.layer} :: {item.name}: "
                      f"error={item.absolute_error:.3e}, tolerance={item.tolerance:.3e}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
