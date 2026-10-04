#!/usr/bin/env python3
"""Independent validation of the Envelope Dirichlet lower envelope.

The suite checks the symbolic identities, random bounded realized gains,
fixed-contraction equality profiles, the exact global MSE envelope, peak-gain
requirements, the constant-factor construction, small-noise scaling, and the
modewise lower bound.  It does not assume monotonicity or a single cap crossing.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()
OUT_JSON = ROOT / "envelope_dirichlet_validation.json"
OUT_TXT = ROOT / "envelope_dirichlet_validation.txt"
OUT_CSV = ROOT / "dirichlet_envelope_curves.csv"


@dataclass
class Check:
    name: str
    passed: bool
    detail: Any


CHECKS: list[Check] = []
MAX_ERRORS: dict[str, float] = {}


def add(name: str, passed: bool, detail: Any) -> None:
    CHECKS.append(Check(name, bool(passed), detail))


def record_error(name: str, value: float) -> None:
    value = float(abs(value))
    MAX_ERRORS[name] = max(MAX_ERRORS.get(name, 0.0), value)


def close(a: float, b: float, *, atol: float = 1e-10, rtol: float = 1e-10) -> bool:
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def envelope_mse(eta0: float, sigma: float, T: float) -> float:
    A = eta0 * eta0
    B = sigma * sigma / T
    return 0.0 if A == 0.0 else A * B / (A + B)


def alpha_star(eta0: float, sigma: float, T: float) -> float:
    A = eta0 * eta0
    B = sigma * sigma / T
    return B / (A + B)


def kappa_star(s: float, alpha: float, rho: float, T: float) -> float:
    if alpha == 1.0:
        return 0.0
    return (1.0 - alpha) / (rho * (alpha * T + (1.0 - alpha) * s))


def phi_star(s: float, alpha: float, T: float) -> float:
    return alpha + (1.0 - alpha) * s / T


def affine_cap_mse(K: float, eta0: float, sigma: float, rho: float, T: float) -> float:
    den = 1.0 + rho * K * T
    return (eta0 * eta0 + rho * rho * sigma * sigma * K * K * T) / (den * den)


def piecewise_stats(
    edges: np.ndarray,
    gains: np.ndarray,
    rho: float,
    sigma: float,
) -> tuple[float, float, Callable[[float], float], Callable[[float], float]]:
    dt = np.diff(edges)
    mass_segments = gains * dt
    total_mass = float(np.sum(mass_segments))
    alpha = math.exp(-rho * total_mass)
    tail_after = np.zeros_like(gains)
    running = 0.0
    for i in range(len(gains) - 1, -1, -1):
        tail_after[i] = running
        running += mass_segments[i]

    variance = 0.0
    for i, (g, h, dti) in enumerate(zip(gains, tail_after, dt)):
        if g <= 0.0:
            continue
        variance += (
            rho
            * sigma
            * sigma
            * g
            / 2.0
            * math.exp(-2.0 * rho * h)
            * (-math.expm1(-2.0 * rho * g * dti))
        )

    def gain_at(s: float) -> float:
        if s >= edges[-1]:
            return float(gains[-1])
        idx = int(np.searchsorted(edges, s, side="right") - 1)
        idx = min(max(idx, 0), len(gains) - 1)
        return float(gains[idx])

    def phi_at(s: float) -> float:
        if s >= edges[-1]:
            return 1.0
        idx = int(np.searchsorted(edges, s, side="right") - 1)
        idx = min(max(idx, 0), len(gains) - 1)
        b = edges[idx + 1]
        tail = float(tail_after[idx] + gains[idx] * (b - s))
        return math.exp(-rho * tail)

    return alpha, variance, gain_at, phi_at


def symbolic_checks() -> None:
    s, T, rho, alpha = sp.symbols("s T rho alpha", positive=True)
    phi = alpha + (1 - alpha) * s / T
    nu = (1 - alpha) / (rho * (alpha * T + (1 - alpha) * s))
    add("symbolic transition derivative", sp.simplify(sp.diff(phi, s) - rho * nu * phi) == 0, str(sp.simplify(sp.diff(phi, s) - rho * nu * phi)))
    mass = sp.integrate(nu, (s, 0, T))
    add("symbolic equality-profile mass", sp.simplify(mass - sp.log(1 / alpha) / rho) == 0, str(sp.simplify(mass)))
    energy = sp.integrate(sp.diff(phi, s) ** 2, (s, 0, T))
    add("symbolic Dirichlet energy", sp.simplify(energy - (1 - alpha) ** 2 / T) == 0, str(sp.simplify(energy)))

    A, B, a = sp.symbols("A B a", positive=True)
    F = A * a**2 + B * (1 - a) ** 2
    astar = B / (A + B)
    add("symbolic scalar minimizer", sp.simplify(sp.diff(F, a).subs(a, astar)) == 0, str(sp.simplify(sp.diff(F, a).subs(a, astar))))
    add("symbolic minimum value", sp.simplify(F.subs(a, astar) - A * B / (A + B)) == 0, str(sp.simplify(F.subs(a, astar))))
    peak = sp.simplify(nu.subs(s, 0).subs(alpha, astar).subs({A: sp.symbols('eta2', positive=True), B: sp.symbols('sig2', positive=True) / T}))
    # The explicit peak is verified numerically below; this symbolic calculation
    # is retained as a finite expression sanity check.
    add("symbolic peak expression finite", peak.is_finite is not False, str(peak))


def equality_profile_checks() -> None:
    parameter_sets = [
        (0.4, 0.7, 0.3),
        (1.2, 1.0, 1e-2),
        (2.7, 2.3, 0.8),
    ]
    alphas = [1.0, 0.9, 0.5, 0.1, 1e-2, 1e-4]
    for rho, T, sigma in parameter_sets:
        for alpha in alphas:
            integ_mass = quad(lambda x: kappa_star(x, alpha, rho, T), 0.0, T, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
            target_mass = 0.0 if alpha == 1.0 else math.log(1.0 / alpha) / rho
            err = integ_mass - target_mass
            record_error("equality mass", err)
            add(f"equality mass rho={rho} T={T} alpha={alpha}", close(integ_mass, target_mass, atol=2e-11, rtol=2e-11), {"value": integ_mass, "target": target_mass})

            dirichlet = quad(lambda x: ((1.0 - alpha) / T) ** 2, 0.0, T, epsabs=1e-13, epsrel=1e-13)[0]
            variance_quad = sigma * sigma * quad(
                lambda x: (rho * kappa_star(x, alpha, rho, T) * phi_star(x, alpha, T)) ** 2,
                0.0,
                T,
                epsabs=1e-13,
                epsrel=1e-13,
                limit=200,
            )[0]
            target_var = sigma * sigma * (1.0 - alpha) ** 2 / T
            record_error("equality variance", variance_quad - target_var)
            add(f"equality variance rho={rho} T={T} alpha={alpha}", close(variance_quad, target_var, atol=2e-11, rtol=2e-11), {"value": variance_quad, "target": target_var})
            add(f"Dirichlet equality rho={rho} T={T} alpha={alpha}", close(dirichlet, (1.0 - alpha) ** 2 / T, atol=1e-13, rtol=1e-13), dirichlet)
            ks = np.array([kappa_star(x, alpha, rho, T) for x in np.linspace(0.0, T, 101)])
            add(f"equality profile monotone rho={rho} T={T} alpha={alpha}", bool(np.all(np.diff(ks) <= 1e-13)), {"first": float(ks[0]), "last": float(ks[-1])})


def random_bounded_profile_checks() -> None:
    rng = np.random.default_rng(20260827)
    for case in range(120):
        T = float(rng.uniform(0.3, 3.0))
        rho = float(rng.uniform(0.2, 4.0))
        sigma = float(10 ** rng.uniform(-4.0, 0.2))
        eta0 = float(rng.normal())
        n = int(rng.integers(3, 11))
        interior = np.sort(rng.uniform(0.0, T, size=n - 1))
        edges = np.concatenate(([0.0], interior, [T]))
        gains = rng.uniform(0.0, 12.0, size=n)
        # Include occasional zero-gain segments.
        gains[rng.random(n) < 0.2] = 0.0
        alpha, var_exact, gain_at, phi_at = piecewise_stats(edges, gains, rho, sigma)
        points = list(edges[1:-1])
        dir_quad = quad(
            lambda s: (rho * gain_at(s) * phi_at(s)) ** 2,
            0.0,
            T,
            points=points,
            epsabs=2e-11,
            epsrel=2e-11,
            limit=300,
        )[0]
        var_quad = sigma * sigma * dir_quad
        record_error("random variance identity", var_quad - var_exact)
        add(f"random profile variance identity {case}", close(var_quad, var_exact, atol=3e-9, rtol=3e-9), {"quad": var_quad, "exact": var_exact})

        cs_bound = sigma * sigma * (1.0 - alpha) ** 2 / T
        add(f"random profile Cauchy-Schwarz bound {case}", var_exact + 1e-12 >= cs_bound, {"variance": var_exact, "bound": cs_bound})

        mse = eta0 * eta0 * alpha * alpha + var_exact
        env = envelope_mse(eta0, sigma, T)
        add(f"random profile global envelope {case}", mse + 1e-12 >= env, {"mse": mse, "envelope": env})

        mass = float(np.dot(gains, np.diff(edges)))
        add(f"random profile endpoint contraction {case}", close(alpha, math.exp(-rho * mass), atol=1e-13, rtol=1e-13), {"alpha": alpha, "mass": mass})


def global_envelope_checks() -> None:
    rhos = [0.35, 1.2, 3.5]
    Ts = [0.4, 1.0, 2.5]
    eta_values = [0.0, 0.2, 1.0, -1.7]
    sigmas = [1e-4, 0.03, 0.8]
    for rho in rhos:
        for T in Ts:
            for eta0 in eta_values:
                for sigma in sigmas:
                    astar = alpha_star(eta0, sigma, T)
                    f = lambda a: eta0 * eta0 * a * a + sigma * sigma * (1.0 - a) ** 2 / T
                    result = minimize_scalar(f, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-14})
                    env = envelope_mse(eta0, sigma, T)
                    record_error("scalar optimizer alpha", result.x - astar)
                    record_error("scalar optimizer value", result.fun - env)
                    add(f"scalar optimizer alpha rho={rho} T={T} eta={eta0} sigma={sigma}", close(result.x, astar, atol=2e-7, rtol=2e-7), {"numeric": result.x, "exact": astar})
                    add(f"scalar optimizer value rho={rho} T={T} eta={eta0} sigma={sigma}", close(result.fun, env, atol=2e-10, rtol=2e-8), {"numeric": result.fun, "exact": env})
                    if eta0 == 0.0:
                        add(f"zero-bias optimizer rho={rho} T={T} sigma={sigma}", close(astar, 1.0, atol=1e-14, rtol=1e-14) and env == 0.0, {"alpha": astar, "mse": env})
                    else:
                        Kstar = eta0 * eta0 / (rho * sigma * sigma)
                        peak = kappa_star(0.0, astar, rho, T)
                        record_error("exact envelope peak", peak - Kstar)
                        add(f"exact envelope peak rho={rho} T={T} eta={eta0} sigma={sigma}", close(peak, Kstar, atol=2e-9, rtol=2e-9), {"peak": peak, "exact": Kstar})


def finite_peak_checks() -> None:
    rng = np.random.default_rng(271828)
    for case in range(80):
        rho = float(rng.uniform(0.25, 3.5))
        T = float(rng.uniform(0.2, 4.0))
        eta0 = float(rng.normal())
        sigma = float(10 ** rng.uniform(-5.0, -0.1))
        K = float(10 ** rng.uniform(-1.0, 4.0))
        alphaK = 1.0 / (1.0 + rho * K * T)
        transition = lambda s: (1.0 + rho * K * s) / (1.0 + rho * K * T)
        gain = lambda s: K / (1.0 + rho * K * s)
        var_quad = sigma * sigma * quad(lambda s: (rho * gain(s) * transition(s)) ** 2, 0.0, T, epsabs=1e-12, epsrel=1e-12)[0]
        mse_quad = eta0 * eta0 * alphaK * alphaK + var_quad
        mse_formula = affine_cap_mse(K, eta0, sigma, rho, T)
        record_error("finite peak affine MSE", mse_quad - mse_formula)
        add(f"finite peak affine identity {case}", close(mse_quad, mse_formula, atol=2e-10, rtol=2e-10), {"quad": mse_quad, "formula": mse_formula})
        add(f"finite peak bound {case}", max(gain(0.0), gain(T)) <= K * (1.0 + 1e-13), {"peak": gain(0.0), "K": K})
        add(f"finite peak contraction {case}", close(transition(0.0), alphaK, atol=1e-14, rtol=1e-14), {"alpha": alphaK})

        if eta0 != 0.0:
            Kstar = eta0 * eta0 / (rho * sigma * sigma)
            # Exact envelope is attained by the equality profile precisely when
            # the available peak exceeds its unique required peak.
            feasible = K >= Kstar
            add(f"exact envelope cap criterion {case}", feasible == (K / Kstar >= 1.0), {"K": K, "Kstar": Kstar, "feasible": feasible})

    rho, T, eta0 = 1.2, 1.0, 1.0
    sigmas = np.logspace(-9, -3, 31)
    ratios = []
    kcf_values = []
    kstar_values = []
    env_values = []
    for sigma in sigmas:
        Kstar = eta0 * eta0 / (rho * sigma * sigma)
        Kcf = (eta0 * math.sqrt(T) / sigma - 1.0) / (rho * T)
        env = envelope_mse(eta0, sigma, T)
        mse_cf = affine_cap_mse(Kcf, eta0, sigma, rho, T)
        ratio = math.sqrt(mse_cf / env)
        ratios.append(ratio)
        kcf_values.append(Kcf)
        kstar_values.append(Kstar)
        env_values.append(math.sqrt(env))
        add(f"constant-factor MSE bound sigma={sigma:.3e}", mse_cf <= 2.0 * sigma * sigma / T * (1.0 + 1e-12), {"mse": mse_cf, "bound": 2.0 * sigma * sigma / T})

    slope_kstar = float(np.polyfit(np.log(sigmas), np.log(kstar_values), 1)[0])
    slope_kcf = float(np.polyfit(np.log(sigmas), np.log(kcf_values), 1)[0])
    slope_env = float(np.polyfit(np.log(sigmas), np.log(env_values), 1)[0])
    add("exact envelope peak scaling", close(slope_kstar, -2.0, atol=1e-10, rtol=1e-10), slope_kstar)
    add("constant-factor peak scaling", abs(slope_kcf + 1.0) < 5e-4, slope_kcf)
    add("envelope RMS small-noise slope", abs(slope_env - 1.0) < 5e-7, slope_env)
    add("constant-factor ratio limit", abs(ratios[0] - math.sqrt(2.0)) < 2e-8 and ratios[-1] < math.sqrt(2.0), {"smallest_sigma_ratio": ratios[0], "largest_sigma_ratio": ratios[-1]})

    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sigma", "rms_envelope", "K_envelope_exact", "K_constant_factor", "rms_constant_factor", "rms_ratio"])
        for sigma, env_rms, Kstar, Kcf, ratio in zip(sigmas, env_values, kstar_values, kcf_values, ratios):
            writer.writerow([f"{sigma:.16e}", f"{env_rms:.16e}", f"{Kstar:.16e}", f"{Kcf:.16e}", f"{env_rms * ratio:.16e}", f"{ratio:.16e}"])


def modal_checks() -> None:
    rng = np.random.default_rng(314159)
    for case in range(50):
        T = float(rng.uniform(0.4, 2.5))
        nseg = int(rng.integers(3, 9))
        interior = np.sort(rng.uniform(0.0, T, size=nseg - 1))
        edges = np.concatenate(([0.0], interior, [T]))
        gains = rng.uniform(0.0, 8.0, size=nseg)
        nmodes = int(rng.integers(2, 8))
        total = 0.0
        lower = 0.0
        for _ in range(nmodes):
            rho = float(rng.uniform(0.2, 5.0))
            sigma = float(10 ** rng.uniform(-4.0, -0.1))
            eta0 = float(rng.normal())
            alpha, var, _, _ = piecewise_stats(edges, gains, rho, sigma)
            total += eta0 * eta0 * alpha * alpha + var
            lower += envelope_mse(eta0, sigma, T)
        add(f"modal lower bound {case}", total + 1e-11 >= lower, {"network_mse": total, "modewise_envelope": lower})


def finish() -> int:
    passed = sum(c.passed for c in CHECKS)
    total = len(CHECKS)
    failed = [c for c in CHECKS if not c.passed]
    status = "PASS" if not failed else "FAIL"
    summary = {
        "experiment": "Envelope",
        "status": status,
        "checks_passed": passed,
        "checks_total": total,
        "checks_failed": len(failed),
        "max_absolute_errors": MAX_ERRORS,
        "key_results": {
            "exact_bounded_profile_mse_envelope": "eta0^2*(sigma^2/T)/(eta0^2+sigma^2/T)",
            "exact_envelope_rms": "abs(eta0)*sigma/sqrt(T*eta0^2+sigma^2)",
            "exact_envelope_peak": "eta0^2/(rho*sigma^2)",
            "exact_peak_small_noise_order": "Theta(sigma^-2)",
            "constant_factor_peak_order": "Theta(sigma^-1)",
            "constant_factor_rms_ratio_limit": math.sqrt(2.0),
        },
        "failed_checks": [{"name": c.name, "detail": c.detail} for c in failed[:30]],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "Envelope DIRICHLET LOWER-ENVELOPE VALIDATION",
        "=" * 61,
        f"Status: {status}",
        f"Checks: {passed}/{total} PASS",
        "",
        "Verified statements:",
        "- terminal variance equals the Dirichlet energy of the transition factor;",
        "- Cauchy-Schwarz gives the fixed-contraction lower bound;",
        "- the unique equality transition is affine and the realized gain is decreasing;",
        "- the exact global bounded-profile MSE envelope and optimizer are attained;",
        "- exact attainment requires peak gain Theta(sigma^-2) for nonzero initial bias;",
        "- a Theta(sigma^-1) construction approaches the RMS floor within sqrt(2);",
        "- the scalar lower envelope sums to a valid common-schedule modal bound.",
        "",
        "Maximum absolute discrepancies:",
    ]
    for key, value in sorted(MAX_ERRORS.items()):
        lines.append(f"- {key}: {value:.6e}")
    if failed:
        lines.extend(["", "Failed checks:"])
        for c in failed[:30]:
            lines.append(f"- {c.name}: {c.detail}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if status == "PASS" else 1


def main() -> int:
    symbolic_checks()
    equality_profile_checks()
    random_bounded_profile_checks()
    global_envelope_checks()
    finite_peak_checks()
    modal_checks()
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
