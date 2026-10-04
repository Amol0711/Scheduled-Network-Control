#!/usr/bin/env python3
"""Numerical checks of contraction, tail, offset and bounded-drift functionals."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import expn

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()
OUT_JSON = ROOT / "schedule_profile_validation.json"
OUT_TXT = ROOT / "schedule_profile_validation.txt"


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    reference: float
    absolute_error: float
    tolerance: float
    passed: bool


def add_check(checks: list[Check], name: str, value: float, reference: float,
              tolerance: float) -> None:
    err = abs(float(value) - float(reference))
    checks.append(Check(name, float(value), float(reference), err,
                        float(tolerance), bool(err <= tolerance)))


def e1(t: float, rho: float, T: float) -> float:
    return ((T - t) / T) ** rho


def i1_tail(t: float, rho: float, T: float) -> float:
    s = (T - t) / T
    return T * s ** (rho + 1.0) / (rho + 1.0)


def e2(t: float, rho: float, T: float) -> float:
    s = (T - t) / T
    return math.exp(rho - rho / s)


def i2_tail(t: float, rho: float, T: float) -> float:
    s = (T - t) / T
    return T * math.exp(rho) * s * float(expn(2, rho / s))


def lambda_rho(s: float, rho: float) -> float:
    if abs(rho - 1.0) <= 1e-14:
        return 0.0 if s == 0.0 else s * math.log(1.0 / s)
    return abs(s - s ** rho) / abs(rho - 1.0)


def lambda1_numeric(t: float, rho: float, T: float) -> float:
    if t == 0.0:
        return 0.0
    return quad(
        lambda u: math.exp(-rho * math.log((T - u) / (T - t))),
        0.0,
        t,
        epsabs=2e-13,
        epsrel=2e-13,
        limit=300,
    )[0]


def lambda2_numeric(t: float, rho: float, T: float) -> float:
    if t == 0.0:
        return 0.0
    s = (T - t) / T
    # z=(T-u)/T gives a stable exponent -rho(1/s-1/z).
    return T * quad(
        lambda z: math.exp(-rho * (1.0 / s - 1.0 / z)),
        s,
        1.0,
        epsabs=2e-13,
        epsrel=2e-13,
        limit=400,
    )[0]


def cumulative_piecewise(t: float, T: float) -> float:
    """Integral of a discontinuous admissible profile on [0,t]."""
    jump_time = 0.43 * T
    bounded_step = 0.37 / T
    canonical = math.log(T / (T - t))
    step_area = bounded_step * max(0.0, t - jump_time)
    return canonical + step_area


def main() -> int:
    checks: list[Check] = []

    parameter_sets = [
        (0.4, 0.7),
        (1.0, 1.0),
        (1.2, 1.0),
        (2.5, 2.3),
    ]
    sample_fractions = (0.0, 0.17, 0.53, 0.91)

    # Canonical and inverse-square contraction/tail identities.
    for rho, T in parameter_sets:
        for frac in sample_fractions:
            t = frac * T
            numeric_e1 = math.exp(-rho * quad(
                lambda u: 1.0 / (T - u), 0.0, t,
                epsabs=2e-13, epsrel=2e-13, limit=300)[0])
            add_check(checks, f"E1_rho{rho}_T{T}_f{frac}", numeric_e1,
                      e1(t, rho, T), 2e-12)

            numeric_i1 = quad(lambda u: e1(u, rho, T), t, T,
                              epsabs=2e-13, epsrel=2e-13, limit=400)[0]
            add_check(checks, f"I1tail_rho{rho}_T{T}_f{frac}", numeric_i1,
                      i1_tail(t, rho, T), 3e-12)

            numeric_e2 = math.exp(-rho * quad(
                lambda u: T / (T - u) ** 2, 0.0, t,
                epsabs=2e-13, epsrel=2e-13, limit=400)[0])
            add_check(checks, f"E2_rho{rho}_T{T}_f{frac}", numeric_e2,
                      e2(t, rho, T), 3e-12)

            numeric_i2 = quad(lambda u: e2(u, rho, T), t, T,
                              epsabs=2e-13, epsrel=2e-13, limit=500)[0]
            add_check(checks, f"I2tail_rho{rho}_T{T}_f{frac}", numeric_i2,
                      i2_tail(t, rho, T), 5e-12)

    # Exact equal-initial-gain offset values used in the numerical study.
    rho = 1.2
    T = 1.0
    I1 = i1_tail(0.0, rho, T)
    I2 = i2_tail(0.0, rho, T)
    add_check(checks, "canonical_offset_rho1p2", I1,
              T / (rho + 1.0), 2e-14)
    add_check(checks, "inverse_square_offset_rho1p2", I2,
              T * math.exp(rho) * float(expn(2, rho)), 2e-14)

    # Endpoint asymptotic I2(t) ~ (T/rho)s^2 E2(t).
    asymptotic_s = 0.005
    t_asym = T * (1.0 - asymptotic_s)
    ratio = i2_tail(t_asym, rho, T) / (
        (T / rho) * asymptotic_s ** 2 * e2(t_asym, rho, T)
    )
    # At finite s the first correction is O(s); this tolerance checks the
    # direction and magnitude without pretending the asymptotic is exact.
    add_check(checks, "inverse_square_tail_asymptotic_ratio", ratio, 1.0, 0.009)

    # Canonical bounded-drift kernel and its integrated area.
    for rho_test in (0.4, 1.0, 1.2, 2.5):
        for frac in (0.09, 0.37, 0.78, 0.96):
            t = frac * T
            s = 1.0 - frac
            numerical = lambda1_numeric(t, rho_test, T)
            exact = T * lambda_rho(s, rho_test)
            add_check(checks, f"Lambda1_rho{rho_test}_f{frac}", numerical,
                      exact, 5e-12)

        area_numeric = quad(lambda tt: lambda1_numeric(tt, rho_test, T),
                            0.0, T, epsabs=2e-11, epsrel=2e-11,
                            limit=250)[0]
        area_exact = T * T / (2.0 * (rho_test + 1.0))
        add_check(checks, f"D1_rho{rho_test}", area_numeric, area_exact,
                  2e-10)

        if abs(rho_test - 1.0) <= 1e-14:
            max_exact = math.exp(-1.0)
        else:
            max_exact = rho_test ** (-rho_test / (rho_test - 1.0))
        grid = np.linspace(0.0, 1.0, 1_000_001)
        vals = np.array([lambda_rho(float(s), rho_test) for s in grid])
        max_numeric = float(vals.max())
        add_check(checks, f"Lambda1max_rho{rho_test}", max_numeric,
                  max_exact, 2e-6)

    # Admissibility-induced rejection of a bounded transverse drift.
    for profile_name, lambda_fun in (
        ("inverse_time", lambda1_numeric),
        ("inverse_square", lambda2_numeric),
    ):
        terminal_values = [lambda_fun(T * (1.0 - s), rho, T)
                           for s in (1e-2, 1e-3, 1e-4)]
        monotone_near_terminal = (
            all(terminal_values[i + 1] <= terminal_values[i]
                for i in range(len(terminal_values) - 1))
            and terminal_values[-1] < terminal_values[0]
        )
        checks.append(Check(
            f"{profile_name}_drift_kernel_terminal_decay",
            float(terminal_values[-1]),
            0.0,
            float(terminal_values[-1]),
            1e-3,
            bool(monotone_near_terminal and terminal_values[-1] < 1e-3),
        ))

    # A discontinuous, locally integrable admissible profile checks the
    # Caratheodory formulation independently of the two smooth examples.
    rho = 1.35
    T = 1.4
    jump_time = 0.43 * T
    bounded_step = 0.37 / T

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        kappa = 1.0 / (T - t) + (bounded_step if t >= jump_time else 0.0)
        return np.array([-rho * kappa * y[0]])

    y_at_jump = math.exp(-rho * cumulative_piecewise(jump_time, T))
    sol1 = solve_ivp(rhs, (0.0, jump_time), [1.0], rtol=2e-12,
                     atol=2e-14, max_step=T / 4000.0)
    sol2 = solve_ivp(rhs, (jump_time, 0.94 * T), [y_at_jump], rtol=2e-12,
                     atol=2e-14, max_step=T / 4000.0)
    numerical_terminal = float(sol2.y[0, -1])
    exact_terminal = math.exp(-rho * cumulative_piecewise(0.94 * T, T))
    add_check(checks, "caratheodory_piecewise_profile", numerical_terminal,
              exact_terminal, 5e-10)

    # The scalar sharp-displacement identity for the same discontinuous profile.
    E_piece: Callable[[float], float] = lambda t: math.exp(
        -rho * cumulative_piecewise(t, T)
    )
    displacement_quad = quad(E_piece, 0.0, T, epsabs=2e-12,
                             epsrel=2e-12, limit=500)[0]
    # Integrate y'=eta, eta'=-rho*kappa*eta on both sides of the jump.
    def rhs_pair(t: float, z: np.ndarray) -> np.ndarray:
        kappa = 1.0 / (T - t) + (bounded_step if t >= jump_time else 0.0)
        return np.array([z[1], -rho * kappa * z[1]])

    z_jump = np.array([
        quad(E_piece, 0.0, jump_time, epsabs=2e-12,
             epsrel=2e-12, limit=300)[0],
        y_at_jump,
    ])
    eps = 1e-7 * T
    sol_pair = solve_ivp(rhs_pair, (jump_time, T - eps), z_jump,
                         rtol=2e-12, atol=2e-14, max_step=T / 5000.0)
    numerical_displacement = float(sol_pair.y[0, -1]) + quad(
        E_piece, T - eps, T, epsabs=2e-14, epsrel=2e-12, limit=200)[0]
    add_check(checks, "sharp_displacement_piecewise_profile",
              numerical_displacement, displacement_quad, 2e-9)

    status = "PASS" if all(c.passed for c in checks) else "FAIL"
    result = {
        "status": status,
        "checks_total": len(checks),
        "checks_passed": sum(c.passed for c in checks),
        "parameters_for_reported_offsets": {"rho": 1.2, "T": 1.0},
        "reported_offsets": {
            "I_kappa1": i1_tail(0.0, 1.2, 1.0),
            "I_kappa2": i2_tail(0.0, 1.2, 1.0),
            "relative_reduction_I2_vs_I1": 1.0 - i2_tail(0.0, 1.2, 1.0) / i1_tail(0.0, 1.2, 1.0),
        },
        "maximum_absolute_error_among_exact_checks": max(
            c.absolute_error for c in checks
            if "asymptotic" not in c.name and "terminal_decay" not in c.name
        ),
        "checks": [asdict(c) for c in checks],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "schedule profiles SCHEDULE-PROFILE VALIDATION",
        "=" * 45,
        f"Status: {status}",
        f"Checks passed: {result['checks_passed']}/{result['checks_total']}",
        f"I_kappa1 (rho=1.2, T=1): {result['reported_offsets']['I_kappa1']:.15g}",
        f"I_kappa2 (rho=1.2, T=1): {result['reported_offsets']['I_kappa2']:.15g}",
        "Relative offset reduction (kappa2 vs kappa1): "
        f"{100.0 * result['reported_offsets']['relative_reduction_I2_vs_I1']:.9f}%",
        "",
    ]
    for c in checks:
        lines.append(
            f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: "
            f"value={c.value:.16g}, reference={c.reference:.16g}, "
            f"abs_err={c.absolute_error:.3e}, tol={c.tolerance:.3e}"
        )
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:7]))
    if status != "PASS":
        for c in checks:
            if not c.passed:
                print(f"FAILED: {c.name}: error={c.absolute_error}, tol={c.tolerance}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
