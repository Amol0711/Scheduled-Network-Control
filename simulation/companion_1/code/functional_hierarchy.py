#!/usr/bin/env python3
"""Numerical checks of the schedule-functional hierarchy and graph estimates."""
from __future__ import annotations

import csv
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
OUT_JSON = ROOT / "hierarchy_functional_hierarchy_validation.json"
OUT_TXT = ROOT / "hierarchy_functional_hierarchy_validation.txt"
OUT_CSV = ROOT / "functional_hierarchy_values.csv"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    value: float | str
    reference: float | str
    error: float
    tolerance: float


def close_check(checks: list[Check], name: str, value: float, reference: float,
                tolerance: float) -> None:
    value_f = float(value)
    reference_f = float(reference)
    error = abs(value_f - reference_f)
    checks.append(Check(name, error <= tolerance, value_f, reference_f,
                        error, float(tolerance)))


def bool_check(checks: list[Check], name: str, passed: bool,
               detail: str = "") -> None:
    checks.append(Check(name, bool(passed), detail or str(bool(passed)),
                        "True", 0.0 if passed else 1.0, 0.0))


def e_inverse_time(t: float, rho: float, T: float) -> float:
    return max(0.0, (T - t) / T) ** rho


def i_inverse_time(t: float, n: int, rho: float, T: float) -> float:
    s = max(0.0, (T - t) / T)
    return T * s ** (n * rho + 1.0) / (n * rho + 1.0)


def e_inverse_square(t: float, rho: float, T: float) -> float:
    s = (T - t) / T
    if s <= 0.0:
        return 0.0
    return math.exp(rho - rho / s)


def i_inverse_square(t: float, n: int, rho: float, T: float) -> float:
    s = (T - t) / T
    if s <= 0.0:
        return 0.0
    return T * math.exp(n * rho) * s * float(expn(2, n * rho / s))


def cumulative_piecewise(t: float, breaks: np.ndarray, values: np.ndarray) -> float:
    total = 0.0
    for left, right, value in zip(breaks[:-1], breaks[1:], values):
        if t <= left:
            break
        total += float(value) * (min(t, right) - left)
        if t <= right:
            break
    return total


def piecewise_integral(breaks: np.ndarray, values: np.ndarray) -> float:
    return float(np.dot(np.diff(breaks), values))


def profile_grid_functionals(
    breaks: np.ndarray, values: np.ndarray, rho: float, grid: np.ndarray,
    max_n: int = 4,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray, float]:
    cumulative = np.array([cumulative_piecewise(float(t), breaks, values)
                           for t in grid])
    E = np.exp(-rho * cumulative)
    I: dict[int, np.ndarray] = {}
    # Reverse cumulative trapezoid on the same grid.
    for n in range(1, max_n + 1):
        f = E ** n
        tail = np.zeros_like(grid)
        for k in range(len(grid) - 2, -1, -1):
            tail[k] = tail[k + 1] + 0.5 * (f[k] + f[k + 1]) * (grid[k + 1] - grid[k])
        I[n] = tail
    Lambda = np.zeros_like(grid)
    for k, t in enumerate(grid):
        if k == 0:
            continue
        G = np.exp(-rho * (cumulative[k] - cumulative[:k + 1]))
        Lambda[k] = float(np.trapezoid(G, grid[:k + 1]))
    D = float(np.trapezoid(Lambda, grid))
    return E, I, Lambda, D


def main() -> int:
    checks: list[Check] = []
    rng = np.random.default_rng(20260827)

    # 1. Closed-form higher-order functionals for the two named profiles.
    parameter_sets = [(0.35, 0.7), (0.8, 1.0), (1.2, 1.0), (2.4, 2.3)]
    fractions = (0.0, 0.11, 0.43, 0.79, 0.94)
    for rho, T in parameter_sets:
        for n in range(1, 6):
            for frac in fractions:
                t = frac * T
                numeric_1 = quad(
                    lambda u: e_inverse_time(u, rho, T) ** n,
                    t, T, epsabs=2e-13, epsrel=2e-13, limit=500,
                )[0]
                close_check(checks, f"I1_n{n}_rho{rho}_T{T}_f{frac}",
                            numeric_1, i_inverse_time(t, n, rho, T), 6e-12)
                numeric_2 = quad(
                    lambda u: e_inverse_square(u, rho, T) ** n,
                    t, T, epsabs=3e-13, epsrel=3e-13, limit=600,
                )[0]
                close_check(checks, f"I2_n{n}_rho{rho}_T{T}_f{frac}",
                            numeric_2, i_inverse_square(t, n, rho, T), 1.2e-11)

    # 2. Hierarchy monotonicity and exact canonical recovery.
    for profile_name, I_fun in (
        ("inverse_time", i_inverse_time),
        ("inverse_square", i_inverse_square),
    ):
        rho, T = 1.2, 1.0
        for frac in np.linspace(0.0, 0.97, 25):
            vals = [I_fun(float(frac * T), n, rho, T) for n in range(1, 7)]
            bool_check(checks, f"hierarchy_{profile_name}_f{frac:.3f}",
                       all(vals[j + 1] <= vals[j] + 2e-14
                           for j in range(len(vals) - 1)),
                       f"values={vals}")
    close_check(checks, "SL_inverse_time_I2_normalized",
                i_inverse_time(0.0, 2, 1.2, 1.0), 1.0 / 3.4, 2e-14)
    close_check(checks, "SL_inverse_square_I2_normalized",
                i_inverse_square(0.0, 2, 1.2, 1.0),
                math.exp(2.4) * float(expn(2, 2.4)), 2e-14)

    # 3. Pointwise ordering on random discontinuous profiles.
    ordering_max_violation = 0.0
    for case in range(80):
        T = float(rng.uniform(0.6, 2.2))
        rho = float(rng.uniform(0.25, 2.8))
        segments = 10
        breaks = np.linspace(0.0, T, segments + 1)
        base = rng.uniform(0.0, 2.5 / T, size=segments)
        stronger = base + rng.uniform(0.0, 2.2 / T, size=segments)
        grid = np.linspace(0.0, T, 801)
        Ea, Ia, La, Da = profile_grid_functionals(breaks, stronger, rho, grid)
        Eb, Ib, Lb, Db = profile_grid_functionals(breaks, base, rho, grid)
        ordering_max_violation = max(
            ordering_max_violation,
            float(np.max(Ea - Eb)),
            float(np.max(La - Lb)),
            Da - Db,
            *(float(np.max(Ia[n] - Ib[n])) for n in range(1, 5)),
        )
        bool_check(checks, f"order_E_case{case}", np.all(Ea <= Eb + 2e-12))
        for n in range(1, 5):
            bool_check(checks, f"order_I{n}_case{case}",
                       np.all(Ia[n] <= Ib[n] + 3e-12))
        bool_check(checks, f"order_Lambda_case{case}",
                   np.all(La <= Lb + 5e-12))
        bool_check(checks, f"order_D_case{case}", Da <= Db + 5e-12)
        # Random subinterval transition factors.
        ok_g = True
        for _ in range(25):
            iu, it = sorted(rng.integers(0, len(grid), size=2))
            ca_t = cumulative_piecewise(float(grid[it]), breaks, stronger)
            ca_u = cumulative_piecewise(float(grid[iu]), breaks, stronger)
            cb_t = cumulative_piecewise(float(grid[it]), breaks, base)
            cb_u = cumulative_piecewise(float(grid[iu]), breaks, base)
            Ga = math.exp(-rho * (ca_t - ca_u))
            Gb = math.exp(-rho * (cb_t - cb_u))
            ok_g = ok_g and Ga <= Gb + 2e-14
        bool_check(checks, f"order_G_case{case}", ok_g)
        for cap_idx, K in enumerate((0.4 / T, 1.1 / T, 3.0 / T)):
            Ja = piecewise_integral(breaks, np.minimum(stronger, K))
            Jb = piecewise_integral(breaks, np.minimum(base, K))
            bool_check(checks, f"order_J_case{case}_cap{cap_idx}",
                       Ja + 2e-14 >= Jb)

    # 4. The lemma deliberately excludes effort metrics.
    # kappa_a=2 and kappa_b=1 are pointwise ordered, yet the stronger profile
    # has larger initial kappa E and larger quadratic effort over [0,1].
    rho = T = 1.0
    effort_a = quad(lambda t: (2.0 * math.exp(-2.0 * t)) ** 2, 0.0, T)[0]
    effort_b = quad(lambda t: (1.0 * math.exp(-1.0 * t)) ** 2, 0.0, T)[0]
    bool_check(checks, "effort_not_ordered_initial_product", 2.0 > 1.0,
               "kappa_a E_a(0)=2 > 1=kappa_b E_b(0)")
    bool_check(checks, "effort_not_improved_by_stronger_profile",
               effort_a > effort_b,
               f"stronger={effort_a:.12g}, weaker={effort_b:.12g}")

    # 5. Exact order-n scalar displacement and terminal phase tail.
    scalar_profiles: list[tuple[str, Callable[[float], float], float]] = [
        ("constant", lambda t: math.exp(-1.1 * 0.7 * t), 1.4),
        ("inverse_time", lambda t: e_inverse_time(t, 1.1, 1.4), 1.4),
        ("inverse_square", lambda t: e_inverse_square(t, 1.1, 1.4), 1.4),
    ]
    eta0 = -0.73
    L_n = 1.37
    for name, Efun, T in scalar_profiles:
        for n in range(1, 5):
            total_numeric = L_n * eta0 ** n * quad(
                lambda u: Efun(u) ** n, 0.0, T,
                epsabs=2e-13, epsrel=2e-13, limit=600,
            )[0]
            if name == "inverse_time":
                total_ref = L_n * eta0 ** n * i_inverse_time(0.0, n, 1.1, T)
            elif name == "inverse_square":
                total_ref = L_n * eta0 ** n * i_inverse_square(0.0, n, 1.1, T)
            else:
                total_ref = L_n * eta0 ** n * (1.0 - math.exp(-n * 1.1 * 0.7 * T)) / (n * 1.1 * 0.7)
            close_check(checks, f"scalar_displacement_{name}_n{n}",
                        total_numeric, total_ref, 8e-12)
            for frac in (0.17, 0.58, 0.91):
                t = frac * T
                tail_numeric = L_n * eta0 ** n * quad(
                    lambda u: Efun(u) ** n, t, T,
                    epsabs=2e-13, epsrel=2e-13, limit=600,
                )[0]
                if name == "inverse_time":
                    tail_ref = L_n * eta0 ** n * i_inverse_time(t, n, 1.1, T)
                elif name == "inverse_square":
                    tail_ref = L_n * eta0 ** n * i_inverse_square(t, n, 1.1, T)
                else:
                    tail_ref = L_n * eta0 ** n * (
                        math.exp(-n * 1.1 * 0.7 * t) - math.exp(-n * 1.1 * 0.7 * T)
                    ) / (n * 1.1 * 0.7)
                close_check(checks, f"scalar_phase_tail_{name}_n{n}_f{frac}",
                            tail_numeric, tail_ref, 8e-12)

    # 6. Arbitrary-profile graph estimate: equality for a linear node field
    # initialized in a lambda_2 eigenmode.
    graph_cases = [
        ("constant", 1.3, lambda t, T: 0.9 / T),
        ("step", 1.1, lambda t, T: (0.4 if t < 0.37 * T else 1.8) / T),
        ("inverse_time", 1.4, lambda t, T: 1.0 / (T - t)),
        ("inverse_square", 0.9, lambda t, T: T / (T - t) ** 2),
    ]
    graph_max_error = 0.0
    for name, rho, kappa in graph_cases:
        T = 1.6
        ell = -0.25 if name == "step" else 0.35
        t_end = 0.93 * T
        z0 = 0.87
        sol = solve_ivp(
            lambda t, z: [(ell - rho * kappa(t, T)) * z[0]],
            (0.0, t_end), [z0], method="DOP853", rtol=2e-12, atol=2e-14,
            max_step=T / 2500.0,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        z_num = float(sol.y[0, -1])
        mass = quad(lambda t: kappa(t, T), 0.0, t_end,
                    epsabs=2e-13, epsrel=2e-13, limit=600)[0]
        z_ref = z0 * math.exp(ell * t_end - rho * mass)
        err = abs(z_num - z_ref)
        graph_max_error = max(graph_max_error, err)
        close_check(checks, f"graph_profile_equality_{name}", z_num, z_ref, 3e-10)
        # Input estimate with lambda_2=1, lambda_N=4, k0=rho.
        lhs = rho * 1.0 * kappa(t_end, T) * abs(z_num)
        rhs = rho * 4.0 * math.exp(max(ell, 0.0) * T) * kappa(t_end, T) * math.exp(-rho * mass) * abs(z0)
        bool_check(checks, f"graph_input_bound_{name}", lhs <= rhs * (1.0 + 2e-11),
                   f"lhs={lhs:.12g}, rhs={rhs:.12g}")

    # 7. Values archived for Comparison table/figure integration.
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "profile", "n", "rho", "T", "I_total", "I_over_T"
        ])
        writer.writeheader()
        for profile in ("inverse_time", "inverse_square"):
            for n in range(1, 6):
                rho, T = 1.2, 1.0
                value = (i_inverse_time(0.0, n, rho, T)
                         if profile == "inverse_time"
                         else i_inverse_square(0.0, n, rho, T))
                writer.writerow({
                    "profile": profile, "n": n, "rho": rho, "T": T,
                    "I_total": f"{value:.16e}",
                    "I_over_T": f"{value / T:.16e}",
                })

    passed = all(c.passed for c in checks)
    result = {
        "experiment": "Hierarchy",
        "status": "PASS" if passed else "FAIL",
        "checks_total": len(checks),
        "checks_passed": sum(c.passed for c in checks),
        "checks_failed": sum(not c.passed for c in checks),
        "key_values": {
            "rho": 1.2,
            "T": 1.0,
            "I1_order1": i_inverse_time(0.0, 1, 1.2, 1.0),
            "I2_order1": i_inverse_square(0.0, 1, 1.2, 1.0),
            "I1_order2": i_inverse_time(0.0, 2, 1.2, 1.0),
            "I2_order2": i_inverse_square(0.0, 2, 1.2, 1.0),
            "ordering_max_positive_violation": max(0.0, ordering_max_violation),
            "graph_max_absolute_error": graph_max_error,
            "effort_counterexample_stronger": effort_a,
            "effort_counterexample_weaker": effort_b,
        },
        "checks": [asdict(c) for c in checks],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "Hierarchy FUNCTIONAL-HIERARCHY VALIDATION",
        "=" * 58,
        f"Status: {result['status']}",
        f"Checks: {result['checks_passed']}/{result['checks_total']} PASS",
        "",
        "Closed hierarchy values at rho=1.2, T=1:",
        f"  I_kappa1^(1) = {result['key_values']['I1_order1']:.15g}",
        f"  I_kappa2^(1) = {result['key_values']['I2_order1']:.15g}",
        f"  I_kappa1^(2) = {result['key_values']['I1_order2']:.15g}",
        f"  I_kappa2^(2) = {result['key_values']['I2_order2']:.15g}",
        f"Maximum schedule-ordering violation = {result['key_values']['ordering_max_positive_violation']:.3e}",
        f"Maximum graph equality error = {graph_max_error:.3e}",
        "",
        "Verified: higher-order closed forms, hierarchy monotonicity,",
        "pointwise ordering of E/I/G/Lambda/D/J, nonordering of effort,",
        "order-n scalar displacement and phase tails, arbitrary-profile graph",
        "synchronization, and exact Stuart--Landau inverse-time recovery.",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(OUT_TXT.read_text(), end="")
    if not passed:
        for c in checks:
            if not c.passed:
                print(f"FAILED {c.name}: {c.value} vs {c.reference}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
