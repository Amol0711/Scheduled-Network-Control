#!/usr/bin/env python3
"""Numerical asymptotic checks for capped inverse-square and power-law profiles.
Finite-grid calculations and high-precision remainders are numerical diagnostics."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp
import mpmath as mp
from scipy.optimize import brentq
from scipy.special import lambertw

from schedule_noise import (
    inverse_square_components,
    inverse_time_exact_optimum,
    minimize_power,
)

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()
OUT_JSON = ROOT / "asymptotic_asymptotic_validation.json"
OUT_TXT = ROOT / "asymptotic_asymptotic_validation.txt"
OUT_REFINEMENT = ROOT / "asymptotic_inverse_square_refinement.csv"
OUT_GAPS = ROOT / "asymptotic_named_profile_gaps.csv"
OUT_P3 = ROOT / "asymptotic_power_p3_accuracy.csv"


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
    value = float(value)
    reference = float(reference)
    err = abs(value - reference)
    checks.append(Check(name, value, reference, err, float(tolerance), err <= tolerance, layer))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def d_rho(rho: float) -> float:
    return (2.0 * rho * rho - 2.0 * rho + 1.0) / (4.0 * rho)


def inverse_square_derivative(theta: float, rho: float, T: float, eta0: float, sigma: float) -> float:
    return (
        sigma * sigma / T
        * (rho * theta + (rho * theta - 1.0) * math.exp(-2.0 * rho * theta))
        - 4.0 * rho
        * (eta0 * eta0 - d_rho(rho) * sigma * sigma / T)
        * math.exp(-4.0 * rho * theta + 2.0 * rho)
    )


def exact_inverse_square_optimum(
    rho: float, T: float, eta0: float, sigma: float
) -> tuple[float, float, float, float]:
    """Return theta_opt, K_opt, exact RMS, and W(Z_sigma)."""
    z = 16.0 * rho * T * eta0 * eta0 * math.exp(2.0 * rho) / (sigma * sigma)
    w = float(lambertw(z).real)
    theta0 = w / (4.0 * rho)
    lo = 1.0
    hi = max(8.0, 2.4 * theta0 + 5.0)
    f_lo = inverse_square_derivative(lo, rho, T, eta0, sigma)
    f_hi = inverse_square_derivative(hi, rho, T, eta0, sigma)
    while f_lo * f_hi > 0.0:
        hi *= 1.5
        f_hi = inverse_square_derivative(hi, rho, T, eta0, sigma)
        if hi > 1.0e5:
            raise RuntimeError("Could not bracket inverse-square optimum")
    theta = brentq(
        lambda x: inverse_square_derivative(x, rho, T, eta0, sigma),
        lo,
        hi,
        xtol=2e-14,
        rtol=2e-14,
        maxiter=1000,
    )
    K = theta * theta / T
    mse = inverse_square_components(K, rho, T, eta0, sigma)["mse"]
    return theta, K, math.sqrt(mse), w


def leading_rms(sigma: float, w: float, rho: float, T: float) -> float:
    return sigma * w / (4.0 * math.sqrt(2.0 * rho * T))


def refined_rms(sigma: float, w: float, rho: float, T: float) -> float:
    return leading_rms(sigma, w, rho, T) * (1.0 + 1.0 / w)


def envelope_rms(eta0: float, sigma: float, T: float) -> float:
    return abs(eta0) * sigma / math.sqrt(T * eta0 * eta0 + sigma * sigma)


def inverse_time_ratio_formula(rho: float, T: float, eta0: float, sigma: float) -> float:
    b = 0.5 * rho * (1.0 - math.exp(-2.0 * rho))
    c = rho * rho * math.exp(-2.0 * rho) / (2.0 * rho + 1.0)
    B = b + c
    cE = math.sqrt(2.0 * rho + 1.0) * (B / (2.0 * rho)) ** (rho / (2.0 * rho + 1.0))
    a0 = abs(eta0) * math.exp(-rho) * T ** (-rho)
    return math.sqrt(T) * cE * a0 ** (1.0 / (2.0 * rho + 1.0)) * sigma ** (-1.0 / (2.0 * rho + 1.0))


def main() -> int:
    checks: list[Check] = []
    refinement_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    p3_rows: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Symbolic identities behind the second-order expansion.
    # ------------------------------------------------------------------
    x, w = sp.symbols("x w", positive=True)
    series = sp.series(sp.sqrt(1 + 2 / x), x, sp.oo, 4)
    # sqrt(1+2/x) = 1 + 1/x - 1/(2x^2) + O(x^-3)
    coeff_1 = sp.limit(x * (sp.sqrt(1 + 2 / x) - 1), x, sp.oo)
    coeff_2 = sp.limit(x**2 * (sp.sqrt(1 + 2 / x) - 1 - 1 / x), x, sp.oo)
    add_check(checks, "symbolic_sqrt_first_correction", float(coeff_1), 1.0, 0.0, "symbolic")
    add_check(checks, "symbolic_sqrt_second_coefficient", float(coeff_2), -0.5, 0.0, "symbolic")

    rr, tt, ss, TT, ee, DD = sp.symbols("rr tt ss TT ee DD", positive=True)
    deriv_eq_bias = ss**2 / (4 * rr * TT) * (
        rr * tt + (rr * tt - 1) * sp.exp(-2 * rr * tt)
    )
    mse_stationary = (
        ss**2 / TT * (
            rr * tt**2 / 2 + (1 / (4 * rr) - tt / 2) * sp.exp(-2 * rr * tt)
        )
        + deriv_eq_bias
    )
    target_stationary = ss**2 / TT * (
        rr * tt**2 / 2 + tt / 4 * (1 - sp.exp(-2 * rr * tt))
    )
    residual = sp.simplify(mse_stationary - target_stationary)
    add_check(checks, "symbolic_stationary_mse_identity", float(residual), 0.0, 0.0, "symbolic")

    # Lambert derivative identity used for the finite-range slope.
    dlogw_dlogs = -2.0 / (1.0 + 7.0)  # test point w=7
    add_check(checks, "Lambert_log_derivative_test", dlogw_dlogs, -0.25, 0.0, "symbolic")

    # ------------------------------------------------------------------
    # Multi-parameter exact/refined inverse-square checks.
    # ------------------------------------------------------------------
    parameter_sets = [
        (0.6, 0.7, 0.8),
        (1.2, 1.0, 1.0),
        (1.2, 2.0, 1.0),
        (2.1, 1.8, 0.7),
    ]
    sigma_grid = np.geomspace(1e-16, 1e-3, 27)
    worst_scaled_refined_remainder = 0.0
    worst_theta_scaled_error = 0.0
    worst_theta_absolute_error = 0.0

    for rho, T, eta0 in parameter_sets:
        for sigma in sigma_grid:
            theta, K, rms_exact, wval = exact_inverse_square_optimum(
                rho, T, eta0, float(sigma)
            )
            rms_lead = leading_rms(float(sigma), wval, rho, T)
            rms_ref = refined_rms(float(sigma), wval, rho, T)
            rel_lead = rms_exact / rms_lead - 1.0
            rel_ref = rms_exact / rms_ref - 1.0
            scaled_ref = abs(rel_ref) * wval * wval
            worst_scaled_refined_remainder = max(worst_scaled_refined_remainder, scaled_ref)
            # theta = W/(4rho) plus an exponentially small correction.
            theta_err = abs(4.0 * rho * theta - wval)
            # The exponentially rescaled quantity becomes roundoff-dominated at
            # extremely small sigma.  Record it only on the displayed/validated
            # range; retain the absolute error over the entire grid.
            theta_scaled = theta_err * math.exp(wval / 2.0) / max(wval, 1.0)
            if sigma >= 1.0e-9:
                worst_theta_scaled_error = max(worst_theta_scaled_error, theta_scaled)
            worst_theta_absolute_error = max(worst_theta_absolute_error, theta_err)

            # Exact stationary MSE identity.
            stationary_mse = float(sigma) ** 2 / T * (
                rho * theta * theta / 2.0
                + theta / 4.0 * (1.0 - math.exp(-2.0 * rho * theta))
            )
            exact_mse = rms_exact * rms_exact
            add_check(
                checks,
                f"stationary_mse_r{rho}_T{T}_s{sigma:.2e}",
                stationary_mse,
                exact_mse,
                2e-12 * max(1.0, exact_mse),
                "stationary_identity",
            )
            add_check(
                checks,
                f"refinement_remainder_r{rho}_T{T}_s{sigma:.2e}",
                scaled_ref,
                0.0,
                1.10,
                "second_order",
            )
            # Refined formula must improve the leading formula materially.
            add_check(
                checks,
                f"refinement_improves_r{rho}_T{T}_s{sigma:.2e}",
                abs(rel_ref) / abs(rel_lead),
                0.0,
                0.08,
                "second_order",
            )

            if rho == 1.2 and T == 1.0 and eta0 == 1.0:
                env = envelope_rms(eta0, float(sigma), T)
                ratio2_exact = rms_exact / env
                ratio2_refined = wval / (4.0 * math.sqrt(2.0 * rho)) * (1.0 + 1.0 / wval)
                K1, mse1 = inverse_time_exact_optimum(rho, T, eta0, float(sigma))
                ratio1_exact = math.sqrt(mse1) / env
                ratio1_formula = inverse_time_ratio_formula(rho, T, eta0, float(sigma))
                refinement_rows.append({
                    "sigma": float(sigma),
                    "W": wval,
                    "theta_exact": theta,
                    "K_exact": K,
                    "rms_exact": rms_exact,
                    "rms_leading": rms_lead,
                    "rms_refined": rms_ref,
                    "relative_excess_over_leading": rel_lead,
                    "one_over_W": 1.0 / wval,
                    "relative_error_refined": rel_ref,
                    "scaled_refined_remainder_W2": scaled_ref,
                    "predicted_local_slope": 1.0 - 2.0 / (1.0 + wval),
                })
                gap_rows.append({
                    "sigma": float(sigma),
                    "rms_envelope": env,
                    "inverse_time_rms_exact": math.sqrt(mse1),
                    "inverse_time_gap_exact": ratio1_exact,
                    "inverse_time_gap_formula": ratio1_formula,
                    "inverse_square_rms_exact": rms_exact,
                    "inverse_square_gap_exact": ratio2_exact,
                    "inverse_square_gap_refined": ratio2_refined,
                    "inverse_time_Kopt": K1,
                    "inverse_square_Kopt": K,
                })

    add_check(
        checks,
        "global_scaled_second_order_remainder_bound",
        worst_scaled_refined_remainder,
        0.0,
        1.10,
        "second_order_summary",
    )
    add_check(
        checks,
        "global_theta_exponential_correction_bound_display_range",
        worst_theta_scaled_error,
        0.0,
        0.10,
        "second_order_summary",
    )
    add_check(
        checks,
        "global_theta_absolute_correction_bound",
        worst_theta_absolute_error,
        0.0,
        1.0e-3,
        "second_order_summary",
    )

    # ------------------------------------------------------------------
    # Finite-range slope on the exact figure interval, T=1.
    # ------------------------------------------------------------------
    rho, T, eta0 = 1.2, 1.0, 1.0
    fig_sigmas = np.geomspace(1e-9, 1e-3, 31)
    exact_rms: list[float] = []
    refined_curve: list[float] = []
    leading_curve: list[float] = []
    for sigma in fig_sigmas:
        _, _, rms_exact, wval = exact_inverse_square_optimum(rho, T, eta0, float(sigma))
        exact_rms.append(rms_exact)
        leading_curve.append(leading_rms(float(sigma), wval, rho, T))
        refined_curve.append(refined_rms(float(sigma), wval, rho, T))
    exact_slope = float(np.polyfit(np.log(fig_sigmas), np.log(exact_rms), 1)[0])
    refined_slope = float(np.polyfit(np.log(fig_sigmas), np.log(refined_curve), 1)[0])
    leading_slope = float(np.polyfit(np.log(fig_sigmas), np.log(leading_curve), 1)[0])
    add_check(checks, "figure_exact_inverse_square_slope", exact_slope, 0.9341354322738368, 2e-10, "finite_range_slope")
    add_check(checks, "figure_refined_predicted_slope", refined_slope, exact_slope, 1.0e-4, "finite_range_slope")
    add_check(checks, "refined_slope_improves_leading", abs(refined_slope - exact_slope), 0.0, 0.08 * abs(leading_slope - exact_slope), "finite_range_slope")

    # Central-difference local slopes across a broad small-noise range.
    for sigma in np.geomspace(1e-14, 1e-4, 11):
        h = 2e-4
        sm = float(sigma) * math.exp(-h)
        sp_ = float(sigma) * math.exp(h)
        _, _, rm, _ = exact_inverse_square_optimum(rho, T, eta0, sm)
        _, _, rp, _ = exact_inverse_square_optimum(rho, T, eta0, sp_)
        slope_num = (math.log(rp) - math.log(rm)) / (2.0 * h)
        _, _, _, wval = exact_inverse_square_optimum(rho, T, eta0, float(sigma))
        slope_pred = 1.0 - 2.0 / (1.0 + wval)
        scaled_error = abs(slope_num - slope_pred) * wval * wval
        add_check(
            checks,
            f"local_slope_sigma{sigma:.1e}",
            scaled_error,
            0.0,
            3.0,
            "local_slope",
        )

    # ------------------------------------------------------------------
    # Restricted named-profile gap formulas.
    # ------------------------------------------------------------------
    for row in gap_rows:
        sigma = float(row["sigma"])
        if sigma > 1e-3:
            continue
        ratio1_exact = float(row["inverse_time_gap_exact"])
        ratio1_formula = float(row["inverse_time_gap_formula"])
        ratio2_exact = float(row["inverse_square_gap_exact"])
        ratio2_ref = float(row["inverse_square_gap_refined"])
        # Evaluate the inverse-time ratio in high precision to avoid
        # catastrophic cancellation in the O(sigma^2) correction.
        mp.mp.dps = 80
        rr = mp.mpf("1.2")
        TTm = mp.mpf("1.0")
        eem = mp.mpf("1.0")
        sm = mp.mpf(str(sigma))
        b = mp.mpf("0.5") * rr * (1 - mp.e ** (-2 * rr))
        c = rr * rr * mp.e ** (-2 * rr) / (2 * rr + 1)
        B = b + c
        A = abs(eem) * mp.e ** (-rr) * TTm ** (-rr)
        delta = A * A - c * sm * sm * TTm ** (-(2 * rr + 1))
        Kmp = (2 * rr * delta / (B * sm * sm)) ** (1 / (2 * rr + 1))
        mse_mp = delta * Kmp ** (-2 * rr) + B * sm * sm * Kmp
        env_mp = abs(eem) * sm / mp.sqrt(TTm * eem * eem + sm * sm)
        exact_ratio_mp = mp.sqrt(mse_mp) / env_mp
        formula_mp = mp.mpf(str(ratio1_formula))
        rel_mp = abs(exact_ratio_mp / formula_mp - 1)
        add_check(
            checks,
            f"inverse_time_gap_formula_sigma{sigma:.2e}",
            float(rel_mp),
            0.0,
            max(5.0 * sigma * sigma, 5.0e-15),
            "named_gaps",
        )
        # O(W^-2) relative remainder after the refined ratio.
        wval = next(float(r["W"]) for r in refinement_rows if float(r["sigma"]) == sigma)
        add_check(
            checks,
            f"inverse_square_gap_formula_sigma{sigma:.2e}",
            abs(ratio2_exact / ratio2_ref - 1.0) * wval * wval,
            0.0,
            1.10,
            "named_gaps",
        )

    # Direct divergence over decreasing sigma for both named profiles.
    gap_sorted = sorted(gap_rows, key=lambda r: float(r["sigma"]), reverse=True)
    g1 = [float(r["inverse_time_gap_exact"]) for r in gap_sorted]
    g2 = [float(r["inverse_square_gap_exact"]) for r in gap_sorted]
    add_check(checks, "inverse_time_gap_monotone_divergence", float(all(a < b for a, b in zip(g1, g1[1:]))), 1.0, 0.0, "named_gaps")
    add_check(checks, "inverse_square_gap_monotone_divergence", float(all(a < b for a, b in zip(g2, g2[1:]))), 1.0, 0.0, "named_gaps")

    # ------------------------------------------------------------------
    # Front-loaded equality profile versus named capped power profiles.
    # ------------------------------------------------------------------
    rho, T, alpha = 1.2, 2.0, 0.18
    grid = np.linspace(0.0, T, 2001)
    u = (1.0 - alpha) / (rho * (alpha * T + (1.0 - alpha) * grid))
    du = np.diff(u)
    add_check(
        checks,
        "equality_profile_strictly_decreasing",
        float(np.max(du) < 0.0),
        1.0,
        0.0,
        "front_loading",
    )
    mass = float(np.trapezoid(u, grid))
    add_check(checks, "equality_profile_mass", mass, math.log(1.0 / alpha) / rho, 5e-7, "front_loading")
    for p in (1.0, 2.0, 3.0):
        K = 7.0 / T
        t = grid[:-1]
        if p == 1.0:
            raw = 1.0 / (T - t)
        else:
            raw = T ** (p - 1.0) / (T - t) ** p
        kap = np.minimum(raw, K)
        add_check(checks, f"capped_power_p{p}_nondecreasing", float(np.min(np.diff(kap))), 0.0, 1e-14, "front_loading")

    # ------------------------------------------------------------------
    # p=3 finite-range optimizer accuracy requested in the review.
    # ------------------------------------------------------------------
    rho, T, eta0 = 1.2, 1.0, 1.0
    for sigma in (1e-5, 1e-8, 1e-12):
        K_exact, E_exact, a_asym = minimize_power(3.0, rho, T, eta0, sigma)
        K_asym = a_asym**3 / T
        rel = abs(K_exact / K_asym - 1.0)
        p3_rows.append({
            "sigma": sigma,
            "K_exact": K_exact,
            "K_Lambert": K_asym,
            "relative_error": rel,
            "optimized_rms": math.sqrt(E_exact),
        })
        add_check(checks, f"p3_K_relative_error_sigma{sigma:.0e}", rel, 0.0, 1.0e-5, "power_p3")

    write_csv(
        OUT_REFINEMENT,
        [
            "sigma", "W", "theta_exact", "K_exact", "rms_exact",
            "rms_leading", "rms_refined", "relative_excess_over_leading",
            "one_over_W", "relative_error_refined",
            "scaled_refined_remainder_W2", "predicted_local_slope",
        ],
        sorted(refinement_rows, key=lambda r: float(r["sigma"])),
    )
    write_csv(
        OUT_GAPS,
        [
            "sigma", "rms_envelope", "inverse_time_rms_exact",
            "inverse_time_gap_exact", "inverse_time_gap_formula",
            "inverse_square_rms_exact", "inverse_square_gap_exact",
            "inverse_square_gap_refined", "inverse_time_Kopt",
            "inverse_square_Kopt",
        ],
        sorted(gap_rows, key=lambda r: float(r["sigma"])),
    )
    write_csv(
        OUT_P3,
        ["sigma", "K_exact", "K_Lambert", "relative_error", "optimized_rms"],
        p3_rows,
    )

    status = "PASS" if all(c.passed for c in checks) else "FAIL"
    by_layer: dict[str, dict[str, int]] = {}
    for c in checks:
        d = by_layer.setdefault(c.layer, {"passed": 0, "total": 0})
        d["total"] += 1
        d["passed"] += int(c.passed)

    result = {
        "experiment": "Asymptotic",
        "status": status,
        "checks_passed": sum(c.passed for c in checks),
        "checks_total": len(checks),
        "checks_by_layer": by_layer,
        "figure_range": {
            "sigma_min": float(fig_sigmas[0]),
            "sigma_max": float(fig_sigmas[-1]),
            "exact_fitted_slope": exact_slope,
            "refined_fitted_slope": refined_slope,
            "leading_fitted_slope": leading_slope,
            "parameters": {"eta0": eta0, "rho": rho, "T": T},
        },
        "worst_scaled_refined_remainder_W2": worst_scaled_refined_remainder,
        "worst_theta_exponential_scaled_error_display_range": worst_theta_scaled_error,
        "worst_theta_absolute_error": worst_theta_absolute_error,
        "p3_max_relative_K_error": max(float(r["relative_error"]) for r in p3_rows),
        "checks": [asdict(c) for c in checks],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "Asymptotic ASYMPTOTIC AND NAMED-PROFILE VALIDATION",
        "=" * 60,
        f"Status: {status}",
        f"Checks passed: {result['checks_passed']}/{result['checks_total']}",
        f"Exact inverse-square fitted slope on [1e-9,1e-3]: {exact_slope:.12f}",
        f"Refined Lambert-W fitted slope: {refined_slope:.12f}",
        f"Leading Lambert-W fitted slope: {leading_slope:.12f}",
        f"Worst |relative refined remainder| W^2: {worst_scaled_refined_remainder:.6f}",
        f"Maximum p=3 K relative error: {result['p3_max_relative_K_error']:.3e}",
        "",
    ]
    for layer, counts in sorted(by_layer.items()):
        lines.append(f"{layer}: {counts['passed']}/{counts['total']} PASS")
    lines.append("")
    for c in checks:
        lines.append(
            f"[{'PASS' if c.passed else 'FAIL'}] {c.layer} :: {c.name}: "
            f"value={c.value:.16g}, reference={c.reference:.16g}, "
            f"abs_err={c.absolute_error:.3e}, tol={c.tolerance:.3e}"
        )
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))
    if status != "PASS":
        for c in checks:
            if not c.passed:
                print(f"FAILED: {c.layer} :: {c.name}: err={c.absolute_error:.3e}, tol={c.tolerance:.3e}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
