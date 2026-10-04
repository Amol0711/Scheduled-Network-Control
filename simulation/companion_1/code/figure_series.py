#!/usr/bin/env python3
"""Generate numerical series for schedule and network comparisons.

Reintegrates deterministic ODE trajectories and evaluates analytic curves;
uses the supplied stochastic terminal statistics without new random paths."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import lambertw

from network_experiments import (
    EPS_T,
    LAM_RING,
    RHO_STAR,
    T,
    V_RING,
    initial_condition,
    modal_terminal_components,
    simulate_deterministic,
    solve_reduced,
)

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()
DATA = output_dir() / "figure_series"
DATA.mkdir(parents=True, exist_ok=True)
PLOT_FLOOR = 1.0e-14


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    summary = json.loads((ROOT / "numerical_summary.json").read_text())
    k_opt = float(summary["Kopt_network_modal"])
    tangent_slope = float(summary["quadratic_tangential_offset_slope"])
    phase_map = summary["phase_map_validation"]

    # Panels (a)--(b): deterministic time histories at a common verified
    # snapshot.  Full-clock phase is supplied by the continuous accumulator.
    x0 = initial_condition()
    t_eval = np.linspace(0.0, T - EPS_T, 4001)
    ybar = solve_reduced(x0.mean(axis=0), t_eval)
    native_phase = np.unwrap(np.arctan2(ybar[:, 1], ybar[:, 0]))

    methods = {
        "constant": None,
        "capped": k_opt,
        "fiber_ptc": None,
        "xu_liu_l2": None,
        "full_clock": None,
    }
    sims = {
        method: simulate_deterministic(
            method, x0, t_eval, cap=cap, compute_energy=False
        )
        for method, cap in methods.items()
    }
    phases = {
        method: np.asarray(sims[method]["phase"])
        for method in ["fiber_ptc", "xu_liu_l2", "full_clock"]
    }

    # Retain every fourth point and a dense terminal layer.
    plot_indices = sorted(
        set(range(0, len(t_eval), 4))
        | set(range(max(0, len(t_eval) - 81), len(t_eval)))
    )
    history_rows: list[dict[str, object]] = []
    for idx in plot_indices:
        t = t_eval[idx]
        history_rows.append(
            {
                "normalized_time": t / T,
                "constant_disagreement": max(
                    float(np.asarray(sims["constant"]["disagreement"])[idx]),
                    PLOT_FLOOR,
                ),
                "capped_disagreement": max(
                    float(np.asarray(sims["capped"]["disagreement"])[idx]),
                    PLOT_FLOOR,
                ),
                "fiber_disagreement": max(
                    float(np.asarray(sims["fiber_ptc"]["disagreement"])[idx]),
                    PLOT_FLOOR,
                ),
                "xu_liu_disagreement": max(
                    float(np.asarray(sims["xu_liu_l2"]["disagreement"])[idx]),
                    PLOT_FLOOR,
                ),
                "native_phase": float(native_phase[idx]),
                "fiber_phase": float(phases["fiber_ptc"][idx]),
                "xu_liu_phase": float(phases["xu_liu_l2"][idx]),
                "full_clock_phase": float(phases["full_clock"][idx]),
            }
        )
    write_rows(
        DATA / "deterministic_timeseries.csv",
        [
            "normalized_time",
            "constant_disagreement",
            "capped_disagreement",
            "fiber_disagreement",
            "xu_liu_disagreement",
            "native_phase",
            "fiber_phase",
            "xu_liu_phase",
            "full_clock_phase",
        ],
        history_rows,
    )

    # Panel (c): graph-rate identity.
    exponent_rows = read_rows(ROOT / "exponent_validation.csv")
    for graph in ["path", "ring", "complete"]:
        rows = [
            {
                "predicted_rho": float(row["predicted_rho"]),
                "fitted_rho": float(row["fitted_rho"]),
            }
            for row in exponent_rows
            if row["graph"] == graph
        ]
        write_rows(
            DATA / f"exponent_{graph}.csv",
            ["predicted_rho", "fitted_rho"],
            rows,
        )

    # Panel (d): quadratic projection-orbit displacement.
    tangent_rows_in = read_rows(ROOT / "tangential_scaling.csv")
    eps0 = float(tangent_rows_in[0]["initial_disagreement_scale"])
    err0 = float(tangent_rows_in[0]["terminal_projection_orbit_error"])
    tangent_rows = []
    for row in tangent_rows_in:
        eps = float(row["initial_disagreement_scale"])
        tangent_rows.append(
            {
                "initial_disagreement_scale": eps,
                "terminal_projection_orbit_error": float(
                    row["terminal_projection_orbit_error"]
                ),
                "quadratic_reference": err0 * (eps / eps0) ** 2,
            }
        )
    write_rows(
        DATA / "tangential_scaling.csv",
        [
            "initial_disagreement_scale",
            "terminal_projection_orbit_error",
            "quadratic_reference",
        ],
        tangent_rows,
    )

    # Panel (e): optimized inverse-time versus inverse-square noise floors.
    optimum_rows = read_rows(ROOT / "schedule_noise_optima.csv")
    inverse_time = {
        float(row["sigma"]): row
        for row in optimum_rows
        if row["schedule"] == "inverse_time"
    }
    inverse_square = {
        float(row["sigma"]): row
        for row in optimum_rows
        if row["schedule"] == "inverse_square"
    }
    envelope_rows = read_rows(ROOT / "dirichlet_envelope_curves.csv")
    envelope = {
        round(float(row["sigma"]), 15): float(row["rms_envelope"])
        for row in envelope_rows
    }
    sigma_values = sorted(set(inverse_time) & set(inverse_square))
    sigma0 = sigma_values[0]
    rms0 = float(inverse_time[sigma0]["optimized_rms"])
    exponent_1 = 2.0 * RHO_STAR / (2.0 * RHO_STAR + 1.0)
    optimized_rows = []
    # The scalar schedule-noise panel uses (eta0,rho,T)=(1,1.2,1).
    scalar_eta0 = 1.0
    scalar_T = 1.0
    for sigma in sigma_values:
        z_sigma = (
            16.0 * RHO_STAR * scalar_T * scalar_eta0**2
            * math.exp(2.0 * RHO_STAR) / sigma**2
        )
        w_sigma = float(lambertw(z_sigma).real)
        leading_inverse_square = (
            sigma * w_sigma / (4.0 * math.sqrt(2.0 * RHO_STAR * scalar_T))
        )
        refined_inverse_square = leading_inverse_square * (1.0 + 1.0 / w_sigma)
        optimized_rows.append(
            {
                "sigma": sigma,
                "inverse_time_rms": float(inverse_time[sigma]["optimized_rms"]),
                "inverse_square_rms": float(inverse_square[sigma]["optimized_rms"]),
                "bounded_profile_envelope_rms": envelope[round(sigma, 15)],
                "inverse_time_power_reference": rms0 * (sigma / sigma0) ** exponent_1,
                "inverse_square_asymptotic_rms": leading_inverse_square,
                "inverse_square_refined_rms": refined_inverse_square,
                "inverse_square_W": w_sigma,
                "inverse_square_local_slope": 1.0 - 2.0 / (1.0 + w_sigma),
            }
        )
    write_rows(
        DATA / "optimized_noise_scaling.csv",
        [
            "sigma",
            "inverse_time_rms",
            "inverse_square_rms",
            "bounded_profile_envelope_rms",
            "inverse_time_power_reference",
            "inverse_square_asymptotic_rms",
            "inverse_square_refined_rms",
            "inverse_square_W",
            "inverse_square_local_slope",
        ],
        optimized_rows,
    )

    # Panel (f): direct phase-map validation.
    phase_rows_in = read_rows(ROOT / "phase_map_validation.csv")
    phase_rows = [
        {
            "s": float(row["s"]),
            "scalar_phase_tail": float(row["scalar_phase_tail"]),
            "stuart_landau_phase_tail": max(
                float(row["stuart_landau_phase_tail"]), 1.0e-15
            ),
            "stuart_landau_s3p4_reference": float(
                row["stuart_landau_s3p4_reference"]
            ),
        }
        for row in phase_rows_in
    ]
    write_rows(
        DATA / "phase_map_validation.csv",
        [
            "s",
            "scalar_phase_tail",
            "stuart_landau_phase_tail",
            "stuart_landau_s3p4_reference",
        ],
        phase_rows,
    )

    # Preserve the schedule noise nonlinear/modal cap evidence as archived data even
    # though the main figure now prioritizes the two Network closure tests.
    phase_offsets = 2.0 * np.array([0.08, -0.05, 0.12, -0.10, 0.03, -0.08])
    phase_offsets -= phase_offsets.mean()
    modal_initial = V_RING.T @ phase_offsets
    sigma_network = float(summary["sigma_network"])
    fine_caps = np.geomspace(0.5, 10.0, 400)
    fine_rows = []
    for cap in fine_caps:
        bias, variance, total = modal_terminal_components(
            float(cap), sigma_network, LAM_RING, modal_initial
        )
        fine_rows.append(
            {
                "K": float(cap),
                "exact_total_mse": total,
                "deterministic_bias": bias,
                "noise_variance": variance,
            }
        )
    write_rows(
        DATA / "noise_curves.csv",
        ["K", "exact_total_mse", "deterministic_bias", "noise_variance"],
        fine_rows,
    )

    # Stochastic replaces the legacy 800-path independent-cap record by a
    # 20,000-path common-random-number study with paired simultaneous intervals.
    mc_rows_in = read_rows(ROOT / "stochastic_crn_cap_validation.csv")
    mc_rows = []
    for row in mc_rows_in:
        half_width = 0.5 * (
            float(row["mc_ci95_high"]) - float(row["mc_ci95_low"])
        )
        mc_rows.append(
            {
                "K": float(row["K"]),
                "K_over_Kopt": float(row["K_over_Kopt"]),
                "modal_exact_mse": float(row["modal_exact_mse"]),
                "nonlinear_mc_mse": float(row["nonlinear_mc_mse"]),
                "mc_error95": half_width,
                "paired_difference_vs_Kopt": float(
                    row["paired_difference_vs_Kopt"]
                ),
                "paired_simultaneous_ci95_low": float(
                    row["paired_simultaneous_ci95_low"]
                ),
                "paired_simultaneous_ci95_high": float(
                    row["paired_simultaneous_ci95_high"]
                ),
                "relative_error_percent": 100.0
                * float(row["relative_modal_error"]),
            }
        )
    write_rows(
        DATA / "noise_monte_carlo.csv",
        [
            "K",
            "K_over_Kopt",
            "modal_exact_mse",
            "nonlinear_mc_mse",
            "mc_error95",
            "paired_difference_vs_Kopt",
            "paired_simultaneous_ci95_low",
            "paired_simultaneous_ci95_high",
            "relative_error_percent",
        ],
        mc_rows,
    )

    step_rows = read_rows(ROOT / "stochastic_step_refinement.csv")
    write_rows(
        DATA / "stochastic_step_refinement.csv",
        list(step_rows[0].keys()),
        step_rows,
    )

    # Archive the common-snapshot comparison metrics as a machine-readable
    # table source.
    comparison_rows_in = read_rows(ROOT / "comparison_metrics.csv")
    write_rows(
        DATA / "comparison_metrics.csv",
        list(comparison_rows_in[0].keys()),
        comparison_rows_in,
    )

    metadata = {
        "experiment": "Asymptotic",
        "data_only": True,
        "output_format": "CSV",
        "plot_floor": PLOT_FLOOR,
        "common_snapshot": float(summary["common_snapshot"]),
        "Kopt_network_modal": k_opt,
        "quadratic_tangential_offset_slope": tangent_slope,
        "scalar_phase_tail_slope": float(phase_map["scalar_fitted_slope"]),
        "stuart_landau_phase_tail_slope": float(
            phase_map["stuart_landau_fitted_slope"]
        ),
        "full_clock_continuous_phase_error": float(
            summary["full_clock_phase_regression"]["continuous_phase_error"]
        ),
        "inverse_square_exact_fitted_slope": 0.9341354322738368,
        "inverse_square_refined_fitted_slope": 0.9342239274939289,
        "source_summary": (
            "numerical_summary.json; stochastic_summary.json; "
            "asymptotic_asymptotic_validation.json"
        ),
    }
    (DATA / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
