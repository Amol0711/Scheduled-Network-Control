#!/usr/bin/env python3
"""Construct common-snapshot numerical comparisons from supplied experiment records."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expn

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()
RHO = 1.2
T = 2.0
ETA0 = 1.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def close(a: float, b: float, tol: float = 2e-12) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def main() -> None:
    method_rows = read_rows(ROOT / "method_metrics.csv")
    methods = {row["method"]: row for row in method_rows}
    summary = json.loads((ROOT / "numerical_summary.json").read_text())
    kopt = float(summary["Kopt_network_modal"])

    a = kopt * T
    i2 = {
        "constant": (1.0 - math.exp(-2.0 * RHO)) / (2.0 * RHO),
        "fiber_ptc": 1.0 / (2.0 * RHO + 1.0),
        "xu_liu_l2": math.exp(2.0 * RHO) * float(expn(2, 2.0 * RHO)),
        "capped": (
            (1.0 - a ** (-(2.0 * RHO + 1.0))) / (2.0 * RHO + 1.0)
            + a ** (-(2.0 * RHO + 1.0))
            * (1.0 - math.exp(-2.0 * RHO))
            / (2.0 * RHO)
        ),
    }

    augmented: list[dict[str, Any]] = []
    for row in method_rows:
        item: dict[str, Any] = dict(row)
        method = row["method"]
        i1 = float(row["normalized_offset_functional_I_over_T"])
        item["normalized_I1_over_T"] = i1
        item["normalized_I2_over_T"] = i2.get(method, "")
        augmented.append(item)
    fields = list(method_rows[0].keys()) + ["normalized_I1_over_T", "normalized_I2_over_T"]
    write_rows(ROOT / "comparison_metrics.csv", fields, augmented)

    # Exact optimized named-schedule curves and the bounded-profile envelope.
    opt_rows = read_rows(ROOT / "schedule_noise_optima.csv")
    by_schedule: dict[str, dict[float, dict[str, str]]] = {}
    for row in opt_rows:
        by_schedule.setdefault(row["schedule"], {})[float(row["sigma"])] = row
    env_rows = read_rows(ROOT / "dirichlet_envelope_curves.csv")
    env = {float(row["sigma"]): float(row["rms_envelope"]) for row in env_rows}
    sigmas = sorted(set(by_schedule["inverse_time"]) & set(by_schedule["inverse_square"]))

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected_i2 = {
        "constant": 0.37886751946274483,
        "fiber_ptc": 0.29411764705882354,
        "xu_liu_l2": 0.24759516974380757,
        "capped": 0.29445518561785633,
    }
    for method, expected in expected_i2.items():
        check(f"order-two functional {method}", close(i2[method], expected, 2e-13), i2[method])

    order = ["xu_liu_l2", "fiber_ptc", "capped", "constant"]
    i2_order = [i2[m] for m in order]
    check("order-two functional ordering", i2_order == sorted(i2_order), i2_order)
    ey_order = [float(methods[m]["terminal_reduced_trajectory_error"]) for m in order]
    check("measured terminal reduced-error ordering", ey_order == sorted(ey_order), ey_order)

    dis_ratio = float(methods["full_clock"]["terminal_disagreement_at_common_snapshot"]) / float(methods["fiber_ptc"]["terminal_disagreement_at_common_snapshot"])
    energy_ratio = float(methods["full_clock"]["control_energy_to_common_snapshot"]) / float(methods["fiber_ptc"]["control_energy_to_common_snapshot"])
    phase_error = float(methods["full_clock"]["terminal_continuous_phase_error"])
    reduced_error = float(methods["full_clock"]["terminal_reduced_trajectory_error"])
    check("full-clock comparable terminal disagreement", dis_ratio < 1.02, dis_ratio)
    check("full-clock comparable integrated effort", energy_ratio < 1.02, energy_ratio)
    check("full-clock large continuous phase error", phase_error > 20.4, phase_error)
    check("full-clock large reduced-flow error", reduced_error > 1.4, reduced_error)

    envelope_values: list[float] = []
    for sigma in sigmas:
        # The archived envelope sweep uses T=eta0=1.
        exact_env = sigma / math.sqrt(1.0 + sigma * sigma)
        csv_env = env[min(env, key=lambda x: abs(x - sigma))]
        rms1 = float(by_schedule["inverse_time"][sigma]["optimized_rms"])
        rms2 = float(by_schedule["inverse_square"][sigma]["optimized_rms"])
        envelope_values.append(csv_env)
        check(f"envelope formula sigma={sigma:.3e}", close(csv_env, exact_env, 2e-12), {"csv": csv_env, "formula": exact_env})
        check(f"inverse-time above envelope sigma={sigma:.3e}", rms1 > csv_env, {"inverse_time": rms1, "envelope": csv_env})
        check(f"inverse-square above envelope sigma={sigma:.3e}", rms2 > csv_env, {"inverse_square": rms2, "envelope": csv_env})
        check(f"inverse-square below inverse-time sigma={sigma:.3e}", rms2 < rms1, {"inverse_square": rms2, "inverse_time": rms1})

    slope = float(np.polyfit(np.log(sigmas), np.log(envelope_values), 1)[0])
    check("bounded-profile envelope displayed-range slope", abs(slope - 1.0) < 2e-7, slope)

    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    result = {
        "experiment": "Comparison",
        "status": status,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "key_values": {
            "I2_over_T": i2,
            "terminal_reduced_error": {m: float(methods[m]["terminal_reduced_trajectory_error"]) for m in order},
            "full_clock_disagreement_ratio_to_inverse_time": dis_ratio,
            "full_clock_energy_ratio_to_inverse_time": energy_ratio,
            "full_clock_phase_error": phase_error,
            "full_clock_terminal_reduced_error": reduced_error,
            "envelope_slope": slope,
            "envelope_sigma_min": min(sigmas),
            "envelope_sigma_max": max(sigmas),
        },
        "checks": checks,
    }
    (ROOT / "comparison_integration_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "Comparison HIERARCHY/ENVELOPE INTEGRATION VALIDATION",
        "=" * 62,
        f"Status: {status}",
        f"Checks: {result['checks_passed']}/{result['checks_total']} PASS",
        "",
        f"I_kappa1^(2)/T = {i2['fiber_ptc']:.15f}",
        f"I_kappa2^(2)/T = {i2['xu_liu_l2']:.15f}",
        f"Full-clock phase error = {phase_error:.12f} rad",
        f"Full-clock/inverse-time disagreement ratio = {dis_ratio:.12f}",
        f"Full-clock/inverse-time energy ratio = {energy_ratio:.12f}",
        f"Bounded-profile envelope slope = {slope:.12f}",
    ]
    (ROOT / "comparison_integration_validation.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
