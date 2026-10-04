#!/usr/bin/env python3
"""Randomized validation of the Actuator fiber-preserving plant realization.

The validation checks the finite-dimensional matrix realization underlying
Corollary (Fiber-preserving scheduled realization): if U spans ker B_parallel
and C = B_perp U has full row rank, then

    H = U C^T (C C^T)^{-1}

satisfies B_parallel H = 0 and B_perp H = I.  It also verifies that the
scheduled input -kappa H R0 eta has zero tangential contribution and exactly
generates -kappa R0 eta in the normal channel.  A separate square case checks
H = Gamma^{-1}.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from artifact_paths import work_dir, output_dir, code_dir
ROOT = work_dir()
JSON_OUT = ROOT / "actuator_plant_realization_validation.json"
TXT_OUT = ROOT / "actuator_plant_realization_validation.txt"
SEED = 20260829
TOL = 2.5e-11


@dataclass
class Check:
    name: str
    passed: bool
    value: float
    tolerance: float
    detail: str = ""


def nullspace(a: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Return an orthonormal basis for ker(a) using SVD."""
    _u, s, vh = np.linalg.svd(a, full_matrices=True)
    if s.size == 0:
        rank = 0
    else:
        rank = int(np.sum(s > rtol * max(a.shape) * s[0]))
    return vh[rank:].T.copy()


def add(checks: list[Check], name: str, value: float, tolerance: float, detail: str = "") -> None:
    checks.append(Check(name=name, passed=bool(np.isfinite(value) and value <= tolerance), value=float(value), tolerance=float(tolerance), detail=detail))


def make_spd(rng: np.random.Generator, q: int) -> np.ndarray:
    a = rng.normal(size=(q, q))
    return a.T @ a + (0.35 + 0.15 * rng.random()) * np.eye(q)


def randomized_general_cases(rng: np.random.Generator, checks: list[Check], cases: int = 640) -> dict[str, float]:
    maxima = {
        "tangential_right_inverse": 0.0,
        "normal_right_inverse": 0.0,
        "scheduled_tangential": 0.0,
        "scheduled_normal": 0.0,
        "range_projection": 0.0,
    }
    for idx in range(cases):
        # Input dimension and tangential rank are selected so ker B_parallel
        # has enough dimensions to cover the normal bundle.
        m_u = int(rng.integers(3, 11))
        r_par = int(rng.integers(0, m_u))
        d_ker = m_u - r_par
        q = int(rng.integers(1, d_ker + 1))
        p = max(1, r_par + int(rng.integers(0, 3)))

        # Construct a rank-r_par tangential block with controlled conditioning.
        if r_par == 0:
            b_par = np.zeros((p, m_u))
        else:
            left = rng.normal(size=(p, r_par))
            while np.linalg.matrix_rank(left) < r_par:
                left = rng.normal(size=(p, r_par))
            right = rng.normal(size=(r_par, m_u))
            while np.linalg.matrix_rank(right) < r_par:
                right = rng.normal(size=(r_par, m_u))
            b_par = left @ right

        u_basis = nullspace(b_par)
        if u_basis.shape[1] < q:
            raise RuntimeError("constructed kernel dimension is smaller than normal dimension")

        # Full-row-rank restriction C on the vertical input subspace.
        c = rng.normal(size=(q, u_basis.shape[1]))
        while np.linalg.matrix_rank(c) < q or np.linalg.cond(c @ c.T) > 2.0e7:
            c = rng.normal(size=(q, u_basis.shape[1]))

        # B_perp is chosen so its restriction to ker B_parallel is exactly C.
        b_perp = c @ u_basis.T
        h = u_basis @ c.T @ np.linalg.inv(c @ c.T)

        e_tan = np.linalg.norm(b_par @ h, ord=2)
        e_norm = np.linalg.norm(b_perp @ h - np.eye(q), ord=2)
        e_range = np.linalg.norm((np.eye(m_u) - u_basis @ u_basis.T) @ h, ord=2)

        kappa = float(np.exp(rng.uniform(math.log(0.05), math.log(40.0))))
        r0 = make_spd(rng, q)
        eta = rng.normal(size=q)
        u_sched = -kappa * h @ r0 @ eta
        e_sched_tan = np.linalg.norm(b_par @ u_sched)
        e_sched_norm = np.linalg.norm(b_perp @ u_sched + kappa * r0 @ eta)

        scale_tan = 1.0 + np.linalg.norm(b_par, ord=2) * np.linalg.norm(h, ord=2)
        scale_norm = 1.0 + np.linalg.norm(b_perp, ord=2) * np.linalg.norm(h, ord=2)
        scale_sched_tan = 1.0 + np.linalg.norm(b_par, ord=2) * np.linalg.norm(u_sched)
        scale_sched_norm = 1.0 + np.linalg.norm(kappa * r0 @ eta)
        scale_range = 1.0 + np.linalg.norm(h, ord=2)

        vals = {
            "tangential_right_inverse": e_tan / scale_tan,
            "normal_right_inverse": e_norm / scale_norm,
            "scheduled_tangential": e_sched_tan / scale_sched_tan,
            "scheduled_normal": e_sched_norm / scale_sched_norm,
            "range_projection": e_range / scale_range,
        }
        for key, value in vals.items():
            maxima[key] = max(maxima[key], float(value))
            add(checks, f"general_{idx:04d}_{key}", float(value), TOL, f"m_u={m_u}, p={p}, q={q}, dimker={u_basis.shape[1]}")

    return maxima


def randomized_square_cases(rng: np.random.Generator, checks: list[Check], cases: int = 160) -> dict[str, float]:
    maxima = {"square_inverse": 0.0, "square_scheduled_normal": 0.0}
    for idx in range(cases):
        q = int(rng.integers(1, 9))
        gamma = rng.normal(size=(q, q))
        # Avoid ill-conditioned instances so validation tests the identity,
        # not floating-point inversion pathologies.
        gamma += (q + 0.7) * np.eye(q)
        while np.linalg.cond(gamma) > 1.0e6:
            gamma = rng.normal(size=(q, q)) + (q + 0.7) * np.eye(q)
        h = np.linalg.inv(gamma)
        e_inv = np.linalg.norm(gamma @ h - np.eye(q), ord=2) / (1.0 + np.linalg.norm(gamma, ord=2) * np.linalg.norm(h, ord=2))

        kappa = float(np.exp(rng.uniform(math.log(0.05), math.log(40.0))))
        r0 = make_spd(rng, q)
        eta = rng.normal(size=q)
        u_sched = -kappa * h @ r0 @ eta
        e_sched = np.linalg.norm(gamma @ u_sched + kappa * r0 @ eta) / (1.0 + np.linalg.norm(kappa * r0 @ eta))
        maxima["square_inverse"] = max(maxima["square_inverse"], float(e_inv))
        maxima["square_scheduled_normal"] = max(maxima["square_scheduled_normal"], float(e_sched))
        add(checks, f"square_{idx:04d}_inverse", float(e_inv), TOL, f"q={q}")
        add(checks, f"square_{idx:04d}_scheduled_normal", float(e_sched), TOL, f"q={q}")
    return maxima


def deterministic_failure_guard(checks: list[Check]) -> dict[str, float]:
    """Verify that loss of vertical surjectivity is detectable.

    This is not a pass/fail identity from the corollary.  It checks that the
    deliberately rank-deficient restriction has nonzero identity residual,
    documenting why the full-row-rank hypothesis is substantive.
    """
    b_par = np.array([[1.0, 0.0, 0.0, 0.0]])
    u_basis = nullspace(b_par)
    c_bad = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    h_pinv = u_basis @ np.linalg.pinv(c_bad)
    residual = float(np.linalg.norm(c_bad @ (u_basis.T @ h_pinv) - np.eye(2), ord=2))
    # For this guard, pass means rank loss is visibly nonzero.
    checks.append(Check("rank_deficiency_guard", residual > 0.5, residual, 0.5, "expected residual > 0.5 when vertical surjectivity fails"))
    return {"rank_deficiency_identity_residual": residual}


def main() -> int:
    rng = np.random.default_rng(SEED)
    checks: list[Check] = []
    general = randomized_general_cases(rng, checks)
    square = randomized_square_cases(rng, checks)
    guard = deterministic_failure_guard(checks)

    passed = sum(c.passed for c in checks)
    total = len(checks)
    result = {
        "experiment": "Actuator",
        "status": "PASS" if passed == total else "FAIL",
        "seed": SEED,
        "checks_total": total,
        "checks_passed": passed,
        "checks_failed": total - passed,
        "randomized_general_cases": 640,
        "randomized_square_cases": 160,
        "tolerance": TOL,
        "maximum_normalized_errors": {**general, **square},
        "hypothesis_guard": guard,
        "checks": [asdict(c) for c in checks],
    }
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "Actuator PLANT-REALIZATION VALIDATION",
        "=" * 48,
        f"Overall: {result['status']}",
        f"Checks: {passed}/{total} PASS",
        f"Seed: {SEED}",
        f"General randomized cases: {result['randomized_general_cases']}",
        f"Square randomized cases: {result['randomized_square_cases']}",
        "",
        "Maximum normalized identity errors:",
    ]
    for key, val in result["maximum_normalized_errors"].items():
        lines.append(f"  {key}: {val:.6e}")
    lines += [
        "",
        f"Rank-deficiency guard residual: {guard['rank_deficiency_identity_residual']:.6e}",
        "",
        "Validated identities:",
        "  B_parallel H = 0",
        "  B_perp H = I",
        "  B_parallel(-kappa H R0 eta) = 0",
        "  B_perp(-kappa H R0 eta) = -kappa R0 eta",
        "  H = Gamma^{-1} in the square nonsingular special case",
    ]
    TXT_OUT.write_text("\n".join(lines) + "\n")
    print(TXT_OUT.read_text(), end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
