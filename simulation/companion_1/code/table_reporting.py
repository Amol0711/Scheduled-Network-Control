"""Data-only tables using explicit display scales and decimal conventions."""
from pathlib import Path
import csv


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_display_tables(output: Path, caps: list[dict], steps: list[dict], rich: list[dict]) -> None:
    """Round only display products; full precision remains in statistics CSVs."""
    cap_rows = []
    for r in caps:
        half = (r['ci95_high'] - r['ci95_low']) * .5
        cap_rows.append(dict(
            cap_ratio=f"{r['K_over_Kopt']:.3f}",
            modal_mse_times_1e4=f"{r['modal_mse']*1e4:.6f}",
            mean_times_1e4=f"{r['nonlinear_mean']*1e4:.6f}",
            ci_halfwidth_times_1e4=f"{half*1e4:.6f}",
            discrepancy_percent=f"{r['relative_discrepancy_pct']:+.3f}",
            paired_difference_times_1e8=f"{r['paired_difference']*1e8:+.3f}",
            bonferroni_low_times_1e8=f"{r['bonferroni15_ci95_low']*1e8:+.3f}",
            bonferroni_high_times_1e8=f"{r['bonferroni15_ci95_high']*1e8:+.3f}"))
    write_rows(output / 'cap_display.csv', cap_rows)
    write_rows(output / 'step_display.csv', [dict(
        cap_ratio=f"{r['K_over_Kopt']:.2f}",
        coarse_mean_times_1e4=f"{r['coarse_mean']*1e4:.6f}",
        fine_mean_times_1e4=f"{r['fine_mean']*1e4:.6f}",
        shift_percent=f"{r['relative_shift_pct']:.4f}",
        paired_low_times_1e8=f"{r['paired_ci95_low']*1e8:.4f}",
        paired_high_times_1e8=f"{r['paired_ci95_high']*1e8:.4f}") for r in steps])
    write_rows(output / 'richardson_display.csv', [dict(
        cap_ratio=f"{r['K_over_Kopt']:.2f}",
        extrapolated_mean_times_1e4=f"{r['richardson_mean']*1e4:.6f}",
        se_times_1e4=f"{r['richardson_se']*1e4:.6f}",
        discrepancy_percent=f"{r['relative_discrepancy_pct']:+.4f}",
        relative_low_percent=f"{r['relative_ci95_low_pct']:+.4f}",
        relative_high_percent=f"{r['relative_ci95_high_pct']:+.4f}") for r in rich])
