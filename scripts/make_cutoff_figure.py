# Copyright 2026 Sergio Sánchez de la Fuente
# SPDX-License-Identifier: Apache-2.0
"""Generate README figures directly from the saved cutoff analysis."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bibliometrics/analysis/results"
FIGURES = ROOT / "figures"


def read(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def trends(summary):
    a, b, c = [read(f"method_{method}_annual.csv") for method in ("a", "b", "c")]
    series = [
        (a, "intrinsic_pct", "A: contextual rules", "#0072B2", "-", "o"),
        (b, "semantic_intrinsic_pct", "B: learned only", "#009E73", "-", "s"),
        (c, "computed_pure", "C: learned only", "#D55E00", "-", "^"),
        (b, "semantic_intrinsic_presence_pct", "B: including combined", "#009E73", "--", None),
        (c, "computed_combined", "C: including combined", "#D55E00", "--", None),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.7))
    plotted = []
    for rows, field, label, color, style, marker in series:
        selected = [row for row in rows if 2010 <= int(row["year"]) <= 2025]
        years = [int(row["year"]) for row in selected]
        values = []
        for row in selected:
            if field.startswith("computed_"):
                learned, explicit, combined = [int(row[k]) for k in ("learned_internal", "explicit_external", "combined")]
                numerator = learned + (combined if field == "computed_combined" else 0)
                denominator = numerator + explicit
                value = 100 * numerator / denominator if denominator else np.nan
            else:
                value = float(row[field]) if row[field] else np.nan
            values.append(value)
        ax.plot(years, values, color=color, linestyle=style, marker=marker, markersize=4,
                linewidth=1.7, label=label)
        plotted.extend({"series": label, "year": year, "percentage": value} for year, value in zip(years, values))
    cutoff = summary["decision"]["final_year"]
    ax.axvline(cutoff, color="#333333", linestyle=":", linewidth=1.2)
    ax.text(cutoff + 0.15, 97, f"Selected cutoff: {cutoff}", va="top", fontsize=10)
    ax.set(title="Annual evolution of world-model terminology", xlabel="Publication year",
           ylabel="Learned-model share in compared categories (%)", ylim=(0, 100), xlim=(2009.7, 2025.3))
    ax.set_xticks([2010, 2013, 2016, 2020, 2023, 2025])
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    save(fig, "annual_trends")
    return plotted


def estimates(summary):
    step = summary["decision"]["step"]
    keys = ["method_a", "method_b_learned_only", "method_b_including_combined",
            "method_c_learned_only", "method_c_including_combined"]
    labels = ["A: contextual rules", "B: learned only", "B: including combined",
              "C: learned only", "C: including combined", "Primary boundary (lower median)",
              "Bayesian check", "Multinomial check"]
    values = [step[key]["year"] for key in keys] + [summary["decision"]["primary_year"],
              summary["decision"]["family_years"]["bayesian_change"],
              summary["decision"]["family_years"]["multinomial_change"]]
    cutoff = summary["decision"]["final_year"]
    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.axvline(cutoff, color="#333333", linestyle=":", linewidth=1.2)
    ax.axhline(4.5, color="#d1d5db", linewidth=1)
    plotted = []
    for index, (label, year) in enumerate(zip(labels, values)):
        support = step[keys[index]]["support"] if index < 5 else [year]
        color = "#009E73" if index >= 5 else "#0072B2"
        ax.hlines(index, min(support), max(support), color=color, linewidth=2, alpha=.65)
        ax.scatter(year, index, color=color, marker="D" if index >= 5 else "o", s=42, zorder=3)
        ax.annotate(str(year), (year, index), xytext=(8, 5), textcoords="offset points", color=color, fontsize=10)
        plotted.append({"series": label, "year": year, "support_min": min(support), "support_max": max(support)})
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set(title="Cutoff estimates and complementary checks", xlabel="Estimated change year", xlim=(2017.5, 2022))
    ax.set_xticks([2018, 2019, 2020, 2021, 2022])
    ax.grid(axis="x", color="#e5e7eb")
    save(fig, "cutoff_estimates")
    return plotted


def sensitivity(summary):
    cutoff = summary["decision"]["final_year"]
    checks = summary["leave_one_source_out"]
    labels = ["None: all six sources"] + [row["omitted_source"] for row in checks]
    values = [cutoff] + [row["final_year"] for row in checks]
    fig, ax = plt.subplots(figsize=(8, 4.1))
    ax.axvline(cutoff, color="#333333", linestyle=":", linewidth=1.2)
    for index, year in enumerate(values):
        color = "#D55E00" if year != cutoff else "#0072B2"
        ax.scatter(year, index, color=color, marker="D" if index == 0 else "o", s=48, zorder=3)
        ax.annotate(str(year), (year, index), xytext=(8, 0), textcoords="offset points", va="center", color=color)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set(title="Sensitivity to database omission", xlabel="Selected inclusive cutoff year", ylabel="Database omitted",
           xlim=(2017.5, 2021))
    ax.set_xticks([2018, 2019, 2020, 2021])
    ax.grid(axis="x", color="#e5e7eb")
    save(fig, "database_sensitivity")
    return [{"omitted_source": label, "year": year} for label, year in zip(labels, values)]


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans"})
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS / "cutoff/summary.json").read_text(encoding="utf-8"))
    outputs = [("cutoff_annual_values.csv", trends(summary)),
               ("cutoff_estimate_values.csv", estimates(summary)),
               ("database_sensitivity_values.csv", sensitivity(summary))]
    for name, rows in outputs:
        with (FIGURES / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("Generated annual_trends, cutoff_estimates, and database_sensitivity in PNG (300 dpi).")


if __name__ == "__main__":
    main()
