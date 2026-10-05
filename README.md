# Foundation and World Models for Outdoor Robot Navigation: A Systematic Review

This repository contains the bibliometric analysis used to support the review's inclusive **2020 publication-year cutoff**. It provides the data-processing and classification scripts, saved results, and figures showing how the boundary was selected and how it changes under alternative calculations and database coverage.

## 1. Sources and corpus preparation

Records were obtained from six bibliographic sources. Overlapping export fragments were consolidated before cross-database deduplication.

| Source | Records entering cross-database deduplication |
| --- | ---: |
| ACM Digital Library | 2,581 |
| IEEE Xplore | 702 |
| ScienceDirect | 191 |
| Scopus | 10,817 |
| SpringerLink | 5,278 |
| Web of Science | 819 |
| **Total** | **20,388** |

[prepare_exports.py](bibliometrics/analysis/scripts/prepare_exports.py) consolidates 18 original exports into seven input files. [corpus.py](bibliometrics/analysis/scripts/corpus.py) parses their bibliographic fields, normalizes identifiers and titles, removes duplicate publications, and applies the contextual rule-based classifier. Duplicates are identified using bibliographic identifiers and normalized titles; conflicting publication years are resolved using the earliest observed year.

| Processing step | Records |
| --- | ---: |
| Consolidated records before cross-database deduplication | 20,388 |
| Duplicate records removed | 2,279 |
| Unique publications | 18,109 |
| Publications without a year | 11 |
| Unique publications with a year | 18,098 |
| Dated publications outside 2010–2025 | 7,505 |
| **Publications in the temporal analysis window, 2010–2025** | **10,593** |

The incomplete 2026 publication year is excluded from the temporal analysis. Records without a year are not assigned to a time period. The dated analysis window includes all classification categories; each plotted proportion uses only the categories being compared.

Source-level counts and field availability are recorded in [sources.csv](bibliometrics/analysis/results/sources.csv); export identities and integrity hashes are listed in [provenance.json](provenance.json). Exact search strings and execution dates are not included in the available bibliometric search record.

## 2. Classifying world-model terminology

Titles, abstracts, and keywords, when available, are classified using three approaches:

| Method | Implementation | Purpose |
| --- | --- | --- |
| A: contextual lexical rules | [rules.py](bibliometrics/analysis/scripts/rules.py) | Identify contextual evidence for learned predictive models or explicit environment representations |
| B: text classifier | [classify_b.py](bibliometrics/analysis/scripts/classify_b.py), `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | Classify learned, explicit, combined, ambiguous, or unrelated uses |
| C: second text classifier | [classify_c.py](bibliometrics/analysis/scripts/classify_c.py), `cross-encoder/nli-deberta-v3-small` | Repeat classification with alternative hypotheses and a relevance filter |

Five annual series are constructed: one for A and two each for B and C. The solid curves compare learned-only and explicit categories. The dashed curves include combined learned-and-explicit representations in the learned numerator and compared-category denominator. Ambiguous and unrelated records are excluded from these proportions. B and C use related architectures, so agreement is a sensitivity check, not independent confirmation.

The text-model inputs are truncated to 192 tokens. SpringerLink exports contain no abstracts or keywords, so those records are classified from titles alone. These differences in available text can affect comparisons between databases.

![Annual learned-model proportions under the three classifiers](figures/annual_trends.png)

The rule-based series shows a pronounced rise around 2019–2020; the text classifiers provide a broader and less uniform picture. The boundary is therefore selected from the recorded decision rule, not by visually choosing the steepest curve.

## 3. Why 2020 is selected

For each of the five series, [select_cutoff.py](bibliometrics/analysis/scripts/select_cutoff.py) fits a sustained before/after change in category proportions. Candidate boundaries range from 2014 to 2022. The lower median of the five estimated years gives the primary boundary.

| Primary series | Estimated change year |
| --- | ---: |
| A: contextual rules | 2019 |
| B: learned only | 2020 |
| B: including combined representations | 2020 |
| C: learned only | 2020 |
| C: including combined representations | 2021 |
| **Primary boundary: lower median** | **2020** |

Bayesian and multinomial change calculations provide two complementary checks. The primary boundary is retained when both checks lie within one year; otherwise the median of the three decision families is used. Here, all three decision-family estimates are **2020**, so the primary boundary is retained. These calculations reuse data and classifications and are not independent votes.

![Primary series estimates and complementary decision-family estimates](figures/cutoff_estimates.png)

Horizontal bars show the range of compatible years recorded for each primary series; markers show the selected estimates. The green markers summarize the primary boundary and the two complementary decision-family checks.

Segmented regression is retained as a secondary slope-change diagnostic. Its central estimate is 2022; only one of the five series improves the information criterion over a single-trend fit. This diagnostic does not enter the automatic cutoff decision rule. The exact calculations, support intervals, and diagnostic outputs are available in [summary.json](bibliometrics/analysis/results/cutoff/summary.json) and the [detailed method description](docs/cutoff_method.md).

## 4. Sensitivity to database coverage

The full decision procedure is repeated after omitting each database in turn.

![Selected cutoff after omitting each database](figures/database_sensitivity.png)

**2020 is retained in five of the six omission checks. Removing SpringerLink changes the boundary to 2018.** This sensitivity limits the claim: 2020 is a supported, reproducible scope boundary for the combined corpus, not a universal historical turning point. Automatic classifications, missing abstracts, database composition, and the selected time window also affect the result.

## 5. Reproduce the analysis

Use Python 3.10 or later, from the repository root:

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

This command reuses the saved classifications, recalculates the complete cutoff analysis, regenerates all figures, and checks numerical agreement with the reference outputs. It does not download language models or run new neural inference.

To also rebuild the corpus and rule-based classifications from the original exports:

```bash
python scripts/reproduce.py --from-raw
```

Place the 18 exports listed in [provenance.json](provenance.json) in `bibliometrics/data/raw/`. Original exports, consolidated inputs, and the full abstract-bearing BibTeX corpus are excluded from Git; the default workflow uses the included classification tables. Before reusing neural predictions during a raw-data rebuild, the scripts check identifiers, titles, years, and text availability against the reconstructed corpus.

New neural inference is optional and requires PyTorch, Transformers, and the relevant model downloads in addition to the pinned analysis dependencies. Method C expects its model locally. Method B records the model revision as `main`; neither method records an immutable model commit. The archived classification tables are therefore the reference for exact numerical reproduction.

## 6. Files

| Location | Contents |
| --- | --- |
| [bibliometrics/analysis/scripts/](bibliometrics/analysis/scripts/) | Export preparation, corpus processing, classification, and cutoff calculation |
| [bibliometrics/analysis/results/](bibliometrics/analysis/results/) | Classification tables, annual series, deduplication records, and cutoff diagnostics |
| [bibliometrics/reference/cutoff_summary.json](bibliometrics/reference/cutoff_summary.json) | Reference output for numerical checks |
| [scripts/](scripts/) | Reproduction entry point, figure generation, and integrity checks |
| [figures/](figures/) | Figures in PNG, SVG, and PDF, with plotted values in CSV |
| [docs/cutoff_method.md](docs/cutoff_method.md) | Detailed classification and decision procedure |
| [provenance.json](provenance.json) | Export manifest and integrity hashes |

## License

Code and original documentation are licensed under the [Apache License 2.0](LICENSE). Third-party bibliographic material remains subject to its original terms.

## Author

Sergio Sánchez de la Fuente.
