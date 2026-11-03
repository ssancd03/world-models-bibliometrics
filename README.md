# Foundation and World Models for Agricultural Robot Navigation: A Systematic Review

This repository contains the supporting materials of the systematic review:

- [review/](review/): search strategies for the six sources, the selection flow and the full-text decision for each assessed report.
- The bibliometric analysis supporting the review's inclusive **2020 publication-year cutoff**: processing scripts, saved results, and figures explaining the decision and its sensitivity. Sections 1 to 5 below describe it.

## 1. Sources and corpus preparation

Records were obtained from six bibliographic sources:

| Source | Records before cross-database deduplication |
| --- | ---: |
| ACM Digital Library | 2,581 |
| IEEE Xplore | 702 |
| ScienceDirect | 191 |
| Scopus | 10,817 |
| SpringerLink | 5,278 |
| Web of Science | 819 |
| **Total** | **20,388** |

Platform export limits required batch downloads, including publication-year ranges for ACM and SpringerLink. The **18 export files are download batches, not separate searches**.

- [prepare_exports.py](bibliometrics/analysis/scripts/prepare_exports.py) consolidates these files into seven inputs, removing overlapping exports. ScienceDirect retains two input files; the other databases have one each.
- [corpus.py](bibliometrics/analysis/scripts/corpus.py) reads the bibliographic fields, normalizes identifiers and titles, and removes cross-database duplicates. Conflicting publication years are resolved using the earliest observed year.

| Processing result | Records |
| --- | ---: |
| Duplicates removed | 2,279 |
| Publications remaining after deduplication | 18,109 |
| Publications without a year | 11 |
| Publications remaining with a year | 18,098 |
| **Publications from 2010–2025 used in the temporal analysis** | **10,593** |

The analysis starts in **2010** to focus on the recent evolution of learned predictive models while retaining a pre-2020 comparison period. It includes complete years through **2025**; **2026 is excluded because the year is incomplete**. Earlier publications remain in the full corpus, and undated records are excluded from the annual calculations.

Source counts and export integrity information are available in [sources.csv](bibliometrics/analysis/results/sources.csv) and [provenance.json](provenance.json). The manifest identifies the original local exports; those files are not distributed on GitHub. Saved classifications are included to reproduce the numerical analysis without them.

## 2. Annual terminology trends

The term *world model* can refer to a learned model that predicts how the environment changes, an explicit representation such as a map, or a combination of both. To examine how its use evolves, titles, abstracts, and keywords, when available, are classified using contextual rules (A) and two text classifiers (B and C).

For each year, the learned-model share is calculated as learned publications divided by learned plus explicit publications, multiplied by 100. B and C also provide a variant that adds combined representations to both the numerator and denominator. Together with A, this gives the five curves below. Ambiguous and unrelated records are excluded: these are proportions within the compared categories, not percentages of all retrieved publications.

![Annual terminology trends](figures/annual_trends.png)

The increase is clearest for A around 2019–2020. C shows a more gradual rise, while B fluctuates and declines in the most recent years. The differences show why the cutoff is calculated from several series rather than selected by visual inspection of one curve. These trends describe terminology, not navigation performance. B and C use related architectures and inputs limited to 192 tokens; SpringerLink lacks abstracts and keywords, so its classifications rely on titles alone.

## 3. Selecting 2020

The next step turns the annual counts into an explicit cutoff. Each candidate year from **2014 to 2022** splits a series into earlier publications and publications from that year onward. A binomial model estimates a separate learned-model proportion for each period and selects the split that best fits the counts. The median of the five selected years gives the primary boundary.

Two complementary checks examine the same data: a Bayesian calculation compares the support for each candidate split, and a multinomial calculation examines changes across learned, explicit, and combined categories together. The primary boundary is retained if both checks fall within one year; otherwise, the lower median of the three decision-family estimates is used.

![Cutoff estimates and complementary checks](figures/cutoff_estimates.png)

The five primary estimates are **2019, 2020, 2020, 2020, and 2021**: their median is **2020**, also selected by both checks. Markers show selected estimates; horizontal bars show candidate years with near-best fits for the primary series. The resulting cutoff is inclusive: studies published in 2020 are eligible for the review. The complementary calculations strengthen the consistency of this selection within the corpus, while using the same data rather than independent samples. Full calculations and additional diagnostics are documented in [cutoff_method.md](docs/cutoff_method.md) and [summary.json](bibliometrics/analysis/results/cutoff/summary.json).

## 4. Database sensitivity

Different databases cover different publications and supply different amounts of text. To assess whether one source drives the decision, the full cutoff procedure is repeated six times, omitting one database at a time. A publication is removed only if it has no remaining source; records also indexed elsewhere are retained. The saved classifications are reused, without rerunning the text classifiers.

![Cutoff after omitting each database](figures/database_sensitivity.png)

The first row shows the complete corpus; the others show the selected year after each omission. **2020 is retained in five of six checks; removing SpringerLink gives 2018.** The selection therefore remains consistent across most source combinations, including the exclusion of Scopus, the largest source before deduplication. The SpringerLink result identifies the one combination that shifts the boundary, providing a clear indication of its sensitivity. The individual checks are available in [leave_one_source_out.csv](bibliometrics/analysis/results/cutoff/leave_one_source_out.csv).

### Overall interpretation: why 2020?

Taken together, the annual trends indicate greater use of learned predictive representations, although the timing and magnitude depend on the classifier. The primary estimates cluster around 2020: three select it directly, the other two select 2019 and 2021, and **2020 lies within the compatible-year sets of all five series**. Both complementary decision families also select 2020, and five of six database-omission checks retain it. This convergence makes **2020 the best-supported inclusive cutoff under the specified procedure for this corpus**: it combines agreement across the primary series, complementary calculations, and most source combinations. The SpringerLink exception is reported alongside that agreement. The cutoff defines the review's scope around this recent change in terminology; it does not claim that world models first appeared in 2020.

## 5. Reproduce the results

The published repository includes the saved classification tables used in the analysis, so the default workflow needs neither the original database exports nor new text-model inference. With Python 3.10 or later, run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

The workflow recalculates the cutoff and database-omission checks, regenerates the three PNG figures and their plotted CSV data, and compares the numerical results with the stored reference. Calculation outputs are written to `bibliometrics/analysis/results/cutoff/`, and figures to `figures/`. A successful run ends with a `PASS` message; the expected inclusive cutoff is **2020**.

To also rebuild the corpus and rule-based labels, run `python scripts/reproduce.py --from-raw` after supplying the 18 original exports at the paths listed in `provenance.json`. This option still reuses the saved B and C predictions after checking record alignment. Original exports are excluded from Git; additional requirements for new text-model inference are described in the [method documentation](docs/cutoff_method.md).

## License

Code and original documentation: [Apache License 2.0](LICENSE). Third-party bibliographic material retains its original terms.
