# Selecting an inclusive 2020 publication cutoff

## Purpose and scope

The cutoff is supported by an analysis of world-model terminology in robotics and navigation. It does not mean that predictive models first appeared in 2020. It defines a practical scope for examining the recent shift towards learned representations.

After consolidating overlapping export fragments, the six-source inputs contain 20,388 records. The parser resolves duplicates using bibliographic identifiers and normalized title matching, producing 18,109 publications. Conflicting years are resolved using the earliest observed year. Eleven records have no publication year. The temporal analysis uses the 10,593 dated records from 2010–2025; the incomplete 2026 year is excluded. Candidate changes range from 2014 to 2022.

Platform export limits required batch downloads, including separate publication-year ranges for ACM Digital Library and SpringerLink. The 18 original files are export batches, not independent searches. They are consolidated into seven analysis inputs before cross-database deduplication.

The analysis starts in 2010 to focus on the recent evolution of learned predictive models while retaining a pre-2020 comparison period. Earlier records remain in the full corpus but are outside this temporal comparison. The lower bound is a scope choice, not an estimated or optimized historical starting point. The window ends at 2025 to avoid comparing the incomplete 2026 year with complete years.

## Classification

Method A uses contextual lexical rules in titles, abstracts, and keywords. Method B uses `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`; Method C uses `cross-encoder/nli-deberta-v3-small` and a relevance filter. The neural methods use related architectures and cannot be regarded as independent validation. The archived predictions are used for numerical reproduction.

The text-model inputs are truncated to 192 tokens. SpringerLink exports contain no abstracts or keywords, so their classifications rely on titles alone.

Five annual series are analyzed: the rule-based learned-versus-explicit series, and two variants for each text classifier. The first variant excludes combined representations; the second counts them with learned representations. Ambiguous and unrelated records are excluded from these compared-category proportions.

## Decision rule

The primary calculation fits a before/after binomial level shift to each series. The lower median of the five estimated years gives the primary boundary. Bayesian change calculations and multinomial category-change calculations provide two complementary checks. Retain the primary boundary if both checks are within one year; otherwise use the median of the three decision families. These are related diagnostics, not independent votes.

The primary series estimates are 2019, 2020, 2020, 2020, and 2021. The primary, Bayesian, and multinomial family estimates are all 2020. Segmented regression is retained as a secondary diagnostic; its central estimate of 2022 does not replace the adopted boundary.

The same procedure is repeated after removing each source. Omitting ACM, IEEE, ScienceDirect, Scopus, or Web of Science retains 2020. Omitting SpringerLink gives 2018. The five unchanged selections support consistency across most source combinations; the SpringerLink exception is shown explicitly in the figure.

## Outputs and limitations

[select_cutoff.py](../bibliometrics/analysis/scripts/select_cutoff.py) contains the exact formulas. [summary.json](../bibliometrics/analysis/results/cutoff/summary.json) includes the decision, support intervals, individual-source estimates, leave-one-source-out estimates, and classifier agreement. [make_cutoff_figure.py](../scripts/make_cutoff_figure.py) generates the figures and their plotted data. [verify_results.py](../scripts/verify_results.py) checks complete numerical outputs against the reference snapshot.

The analysis is affected by terminology, database composition, automatic labeling, missing abstracts, and time-window choices. It establishes a reproducible scope boundary, not a causal historical turning point or proof that every pre-2020 study is irrelevant.

## Reproduction details

The default reproduction command uses the saved classification tables and the dependencies pinned in `requirements.txt`; it does not perform new neural inference. The `--from-raw` option also reconstructs the corpus and rule-based labels, then checks alignment before reusing the saved text-model predictions. Supply the 18 original exports listed in `provenance.json` for this option; those files are excluded from Git.

New text-model inference requires PyTorch, Transformers, and the relevant model downloads in addition to the analysis dependencies. Method C expects its model to be available locally. Method B records the model revision as `main`; neither method records an immutable model commit. The archived classifications are therefore the reference for exact numerical reproduction.
