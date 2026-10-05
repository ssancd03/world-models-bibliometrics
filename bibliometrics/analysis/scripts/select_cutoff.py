# Copyright 2026 Sergio Sánchez de la Fuente
# SPDX-License-Identifier: Apache-2.0
"""Select the publication cutoff and evaluate temporal and database sensitivity."""
from __future__ import annotations
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median_low
import numpy as np
from classify_b import normalized_title
STUDY_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = STUDY_DIR / 'analysis' / 'results' / 'method_b_classifications.csv'
METHOD_C_FILE = STUDY_DIR / 'analysis' / 'results' / 'method_c_classifications.csv'
OUTPUT_DIR = STUDY_DIR / 'analysis' / 'results' / 'cutoff'
START_YEAR = 2010
END_YEAR = 2025
YEARS = np.arange(START_YEAR, END_YEAR + 1)
CANDIDATES = list(range(2014, 2023))
SEMANTIC_CLASSES = ['INTRINSIC', 'EXTRINSIC', 'COMBINED', 'AMBIGUOUS', 'UNRELATED']
SEMANTIC_INDEX = {name: index for (index, name) in enumerate(SEMANTIC_CLASSES)}

def read_records() -> list[dict[str, str]]:
    with INPUT_FILE.open('r', encoding='utf-8-sig', newline='') as handle:
        records = list(csv.DictReader(handle))
    if len({row['record_id'] for row in records}) != len(records):
        raise RuntimeError('Method B contains duplicate identifiers')
    return records

def logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))

def binomial_loglik(k: np.ndarray, n: np.ndarray, probability: float | np.ndarray) -> float:
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    return float(np.sum(k * np.log(p) + (n - k) * np.log1p(-p)))

def step_change(k: np.ndarray, n: np.ndarray) -> dict:
    profiles = []
    for cutoff in CANDIDATES:
        before = YEARS < cutoff
        after = ~before
        p_before = float(k[before].sum() / n[before].sum())
        p_after = float(k[after].sum() / n[after].sum())
        loglik = binomial_loglik(k[before], n[before], p_before)
        loglik += binomial_loglik(k[after], n[after], p_after)
        profiles.append((cutoff, loglik, p_before, p_after))
    best = max(profiles, key=lambda item: item[1])
    support = [year for (year, ll, _, _) in profiles if 2 * (best[1] - ll) <= 3.84]
    overall_p = float(k.sum() / n.sum())
    null_loglik = binomial_loglik(k, n, overall_p)
    sample_size = int(n.sum())
    delta_bic = math.log(sample_size) - 2 * null_loglik - (3 * math.log(sample_size) - 2 * best[1])
    return {'year': best[0], 'support': support, 'loglik': best[1], 'before_probability': best[2], 'after_probability': best[3], 'delta_bic_vs_no_change': delta_bic}

def logistic_fit(x: np.ndarray, k: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, float]:
    beta = np.zeros(x.shape[1], dtype=float)
    beta[0] = math.log((k.sum() + 0.5) / (n.sum() - k.sum() + 0.5))
    for _ in range(80):
        eta = np.clip(x @ beta, -30, 30)
        probability = 1.0 / (1.0 + np.exp(-eta))
        gradient = x.T @ (k - n * probability)
        weights = n * probability * (1.0 - probability)
        information = x.T @ (weights[:, None] * x)
        information += np.eye(x.shape[1]) * 1e-09
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(information, gradient, rcond=None)[0]
        beta += step
        if float(np.max(np.abs(step))) < 1e-09:
            break
    eta = np.clip(x @ beta, -30, 30)
    probability = 1.0 / (1.0 + np.exp(-eta))
    loglik = binomial_loglik(k, n, probability)
    return (beta, loglik)

def segmented_change(k: np.ndarray, n: np.ndarray) -> dict:
    time = YEARS.astype(float) - START_YEAR
    profiles = []
    for cutoff in CANDIDATES:
        boundary = cutoff - 0.5 - START_YEAR
        hinge = np.maximum(0.0, time - boundary)
        design = np.column_stack([np.ones_like(time), time, hinge])
        (beta, loglik) = logistic_fit(design, k, n)
        profiles.append((cutoff, loglik, beta))
    best = max(profiles, key=lambda item: item[1])
    support = [year for (year, ll, _) in profiles if 2 * (best[1] - ll) <= 3.84]
    null_design = np.column_stack([np.ones_like(time), time])
    (_, null_loglik) = logistic_fit(null_design, k, n)
    sample_size = int(n.sum())
    delta_bic = 2 * math.log(sample_size) - 2 * null_loglik - (4 * math.log(sample_size) - 2 * best[1])
    beta = best[2]
    return {'year': best[0], 'support': support, 'loglik': best[1], 'odds_ratio_before_per_year': math.exp(float(beta[1])), 'odds_ratio_after_per_year': math.exp(float(beta[1] + beta[2])), 'delta_bic_vs_single_trend': delta_bic}

def log_beta_binomial_block(successes: int, total: int, alpha: float=0.5) -> float:
    failures = total - successes
    return math.lgamma(successes + alpha) + math.lgamma(failures + alpha) - math.lgamma(total + 2 * alpha) - 2 * math.lgamma(alpha) + math.lgamma(2 * alpha)

def highest_probability_set(years: list[int], probabilities: np.ndarray, mass: float=0.95) -> list[int]:
    selected = []
    cumulative = 0.0
    for index in np.argsort(probabilities)[::-1]:
        selected.append(years[int(index)])
        cumulative += float(probabilities[int(index)])
        if cumulative >= mass:
            break
    return sorted(selected)

def bayesian_binomial_change(k: np.ndarray, n: np.ndarray) -> dict:
    log_marginals = []
    for cutoff in CANDIDATES:
        before = YEARS < cutoff
        after = ~before
        log_marginal = log_beta_binomial_block(int(k[before].sum()), int(n[before].sum()))
        log_marginal += log_beta_binomial_block(int(k[after].sum()), int(n[after].sum()))
        log_marginals.append(log_marginal)
    log_marginals_array = np.asarray(log_marginals)
    probabilities = np.exp(log_marginals_array - logsumexp(log_marginals_array))
    mode_index = int(np.argmax(probabilities))
    return {'year': CANDIDATES[mode_index], 'support_95': highest_probability_set(CANDIDATES, probabilities), 'probabilities': {str(year): float(probability) for (year, probability) in zip(CANDIDATES, probabilities)}}

def log_dirichlet_multinomial_block(counts: np.ndarray, alpha: float=0.5) -> float:
    category_count = len(counts)
    total = int(counts.sum())
    result = math.lgamma(category_count * alpha) - math.lgamma(total + category_count * alpha)
    result += sum((math.lgamma(int(value) + alpha) - math.lgamma(alpha) for value in counts))
    return result

def bayesian_multinomial_change(counts: np.ndarray) -> dict:
    log_marginals = []
    for cutoff in CANDIDATES:
        before = YEARS < cutoff
        after = ~before
        log_marginal = log_dirichlet_multinomial_block(counts[before].sum(axis=0))
        log_marginal += log_dirichlet_multinomial_block(counts[after].sum(axis=0))
        log_marginals.append(log_marginal)
    log_marginals_array = np.asarray(log_marginals)
    probabilities = np.exp(log_marginals_array - logsumexp(log_marginals_array))
    mode_index = int(np.argmax(probabilities))
    return {'year': CANDIDATES[mode_index], 'support_95': highest_probability_set(CANDIDATES, probabilities), 'probabilities': {str(year): float(probability) for (year, probability) in zip(CANDIDATES, probabilities)}}

def empty_counts() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.zeros((len(YEARS), 2), dtype=int), np.zeros((len(YEARS), len(SEMANTIC_CLASSES)), dtype=int), np.zeros((len(YEARS), len(SEMANTIC_CLASSES)), dtype=int), np.zeros(len(YEARS), dtype=int))

def counts_from_records(records: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (a_counts, b_counts, c_counts, totals) = empty_counts()
    for row in records:
        value = row['year'].strip()
        if not value:
            continue
        year = int(value)
        if year < START_YEAR or year > END_YEAR:
            continue
        index = year - START_YEAR
        totals[index] += 1
        if row['v2_class'] == 'INTRINSIC':
            a_counts[index, 0] += 1
        elif row['v2_class'] == 'EXTRINSIC':
            a_counts[index, 1] += 1
        semantic_index = SEMANTIC_INDEX.get(row['semantic_class'])
        if semantic_index is not None:
            b_counts[index, semantic_index] += 1
        method_c_index = SEMANTIC_INDEX.get(row.get('method_c_class', ''))
        if method_c_index is not None:
            c_counts[index, method_c_index] += 1
    return (a_counts, b_counts, c_counts, totals)

def build_series(a_counts: np.ndarray, b_counts: np.ndarray, c_counts: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {'method_a': (a_counts[:, 0], a_counts[:, 0] + a_counts[:, 1]), 'method_b_learned_only': (b_counts[:, 0], b_counts[:, 0] + b_counts[:, 1]), 'method_b_including_combined': (b_counts[:, 0] + b_counts[:, 2], b_counts[:, 0] + b_counts[:, 1] + b_counts[:, 2]), 'method_c_learned_only': (c_counts[:, 0], c_counts[:, 0] + c_counts[:, 1]), 'method_c_including_combined': (c_counts[:, 0] + c_counts[:, 2], c_counts[:, 0] + c_counts[:, 1] + c_counts[:, 2])}

def decide_from_counts(a_counts: np.ndarray, b_counts: np.ndarray, c_counts: np.ndarray) -> dict:
    series = build_series(a_counts, b_counts, c_counts)
    step_results = {name: step_change(k, n) for (name, (k, n)) in series.items()}
    segmented_results = {name: segmented_change(k, n) for (name, (k, n)) in series.items()}
    bayesian_results = {name: bayesian_binomial_change(k, n) for (name, (k, n)) in series.items()}
    b_relevant_multinomial = bayesian_multinomial_change(b_counts[:, :3])
    b_full_multinomial = bayesian_multinomial_change(b_counts)
    c_relevant_multinomial = bayesian_multinomial_change(c_counts[:, :3])
    c_full_multinomial = bayesian_multinomial_change(c_counts)
    family_years = {'level_shift': int(median_low([result['year'] for result in step_results.values()])), 'segmented_regression': int(median_low([result['year'] for result in segmented_results.values()])), 'bayesian_change': int(median_low([result['year'] for result in bayesian_results.values()])), 'multinomial_change': int(median_low([b_relevant_multinomial['year'], c_relevant_multinomial['year']]))}
    segmented_bic_support = sum((result['delta_bic_vs_single_trend'] > 0 for result in segmented_results.values()))
    segmented_accelerations = sum((result['odds_ratio_after_per_year'] > result['odds_ratio_before_per_year'] for result in segmented_results.values()))
    segmented_eligible = segmented_bic_support >= 2 and segmented_accelerations >= 2
    primary = family_years['level_shift']
    validators = [family_years['bayesian_change'], family_years['multinomial_change']]
    validators_close = sum((abs(year - primary) <= 1 for year in validators))
    if validators_close == len(validators):
        final_year = primary
        rule_applied = 'Retain the primary binomial change: both validators lie within one year.'
    else:
        final_year = int(median_low([primary, *validators]))
        rule_applied = 'Use the median of the three decision families.'
    return {'final_year': final_year, 'primary_year': primary, 'validators_close': validators_close, 'rule_applied': rule_applied, 'segmented_diagnostic': {'eligible_as_primary': segmented_eligible, 'series_with_bic_support': segmented_bic_support, 'series_showing_acceleration_after_break': segmented_accelerations, 'role': 'secondary diagnostic for slope changes'}, 'family_years': family_years, 'step': step_results, 'segmented': segmented_results, 'bayesian': bayesian_results, 'multinomial_b_relevant': b_relevant_multinomial, 'multinomial_b_all_categories': b_full_multinomial, 'multinomial_c_relevant': c_relevant_multinomial, 'multinomial_c_all_categories': c_full_multinomial}

def leave_one_source_out(records: list[dict[str, str]]) -> list[dict]:
    sources = sorted({source for row in records for source in row['origins'].split('; ')})
    results = []
    for omitted in sources:
        selected = []
        for row in records:
            remaining_sources = [source for source in row['origins'].split('; ') if source != omitted]
            if remaining_sources:
                selected.append(row)
        (a_counts, b_counts, c_counts, _) = counts_from_records(selected)
        decision = decide_from_counts(a_counts, b_counts, c_counts)
        results.append({'omitted_source': omitted, 'remaining_records': len(selected), 'final_year': decision['final_year'], 'primary_year': decision['primary_year'], **decision['family_years']})
    return results

def source_only(records: list[dict[str, str]]) -> list[dict]:
    sources = sorted({source for row in records for source in row['origins'].split('; ')})
    results = []
    for source in sources:
        selected = [row for row in records if source in row['origins'].split('; ')]
        (a_counts, b_counts, c_counts, _) = counts_from_records(selected)
        decision = decide_from_counts(a_counts, b_counts, c_counts)
        results.append({'source': source, 'records': len(selected), 'final_year': decision['final_year'], 'primary_year': decision['primary_year'], **decision['family_years']})
    return results

def cohen_kappa(first: list[str], second: list[str]) -> float | None:
    if not first:
        return None
    agreement = sum((a == b for (a, b) in zip(first, second))) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum((first_counts[label] * second_counts[label] for label in ('INTRINSIC', 'EXTRINSIC'))) / len(first) ** 2
    if expected >= 1:
        return None
    return (agreement - expected) / (1 - expected)

def classifier_agreement(records: list[dict[str, str]]) -> dict:
    fields = {'method_a': 'v2_class', 'method_b': 'semantic_class', 'method_c': 'method_c_class'}
    pairwise = {}
    names = list(fields)
    allowed = {'INTRINSIC', 'EXTRINSIC'}
    for (index, first_name) in enumerate(names):
        for second_name in names[index + 1:]:
            first_field = fields[first_name]
            second_field = fields[second_name]
            selected = [row for row in records if row[first_field] in allowed and row[second_field] in allowed]
            first = [row[first_field] for row in selected]
            second = [row[second_field] for row in selected]
            pairwise[f'{first_name}__{second_name}'] = {'comparable_records': len(selected), 'agreements': sum((a == b for (a, b) in zip(first, second))), 'agreement_fraction': sum((a == b for (a, b) in zip(first, second))) / len(selected) if selected else None, 'cohen_kappa': cohen_kappa(first, second)}
    all_three = [row for row in records if all((row[field] in allowed for field in fields.values()))]
    return {'class_counts': {name: dict(Counter((row[field] for row in records))) for (name, field) in fields.items()}, 'pairwise_pure_categories': pairwise, 'all_three_comparable_records': len(all_three), 'all_three_same_label': sum((len({row[field] for field in fields.values()}) == 1 for row in all_three))}

def write_outputs(summary: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    with (OUTPUT_DIR / 'methods.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['family', 'series', 'year', 'compatible_years'])
        for family in ('step', 'segmented', 'bayesian'):
            for (series, result) in summary['decision'][family].items():
                support = result.get('support', result.get('support_95', []))
                writer.writerow([family, series, result['year'], ';'.join(map(str, support))])
        for name in ('multinomial_b_relevant', 'multinomial_b_all_categories', 'multinomial_c_relevant', 'multinomial_c_all_categories'):
            result = summary['decision'][name]
            series = 'method_b' if '_b_' in name else 'method_c'
            writer.writerow([name, series, result['year'], ';'.join(map(str, result['support_95']))])
    with (OUTPUT_DIR / 'posteriors.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['family', 'series', 'year', 'probability'])
        for (series, result) in summary['decision']['bayesian'].items():
            for (year, probability) in result['probabilities'].items():
                writer.writerow(['bayesian', series, year, probability])
        for name in ('multinomial_b_relevant', 'multinomial_b_all_categories', 'multinomial_c_relevant', 'multinomial_c_all_categories'):
            series = 'method_b' if '_b_' in name else 'method_c'
            for (year, probability) in summary['decision'][name]['probabilities'].items():
                writer.writerow([name, series, year, probability])
    with (OUTPUT_DIR / 'leave_one_source_out.csv').open('w', encoding='utf-8', newline='') as handle:
        fields = list(summary['leave_one_source_out'][0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary['leave_one_source_out'])
    with (OUTPUT_DIR / 'individual_sources.csv').open('w', encoding='utf-8', newline='') as handle:
        fields = list(summary['source_only'][0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary['source_only'])

def main() -> None:
    records = read_records()
    if not METHOD_C_FILE.exists():
        raise RuntimeError(f'Method C classifications missing: {METHOD_C_FILE}')
    with METHOD_C_FILE.open('r', encoding='utf-8-sig', newline='') as handle:
        method_c_rows = list(csv.DictReader(handle))
    method_c = {row['record_id']: row for row in method_c_rows}
    if len(method_c) != len(method_c_rows):
        raise RuntimeError('Method C contains duplicate identifiers')
    record_ids = {row['record_id'] for row in records}
    if set(method_c) != record_ids:
        raise RuntimeError(f'Incomplete Method C: {len(method_c)} of {len(records)} publications')
    mismatches = []
    for row in records:
        method_c_row = method_c[row['record_id']]
        if method_c_row['year'].strip() != row['year'].strip() or normalized_title(method_c_row['title']) != normalized_title(row['title']):
            mismatches.append(row['record_id'])
        row['method_c_class'] = method_c_row['method_c_class']
    if mismatches:
        raise RuntimeError(f'Methods B and C are misaligned; first identifiers: {mismatches[:5]}')
    (a_counts, b_counts, c_counts, totals) = counts_from_records(records)
    decision = decide_from_counts(a_counts, b_counts, c_counts)
    summary = {'scope': {'records_total': len(records), 'records_with_year': sum((bool(row['year'].strip()) for row in records)), 'analysis_window': [START_YEAR, END_YEAR], 'candidate_cutoffs': CANDIDATES, 'records_in_analysis_window': int(totals.sum()), 'cutoff_is_inclusive': True, 'classifiers': ['method_a', 'method_b', 'method_c'], 'cross_method_alignment_validated_by': 'record_id + normalized title + publication year'}, 'automatic_rule': {'primary_family': 'binomial_level_shift', 'validation_families': ['bayesian_change', 'multinomial_change'], 'agreement_tolerance_years': 1, 'minimum_close_validators': 2, 'fallback': 'median of the three decision families', 'segmented_regression_role': 'secondary diagnostic; primary eligibility requires improved BIC and acceleration in at least two series'}, 'decision': decision, 'leave_one_source_out': leave_one_source_out(records), 'source_only': source_only(records), 'classifier_agreement': classifier_agreement(records)}
    write_outputs(summary)
    print(json.dumps({'final_year': decision['final_year'], 'primary_year': decision['primary_year'], 'family_years': decision['family_years'], 'rule_applied': decision['rule_applied']}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
