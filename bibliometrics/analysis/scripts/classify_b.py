# Copyright 2026 Sergio Sánchez de la Fuente
# SPDX-License-Identifier: Apache-2.0
"""Classify world-model terminology using the first text model."""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from corpus import clean_value, normalize_for_match, parse_bib_entry, split_bib_entries
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'results'
CORPUS_PATH = OUT_DIR / 'classified_corpus.bib'
V2_PATH = OUT_DIR / 'method_a_classifications.csv'
FIG_DIR = ROOT / 'figures'
CHECKPOINT_PATH = OUT_DIR / 'method_b_checkpoint.jsonl'
MODEL_NAME = 'MoritzLaurer/deberta-v3-base-zeroshot-v2.0'
MODEL_REVISION = 'main'
MAX_LENGTH = 192
HYPOTHESES = {'INTRINSIC': 'This research uses a learned predictive internal world model that represents or predicts environment dynamics for an artificial agent or robot.', 'EXTRINSIC': 'This research uses an explicit external representation of the environment, such as a map, geometric scene, spatial database, knowledge base, or symbolic world model, for an artificial agent or robot.', 'COMBINED': 'The main contribution combines a learned predictive internal world model with an explicit external map, geometric scene, spatial database, knowledge base, or symbolic environment representation for an artificial agent or robot.', 'UNRELATED': 'The main subject of this research is something other than world models or environment representations used by artificial agents or robots.'}

def deescape_latex(value: str) -> str:
    value = clean_value(value)
    value = value.replace('\\_', '_').replace('\\&', '&').replace('~', ' ')
    value = re.sub('\\\\[a-zA-Z]+\\s*', ' ', value)
    value = value.replace('{', '').replace('}', '')
    return re.sub('\\s+', ' ', value).strip()

def load_records() -> list[dict[str, object]]:
    raw = CORPUS_PATH.read_text(encoding='utf-8-sig', errors='replace')
    records: list[dict[str, object]] = []
    for (position, entry) in enumerate(split_bib_entries(raw), 1):
        (_, key, fields) = parse_bib_entry(entry)
        record_id = key.replace('WM_', 'WM')
        title = deescape_latex(fields.get('title', ''))
        abstract = deescape_latex(fields.get('abstract', ''))
        keywords = deescape_latex(fields.get('keywords', ''))
        year_match = re.search('\\b(?:19|20)\\d{2}\\b', fields.get('year', ''))
        year = int(year_match.group()) if year_match else None
        premise = f'Title: {title}. Keywords: {keywords}. Abstract: {abstract}'
        records.append({'record_id': record_id, 'position': position, 'year': year, 'title': title, 'abstract_present': bool(abstract), 'keywords_present': bool(keywords), 'premise': premise})
    return records

def load_v2() -> dict[str, dict[str, str]]:
    with V2_PATH.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row['record_id']: row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError('Duplicate record identifiers in Method A output')
    return indexed

def normalized_title(value: str) -> str:
    return normalize_for_match(deescape_latex(value))

def validate_and_align_reused(current: list[dict[str, object]], saved: list[dict[str, object]], *, label: str) -> list[dict[str, object]]:
    """Reject stale semantic rows instead of silently joining them by position/ID."""
    saved_by_id = {str(row['record_id']): row for row in saved}
    if len(saved_by_id) != len(saved):
        raise RuntimeError(f'Duplicate record identifiers in saved {label} output')
    current_ids = {str(row['record_id']) for row in current}
    if set(saved_by_id) != current_ids:
        missing = sorted(current_ids - set(saved_by_id))[:5]
        extra = sorted(set(saved_by_id) - current_ids)[:5]
        raise RuntimeError(f'Saved {label} output belongs to a different corpus: missing={missing}, extra={extra}')
    aligned: list[dict[str, object]] = []
    mismatches: list[str] = []
    for reference in current:
        record_id = str(reference['record_id'])
        row = saved_by_id[record_id]
        checks = {'position': int(row['position']) == int(reference['position']), 'year': row['year'] == reference['year'], 'title': normalized_title(str(row['title'])) == normalized_title(str(reference['title'])), 'abstract_present': row['abstract_present'] == reference['abstract_present'], 'keywords_present': row['keywords_present'] == reference['keywords_present']}
        failed = [name for (name, valid) in checks.items() if not valid]
        if failed:
            mismatches.append(f"{record_id}: {', '.join(failed)}")
        aligned.append(row)
    if mismatches:
        raise RuntimeError(f'Saved {label} output is misaligned with the current corpus; first mismatches: {mismatches[:5]}')
    return aligned

def validate_method_a(current: list[dict[str, object]], v2: dict[str, dict[str, str]]) -> None:
    current_ids = {str(row['record_id']) for row in current}
    if set(v2) != current_ids:
        raise RuntimeError('Method A identifiers do not match the current corpus')
    mismatches = []
    for reference in current:
        record_id = str(reference['record_id'])
        row = v2[record_id]
        year = int(row['year']) if row['year'] else None
        if year != reference['year'] or normalized_title(row['title']) != normalized_title(str(reference['title'])):
            mismatches.append(record_id)
    if mismatches:
        raise RuntimeError(f'Method A is misaligned with the current corpus; first identifiers: {mismatches[:5]}')

def label_ids(model) -> tuple[int, int]:
    labels = {int(idx): str(label).casefold() for (idx, label) in model.config.id2label.items()}
    entailment = next((idx for (idx, label) in labels.items() if 'entail' in label and 'not' not in label))
    other = next((idx for idx in labels if idx != entailment))
    return (entailment, other)

def semantic_label(scores: dict[str, float]) -> tuple[str, str, float, float]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ((top_label, top_score), (_, second_score)) = ranked[:2]
    margin = top_score - second_score
    if top_score < 0.4 or margin < 0.1:
        return ('AMBIGUOUS', 'LOW', top_score, margin)
    confidence = 'HIGH' if top_score >= 0.65 and margin >= 0.3 else 'MEDIUM'
    return (top_label, confidence, top_score, margin)

def infer(records: list[dict[str, object]]) -> list[dict[str, object]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).to(device)
    if device.type == 'cuda':
        model = model.half()
    model.eval()
    (entailment_id, _) = label_ids(model)
    labels = list(HYPOTHESES)
    hypotheses = [HYPOTHESES[label] for label in labels]
    batch_size = 32 if device.type == 'cuda' else 4
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, object]] = []
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open('r', encoding='utf-8') as handle:
            completed = [json.loads(line) for line in handle if line.strip()]
        print(f'Resuming from {len(completed):,} records', flush=True)
    next_report = (len(completed) // 1000 + 1) * 1000
    for start in range(len(completed), len(records), batch_size):
        batch = records[start:start + batch_size]
        premises = [str(row['premise']) for row in batch for _ in labels]
        paired_hypotheses = hypotheses * len(batch)
        tokens = tokenizer(premises, paired_hypotheses, padding=True, truncation='only_first', max_length=MAX_LENGTH, return_tensors='pt').to(device)
        with torch.inference_mode():
            logits = model(**tokens).logits.float()
        entailment_logits = logits[:, entailment_id].reshape(len(batch), len(labels))
        support = torch.softmax(entailment_logits, dim=1).flatten().cpu().tolist()
        batch_results = []
        for (offset, row) in enumerate(batch):
            values = support[offset * len(labels):(offset + 1) * len(labels)]
            scores = dict(zip(labels, values))
            (label, confidence, top_score, margin) = semantic_label(scores)
            batch_results.append({**{k: v for (k, v) in row.items() if k != 'premise'}, 'semantic_class': label, 'semantic_confidence': confidence, 'score_intrinsic': scores['INTRINSIC'], 'score_extrinsic': scores['EXTRINSIC'], 'score_combined': scores['COMBINED'], 'score_unrelated': scores['UNRELATED'], 'top_score': top_score, 'score_margin': margin})
        completed.extend(batch_results)
        with CHECKPOINT_PATH.open('a', encoding='utf-8') as handle:
            for result in batch_results:
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
        if len(completed) >= next_report or len(completed) == len(records):
            print(f'Analizados {start + len(batch):,}/{len(records):,}', flush=True)
            next_report += 1000
    return completed

def sustained_crossing(rows: list[dict[str, object]], *, include_both: bool=False) -> int | None:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        year = row['year']
        label = row['semantic_class']
        allowed = {'INTRINSIC', 'EXTRINSIC', 'COMBINED'} if include_both else {'INTRINSIC', 'EXTRINSIC'}
        if isinstance(year, int) and label in allowed:
            counts[year][str(label)] += 1
    eligible = []
    for year in sorted(counts):
        total = counts[year]['INTRINSIC'] + counts[year]['EXTRINSIC']
        numerator = counts[year]['INTRINSIC']
        if include_both:
            total += counts[year]['COMBINED']
            numerator += counts[year]['COMBINED']
        if total >= 10:
            eligible.append((year, numerator / total))
    proportions = dict(eligible)
    for (year, value) in eligible:
        if value > 0.5 and all((proportions.get(year + step, 0) > 0.5 for step in (1, 2))):
            return year
    return None

def write_outputs(rows: list[dict[str, object]], v2: dict[str, dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined: list[dict[str, object]] = []
    for row in rows:
        old = v2[str(row['record_id'])]
        combined.append({**row, 'origins': old['origins'], 'v2_class': old['wm_class'], 'v2_evidence': old['wm_evidence'], 'v2_confidence': old['wm_confidence']})
    output_fields = list(combined[0])
    with (OUT_DIR / 'method_b_classifications.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(combined)
    semantic_counts = Counter((str(row['semantic_class']) for row in combined))
    v2_counts = Counter((str(row['v2_class']) for row in combined))
    cross = Counter(((str(row['v2_class']), str(row['semantic_class'])) for row in combined))
    v2_definitive = [row for row in combined if row['v2_class'] in {'INTRINSIC', 'EXTRINSIC'}]
    comparable = [row for row in v2_definitive if row['semantic_class'] in {'INTRINSIC', 'EXTRINSIC'}]
    agreements = sum((row['v2_class'] == row['semantic_class'] for row in comparable))
    years = sorted({int(row['year']) for row in combined if isinstance(row['year'], int)})
    annual_rows = []
    for year in years:
        selected = [row for row in combined if row['year'] == year]
        sem = Counter((str(row['semantic_class']) for row in selected))
        old = Counter((str(row['v2_class']) for row in selected))
        sem_denom = sem['INTRINSIC'] + sem['EXTRINSIC']
        sem_presence_denom = sem_denom + sem['COMBINED']
        v2_denom = old['INTRINSIC'] + old['EXTRINSIC']
        annual_rows.append({'year': year, 'total': len(selected), 'semantic_intrinsic': sem['INTRINSIC'], 'semantic_extrinsic': sem['EXTRINSIC'], 'semantic_both': sem['COMBINED'], 'semantic_ambiguous': sem['AMBIGUOUS'], 'semantic_unrelated': sem['UNRELATED'], 'semantic_intrinsic_pct': 100 * sem['INTRINSIC'] / sem_denom if sem_denom else '', 'semantic_intrinsic_presence_pct': 100 * (sem['INTRINSIC'] + sem['COMBINED']) / sem_presence_denom if sem_presence_denom else '', 'v2_intrinsic': old['INTRINSIC'], 'v2_extrinsic': old['EXTRINSIC'], 'v2_ambiguous': old['AMBIGUOUS'], 'v2_intrinsic_pct': 100 * old['INTRINSIC'] / v2_denom if v2_denom else ''})
    pd.DataFrame(annual_rows).to_csv(OUT_DIR / 'method_b_annual.csv', index=False, encoding='utf-8')
    summary = {'scope': {'deduplicated_records': len(combined), 'records_without_year': sum((row['year'] is None for row in combined)), 'content_available': 'title + abstract + keywords when present; no article full text'}, 'independent_model': {'name': MODEL_NAME, 'revision': MODEL_REVISION, 'method': 'zero-shot natural-language inference; v2 labels hidden during inference', 'max_tokens': MAX_LENGTH, 'hypotheses': HYPOTHESES, 'class_counts': dict(semantic_counts), 'strict_pure_intrinsic_crossing': sustained_crossing(combined), 'intrinsic_presence_crossing_including_hybrids': sustained_crossing(combined, include_both=True)}, 'v2': {'class_counts': dict(v2_counts)}, 'comparison': {'cross_counts': {f'{a}__TO__{b}': n for ((a, b), n) in sorted(cross.items())}, 'v2_definitive_records': len(v2_definitive), 'comparable_intrinsic_extrinsic_records': len(comparable), 'same_intrinsic_extrinsic_label': agreements, 'agreement_fraction_when_both_definitive': agreements / len(comparable) if comparable else None, 'v2_ambiguous_resolved_semantically': sum((row['v2_class'] == 'AMBIGUOUS' and row['semantic_class'] in {'INTRINSIC', 'EXTRINSIC', 'UNRELATED'} for row in combined))}}
    (OUT_DIR / 'method_b_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-existing', action='store_true', help='Regenerate summaries from the completed semantic CSV.')
    args = parser.parse_args()
    current_records = load_records()
    v2 = load_v2()
    validate_method_a(current_records, v2)
    existing_path = OUT_DIR / 'method_b_classifications.csv'
    if args.reuse_existing and existing_path.exists():
        with existing_path.open('r', encoding='utf-8-sig', newline='') as handle:
            records = list(csv.DictReader(handle))
        for row in records:
            row['position'] = int(row['position'])
            row['year'] = int(row['year']) if row['year'] else None
            row['abstract_present'] = row['abstract_present'] == 'True'
            row['keywords_present'] = row['keywords_present'] == 'True'
            for field in ('score_intrinsic', 'score_extrinsic', 'score_combined', 'score_unrelated', 'top_score', 'score_margin'):
                row[field] = float(row[field])
        records = validate_and_align_reused(current_records, records, label='Method B')
    else:
        records = current_records
    if len(records) != 18109 or len(v2) != len(records):
        raise RuntimeError(f'Unexpected corpus sizes: semantic={len(records)}, v2={len(v2)}')
    missing = {str(row['record_id']) for row in records} - set(v2)
    if missing:
        raise RuntimeError(f'Records absent from v2 audit: {sorted(missing)[:5]}')
    inferred = records if args.reuse_existing else infer(records)
    write_outputs(inferred, v2)
    if not args.reuse_existing and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    print(f'Results written to {OUT_DIR}')
if __name__ == '__main__':
    main()
