# Copyright 2026 Sergio Sánchez de la Fuente
# SPDX-License-Identifier: Apache-2.0
"""Classify world-model terminology using alternative hypotheses and a relevance filter."""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from classify_b import load_records, validate_and_align_reused
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
MODEL_NAME = 'cross-encoder/nli-deberta-v3-small'
MAX_LENGTH = 192
BATCH_SIZE = 16
CHECKPOINT = RESULTS_DIR / 'method_c_checkpoint.jsonl'
OUTPUT = RESULTS_DIR / 'method_c_classifications.csv'
HYPOTHESES = {'INTRINSIC': ['This research uses a learned predictive internal world model for an artificial agent or robot.', 'The agent learns a model from data to predict future states, environment dynamics, or consequences of actions.'], 'EXTRINSIC': ['This research uses an explicit map, geometric scene, spatial database, knowledge base, or symbolic representation of the environment.', 'The world representation is an explicit external model of the environment used by an artificial agent or robot.'], 'COMBINED': ['This research combines a learned predictive model with an explicit map or environment representation.', 'The artificial agent uses both learned environment dynamics and an explicit spatial, geometric, or symbolic representation.']}
RELEVANCE_HYPOTHESES = ['The main topic is world models or environment representations used by artificial agents or robots.', 'This paper studies how a robot or artificial agent represents or predicts its environment.']

def nli_label_ids(model) -> tuple[int, int]:
    labels = {int(index): str(label).casefold() for (index, label) in model.config.id2label.items()}
    entailment = next((index for (index, label) in labels.items() if 'entail' in label and 'not' not in label))
    contradiction = next((index for (index, label) in labels.items() if 'contrad' in label))
    return (entailment, contradiction)

def choose_label(scores: dict[str, float], relevance: float) -> tuple[str, str, float, float]:
    if relevance < 0.5:
        confidence = 'HIGH' if relevance < 0.25 else 'MEDIUM'
        return ('UNRELATED', confidence, 1.0 - relevance, abs(0.5 - relevance) * 2)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ((top_label, top_score), (_, second_score)) = ranked[:2]
    margin = top_score - second_score
    if top_score < 0.4 or margin < 0.1:
        return ('AMBIGUOUS', 'LOW', top_score, margin)
    confidence = 'HIGH' if top_score >= 0.65 and margin >= 0.3 else 'MEDIUM'
    return (top_label, confidence, top_score, margin)

def load_origins() -> dict[str, str]:
    with (RESULTS_DIR / 'method_b_classifications.csv').open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    origins = {row['record_id']: row['origins'] for row in rows}
    if len(origins) != len(rows):
        raise RuntimeError('Duplicate record identifiers in Method B output')
    return origins

def infer(records: list[dict[str, object]]) -> list[dict[str, object]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, local_files_only=True).to(device)
    if device.type == 'cuda':
        model = model.half()
    model.eval()
    (entailment, contradiction) = nli_label_ids(model)
    labels = list(HYPOTHESES)
    prompts_per_label = len(next(iter(HYPOTHESES.values())))
    completed: list[dict[str, object]] = []
    if CHECKPOINT.exists():
        with CHECKPOINT.open('r', encoding='utf-8') as handle:
            completed = [json.loads(line) for line in handle if line.strip()]
        print(f'Resuming from {len(completed):,} publications', flush=True)
    next_report = (len(completed) // 1000 + 1) * 1000
    for start in range(len(completed), len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        premises: list[str] = []
        hypotheses: list[str] = []
        for row in batch:
            for label in labels:
                for hypothesis in HYPOTHESES[label]:
                    premises.append(str(row['premise']))
                    hypotheses.append(hypothesis)
            for hypothesis in RELEVANCE_HYPOTHESES:
                premises.append(str(row['premise']))
                hypotheses.append(hypothesis)
        tokens = tokenizer(premises, hypotheses, padding=True, truncation='only_first', max_length=MAX_LENGTH, return_tensors='pt').to(device)
        with torch.inference_mode():
            logits = model(**tokens).logits.float()
        pairs_per_record = len(labels) * prompts_per_label + len(RELEVANCE_HYPOTHESES)
        logits = logits.reshape(len(batch), pairs_per_record, -1)
        class_logits = logits[:, :len(labels) * prompts_per_label, entailment]
        class_logits = class_logits.reshape(len(batch), len(labels), prompts_per_label)
        averaged_logits = class_logits.mean(dim=2)
        probabilities = torch.softmax(averaged_logits, dim=1).cpu().tolist()
        relevance_logits = logits[:, len(labels) * prompts_per_label:, :]
        relevance_pair_logits = relevance_logits[:, :, [contradiction, entailment]]
        relevance_scores = torch.softmax(relevance_pair_logits, dim=2)[:, :, 1].mean(dim=1).cpu().tolist()
        batch_results = []
        for (row, values, relevance) in zip(batch, probabilities, relevance_scores):
            scores = dict(zip(labels, values))
            (label, confidence, top_score, margin) = choose_label(scores, relevance)
            batch_results.append({'record_id': row['record_id'], 'position': row['position'], 'year': row['year'], 'title': row['title'], 'abstract_present': row['abstract_present'], 'keywords_present': row['keywords_present'], 'method_c_class': label, 'method_c_confidence': confidence, 'score_intrinsic': scores['INTRINSIC'], 'score_extrinsic': scores['EXTRINSIC'], 'score_combined': scores['COMBINED'], 'score_unrelated': 1.0 - relevance, 'relevance_score': relevance, 'top_score': top_score, 'score_margin': margin})
        completed.extend(batch_results)
        with CHECKPOINT.open('a', encoding='utf-8') as handle:
            for result in batch_results:
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
        if len(completed) >= next_report or len(completed) == len(records):
            print(f'Analizadas {len(completed):,}/{len(records):,} publications', flush=True)
            next_report += 1000
    return completed

def write_outputs(rows: list[dict[str, object]], origins: dict[str, str]) -> None:
    fields = [field for field in rows[0] if field != 'origins'] + ['origins']
    with OUTPUT.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'origins': origins[str(row['record_id'])]})
    annual: dict[int, Counter[str]] = defaultdict(Counter)
    totals = Counter((str(row['method_c_class']) for row in rows))
    for row in rows:
        if isinstance(row['year'], int):
            annual[int(row['year'])][str(row['method_c_class'])] += 1
    with (RESULTS_DIR / 'method_c_annual.csv').open('w', encoding='utf-8', newline='') as handle:
        fields = ['year', 'total', 'learned_internal', 'explicit_external', 'combined', 'ambiguous', 'out_of_scope']
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for year in sorted(annual):
            counts = annual[year]
            writer.writerow({'year': year, 'total': sum(counts.values()), 'learned_internal': counts['INTRINSIC'], 'explicit_external': counts['EXTRINSIC'], 'combined': counts['COMBINED'], 'ambiguous': counts['AMBIGUOUS'], 'out_of_scope': counts['UNRELATED']})
    summary = {'model': MODEL_NAME, 'local_files_only': True, 'records': len(rows), 'max_tokens': MAX_LENGTH, 'hypotheses_per_category': 2, 'relevance_gate': {'threshold': 0.5, 'hypotheses': RELEVANCE_HYPOTHESES, 'uses_entailment_vs_contradiction': True}, 'labels_from_methods_a_or_b_used_as_input': False, 'class_counts': dict(totals), 'hypotheses': HYPOTHESES}
    (RESULTS_DIR / 'method_c_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-existing', action='store_true', help='Validate and reuse the completed Method C CSV without model inference.')
    args = parser.parse_args()
    records = load_records()
    if len(records) != 18109:
        raise RuntimeError(f'Unexpected number of publications: {len(records)}')
    origins = load_origins()
    if args.reuse_existing and OUTPUT.exists():
        with OUTPUT.open('r', encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row.pop('origins', None)
            row['position'] = int(row['position'])
            row['year'] = int(row['year']) if row['year'] else None
            row['abstract_present'] = row['abstract_present'] == 'True'
            row['keywords_present'] = row['keywords_present'] == 'True'
            for field in ('score_intrinsic', 'score_extrinsic', 'score_combined', 'score_unrelated', 'relevance_score', 'top_score', 'score_margin'):
                row[field] = float(row[field])
        rows = validate_and_align_reused(records, rows, label='Method C')
    else:
        rows = infer(records)
    write_outputs(rows, origins)
    if not args.reuse_existing and CHECKPOINT.exists():
        CHECKPOINT.unlink()
    print(f'Results written to {OUTPUT}')
if __name__ == '__main__':
    main()
