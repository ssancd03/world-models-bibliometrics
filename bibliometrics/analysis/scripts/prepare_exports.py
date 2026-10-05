# Copyright 2026 Sergio Sánchez de la Fuente
# SPDX-License-Identifier: Apache-2.0
"""Consolidate overlapping bibliographic exports into analysis inputs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[1]
DOWNLOADS_DIR = BASE_DIR / 'data/raw'
WORKING_DIR = BASE_DIR / 'data/inputs'

def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        return str(path.resolve())

def load_pipeline_module():
    path = SCRIPT_DIR / 'corpus.py'
    spec = importlib.util.spec_from_file_location('wm_pipeline', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def acm_identity(entry: str, pipeline) -> tuple[str, str]:
    (_, key, fields) = pipeline.parse_bib_entry(entry)
    doi = pipeline.normalize_doi(fields.get('doi', ''))
    if not doi:
        doi = pipeline.normalize_doi(fields.get('url', ''))
    if doi:
        return ('doi', doi)
    stable_id = pipeline.clean_value(key).casefold()
    if not stable_id:
        raise RuntimeError('ACM entry without DOI or stable BibTeX identifier')
    return ('acm_id', stable_id)

def prepare_acm(pipeline) -> dict[str, object]:
    inputs = [DOWNLOADS_DIR / 'acm_1962_2010_n947.bib', DOWNLOADS_DIR / 'acm_filter_2011_2020_n576.bib', DOWNLOADS_DIR / 'acm_2021_2024_n564.bib', DOWNLOADS_DIR / 'acm.bib']
    blocks: list[str] = []
    input_counts: dict[str, int] = {}
    for path in inputs:
        raw_blocks = pipeline.split_bib_entries(path.read_text(encoding='utf-8-sig', errors='replace'))
        input_counts[path.name] = len(raw_blocks)
        blocks.extend(raw_blocks)
    selected: dict[tuple[str, str], str] = {}
    for block in blocks:
        selected.setdefault(acm_identity(block, pipeline), block.strip())
    if len(selected) != 2581:
        raise RuntimeError(f'Expected 2581 unique ACM records, found {len(selected)}')
    output = WORKING_DIR / 'acm.bib'
    output.write_text('\n\n'.join(selected.values()) + '\n', encoding='utf-8')
    return {'input_files': [display_path(path) for path in inputs], 'input_records': input_counts, 'raw_downloaded_records': len(blocks), 'exact_overlap_removed': len(blocks) - len(selected), 'working_records': len(selected), 'working_file': display_path(output), 'sha256': sha256(output), 'identity_rule': 'normalized DOI; otherwise stable ACM BibTeX identifier'}

def springer_doi(row: dict[str, str]) -> str:
    value = (row.get('Item DOI') or '').strip().casefold()
    for prefix in ('https://doi.org/', 'http://doi.org/', 'doi:'):
        if value.startswith(prefix):
            value = value[len(prefix):]
    value = value.rstrip(' .')
    return value if re.fullmatch('10\\.\\d{4,9}/\\S+', value) else ''

def springer_identity(row: dict[str, str]) -> str:
    doi = springer_doi(row)
    if doi:
        return f'doi:{doi}'
    url = (row.get('URL') or '').strip().casefold()
    if url:
        return f'url:{url}'
    title = (row.get('Item Title') or '').strip().casefold()
    year = (row.get('Publication Year') or '').strip()
    if title:
        return f'title_year:{title}|{year}'
    raise RuntimeError(f'Springer row without stable identity: {row}')

def read_springer(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f'Missing CSV header in {path}')
        return (list(reader.fieldnames), list(reader))

def prepare_springer() -> dict[str, object]:
    dated_inputs = [DOWNLOADS_DIR / 'springer_1963_1995_n718.csv', DOWNLOADS_DIR / 'springer_1996_1999_n460.csv', DOWNLOADS_DIR / 'springer_2000_2006_n849.csv', DOWNLOADS_DIR / 'springer_2007_2009_n451.csv', DOWNLOADS_DIR / 'springer_2010_2015_n839.csv', DOWNLOADS_DIR / 'springer_2016_2019_n538.csv', DOWNLOADS_DIR / 'springer_2020_2023_n632.csv', DOWNLOADS_DIR / 'springer_2024_2027_n780.csv']
    working = WORKING_DIR / 'springer.csv'
    original_truncated = DOWNLOADS_DIR / 'SearchResults.csv'
    fieldnames: list[str] | None = None
    selected: dict[str, dict[str, str]] = {}
    input_counts: dict[str, int] = {}
    for path in dated_inputs + [original_truncated]:
        (header, rows) = read_springer(path)
        if fieldnames is None:
            fieldnames = header
        elif header != fieldnames:
            raise RuntimeError(f'Incompatible Springer CSV header in {path}')
        input_counts[path.name] = len(rows)
        for row in rows:
            selected.setdefault(springer_identity(row), row)
    if len(selected) != 5278:
        raise RuntimeError(f'Expected 5278 recoverable Springer records, found {len(selected)}')
    assert fieldnames is not None
    with working.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(selected.values())
    dated_records = sum((input_counts[path.name] for path in dated_inputs))
    undated = sum((not (row.get('Publication Year') or '').strip() for row in selected.values()))
    return {'input_files': [display_path(path) for path in dated_inputs] + [display_path(original_truncated)], 'input_records': input_counts, 'dated_slice_records': dated_records, 'records_added_from_initial_export': len(selected) - dated_records, 'working_records': len(selected), 'platform_result_count': 5279, 'unretrieved_records': 1, 'records_without_year': undated, 'working_file': display_path(working), 'sha256': sha256(working), 'identity_rule': 'validated normalized DOI; otherwise stable URL (one exported Collection record)'}

def prepare_sciencedirect(pipeline) -> dict[str, object]:
    inputs = [DOWNLOADS_DIR / 'ScienceDirect_citations_1785485241727.bib', DOWNLOADS_DIR / 'ScienceDirect_citations_1785485254422.bib']
    outputs = [WORKING_DIR / 'science_direct_1.bib', WORKING_DIR / 'science_direct_2.bib']
    counts: dict[str, int] = {}
    for (source, output) in zip(inputs, outputs):
        shutil.copy2(source, output)
        count = len(pipeline.split_bib_entries(output.read_text(encoding='utf-8-sig', errors='replace')))
        counts[output.name] = count
    if sum(counts.values()) != 191:
        raise RuntimeError(f'Expected 191 ScienceDirect records, found {sum(counts.values())}')
    return {'input_files': [display_path(path) for path in inputs], 'working_files': [display_path(path) for path in outputs], 'working_records': sum(counts.values()), 'records_by_file': counts, 'platform_result_count': 191, 'sha256': {path.name: sha256(path) for path in outputs}}

def prepare_direct_exports() -> dict[str, object]:
    files = {'scopus': ('scopus_export_Jul 30-2026_6256a535-d020-49cc-a6cf-09dbeb4af854.bib', 'scopus.bib'), 'ieee': ('export2026.07.30-06.36.15.csv', 'ieee.csv'), 'web_of_science': ('savedrecs.bib', 'web_of_science.bib')}
    result: dict[str, object] = {}
    for (source, (original_name, working_name)) in files.items():
        original = DOWNLOADS_DIR / original_name
        working = WORKING_DIR / working_name
        shutil.copy2(original, working)
        result[source] = {'original_file': display_path(original), 'working_file': display_path(working), 'sha256': sha256(working)}
    return result

def main() -> None:
    global DOWNLOADS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--downloads-dir', type=Path, default=DOWNLOADS_DIR, help='Directory containing the dated ACM/Springer/ScienceDirect exports (default: data/raw in the study directory)')
    args = parser.parse_args()
    DOWNLOADS_DIR = args.downloads_dir.resolve()
    if not DOWNLOADS_DIR.is_dir():
        raise SystemExit(f'Downloads directory not found: {DOWNLOADS_DIR}')
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline_module()
    result = {'policy': 'Original downloads unchanged; analysis uses only data/inputs', 'originals_directory': display_path(DOWNLOADS_DIR), 'working_directory': display_path(WORKING_DIR), 'direct_exports': prepare_direct_exports(), 'acm': prepare_acm(pipeline), 'springer': prepare_springer(), 'sciencedirect': prepare_sciencedirect(pipeline)}
    manifest_text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    audit_manifest = BASE_DIR / 'analysis' / 'results' / 'input_provenance.json'
    audit_manifest.parent.mkdir(parents=True, exist_ok=True)
    audit_manifest.write_text(manifest_text, encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
