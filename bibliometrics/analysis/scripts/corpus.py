# Copyright 2026 Sergio Sánchez de la Fuente
# SPDX-License-Identifier: Apache-2.0
"""Parse, deduplicate, and classify the bibliometric corpus."""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from rules import CLASSIFIER_VERSION, Classification, classify as classify_contextual, configuration as classifier_configuration
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / 'data/inputs'
OUT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = OUT_DIR / 'results'
CORPUS_DIR = RESULTS_DIR
AUDIT_DIR = RESULTS_DIR
FIG_DIR = OUT_DIR / 'figures'
EXPECTED_COUNTS = {'Scopus': 10817, 'SpringerLink': 5279, 'ACM Digital Library': 2581, 'Web of Science': 819, 'IEEE Xplore': 702, 'ScienceDirect': 191}
SOURCE_ORDER = ['Scopus', 'SpringerLink', 'ACM Digital Library', 'Web of Science', 'IEEE Xplore', 'ScienceDirect']
INPUT_SOURCE_ORDER = ['ACM Digital Library', 'IEEE Xplore', 'Web of Science', 'ScienceDirect', 'Scopus', 'SpringerLink']
SOURCE_PREFIX = {'Scopus': 'SCOPUS', 'SpringerLink': 'SPRINGER', 'ACM Digital Library': 'ACM', 'Web of Science': 'WOS', 'IEEE Xplore': 'IEEE', 'ScienceDirect': 'SD'}
STANDARD_FIELDS = ['author', 'editor', 'title', 'year', 'journal', 'booktitle', 'publisher', 'address', 'volume', 'number', 'pages', 'month', 'edition', 'series', 'doi', 'url', 'isbn', 'issn', 'language', 'abstract', 'keywords', 'note']
BASE_INTRINSIC = ['learned', 'latent', 'predictive', 'generative', 'neural', 'recurrent state space', 'RSSM', 'model-based reinforcement learning', 'imagination', 'rollout', 'dreamer', 'action-conditioned', 'variational', 'transformer', 'dynamics model', 'next-state prediction', 'video prediction', 'self-supervised', 'foundation model', 'latent dynamics']
BASE_EXTRINSIC = ['occupancy grid', 'geometric map', 'metric map', 'topological map', 'symbolic', 'blackboard', 'CAD model', 'a priori map', 'map building', 'sensor fusion', 'SLAM', 'world modeling module', 'environment representation', 'spatial database']
ADDED_INTRINSIC = ['MBRL', 'forward model']
ADDED_EXTRINSIC = ['semantic map', '3D map', 'OctoMap', 'scene graph', 'knowledge graph', 'knowledge base']
KEY_TERM_SERIES = ['latent', 'predictive', 'dreamer', 'model-based reinforcement learning', 'RSSM', 'generative', 'occupancy grid', 'SLAM']
BIB_ENTRY_RE = re.compile('(?im)^\\s*@(article|book|inbook|incollection|inproceedings|conference|proceedings|phdthesis|mastersthesis|techreport|misc|unpublished)\\s*[{(]')

@dataclass
class Record:
    source: str
    input_file: str
    input_index: int
    entry_type: str
    original_key: str
    fields: dict[str, str]
    raw_fields: set[str] = field(default_factory=set)
    raw_id: str = ''
    doi_normalized: str = ''
    doi_origin: str = ''
    title_normalized: str = ''
    year_int: int | None = None
    wm_class: str = ''
    wm_evidence: str = ''
    wm_confidence: str = ''
    wm_classifier_version: str = ''
    intrinsic_hits: list[str] = field(default_factory=list)
    extrinsic_hits: list[str] = field(default_factory=list)
    weak_intrinsic_hits: list[str] = field(default_factory=list)
    weak_extrinsic_hits: list[str] = field(default_factory=list)
    negative_hits: list[str] = field(default_factory=list)

class UnionFind:

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.members = [{i} for i in range(n)]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        (ra, rb) = (self.find(a), self.find(b))
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            (ra, rb) = (rb, ra)
        self.parent[rb] = ra
        self.members[ra].update(self.members[rb])
        self.members[rb].clear()
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra

def ensure_dirs() -> None:
    for directory in [RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def detect_source(path: Path) -> str | None:
    name = path.name.lower()
    if name.startswith('scopus') and path.suffix.lower() == '.bib':
        return 'Scopus'
    if name.startswith('web_of_science') and path.suffix.lower() == '.bib':
        return 'Web of Science'
    if name.startswith('acm') and path.suffix.lower() == '.bib':
        return 'ACM Digital Library'
    if name.startswith('science_direct'):
        return 'ScienceDirect'
    if path.suffix.lower() == '.csv':
        with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as handle:
            header = handle.readline()
        if 'Document Title' in header and 'Date Added To Xplore' in header:
            return 'IEEE Xplore'
        if 'Item Title' in header and 'Publication Title' in header:
            return 'SpringerLink'
    return None

def input_files() -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for path in sorted(INPUT_DIR.iterdir()):
        if not path.is_file():
            continue
        source = detect_source(path)
        if source:
            found.append((path, source))
    rank = {source: index for (index, source) in enumerate(INPUT_SOURCE_ORDER)}
    return sorted(found, key=lambda item: (rank[item[1]], item[0].name.casefold()))

def split_bib_entries(raw: str) -> list[str]:
    starts = list(BIB_ENTRY_RE.finditer(raw))
    return [raw[m.start():starts[i + 1].start() if i + 1 < len(starts) else len(raw)] for (i, m) in enumerate(starts)]

def parse_bib_entry(text: str) -> tuple[str, str, dict[str, str]]:
    """Tolerant parser for brace/quote-delimited BibTeX fields.

    Escaped braces are treated as text.  This recovers entries whose citation
    keys contain spaces and WoS abstracts containing literal LaTeX braces.
    """
    text = text.replace('US{\\}3.7 trillion', 'US$3.7 trillion')
    text = text.replace('\\{i.e., subspaces', '(i.e., subspaces')
    head = re.match('(?is)^\\s*@([a-z]+)\\s*[{(]\\s*([^,]+),', text)
    if not head:
        raise ValueError('BibTeX entry header not recognised')
    entry_type = head.group(1).lower()
    key = head.group(2).strip()
    pos = head.end()
    fields: dict[str, str] = {}
    length = len(text)
    while pos < length:
        while pos < length and (text[pos].isspace() or text[pos] == ','):
            pos += 1
        if pos >= length or text[pos] in '})':
            break
        name_start = pos
        while pos < length and text[pos] not in '=,}\n\r':
            pos += 1
        name = text[name_start:pos].strip().lower()
        if not name or pos >= length or text[pos] != '=':
            next_line = text.find('\n', pos)
            pos = length if next_line < 0 else next_line + 1
            continue
        pos += 1
        while pos < length and text[pos].isspace():
            pos += 1
        if pos >= length:
            fields[name] = ''
            break
        if text[pos] == '{':
            pos += 1
            start = pos
            depth = 1
            chunks: list[str] = []
            chunk_start = pos
            while pos < length and depth:
                char = text[pos]
                if char == '\\' and pos + 1 < length:
                    pos += 2
                    continue
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        chunks.append(text[chunk_start:pos])
                        pos += 1
                        break
                pos += 1
            if depth:
                raise ValueError(f'Unclosed braced value for field {name}')
            value = ''.join(chunks)
        elif text[pos] == '"':
            pos += 1
            chunks = []
            while pos < length:
                char = text[pos]
                if char == '\\' and pos + 1 < length:
                    chunks.append(text[pos:pos + 2])
                    pos += 2
                elif char == '"':
                    pos += 1
                    break
                else:
                    chunks.append(char)
                    pos += 1
            value = ''.join(chunks)
        else:
            start = pos
            while pos < length and text[pos] not in ',\n\r}':
                pos += 1
            value = text[start:pos].strip()
        fields[name] = value
    return (entry_type, key, fields)
TEX_SIMPLE = {'\\&': '&', '\\%': '%', '\\_': '_', '\\#': '#', '\\$': '$', '\\{': '{', '\\}': '}', '\\textbackslash{}': '\\', '\\textendash': '–', '\\textemdash': '—', '\\textquotesingle': "'", '\\textquotedbl': '"', '\\textless': '<', '\\textgreater': '>', '\\~{}': '~'}
ACCENT_COMBINING = {"'": '́', '`': '̀', '"': '̈', '^': '̂', '~': '̃', '=': '̄', '.': '̇', 'c': '̧', 'v': '̌', 'u': '̆', 'H': '̋', 'k': '̨'}

def decode_tex_accents(value: str) -> str:
    special_i = re.compile('\\\\([\\\'`\\"\\^~=\\.cvuHk])\\s*\\{\\\\i\\}')

    def repl_i(match: re.Match[str]) -> str:
        return unicodedata.normalize('NFC', 'i' + ACCENT_COMBINING[match.group(1)])
    value = special_i.sub(repl_i, value)
    punctuation_pattern = re.compile('\\\\([\\\'`\\"\\^~=\\.])\\s*(?:\\{([A-Za-z])\\}|([A-Za-z]))')
    named_pattern = re.compile('\\\\([cvuHk])\\s*\\{([A-Za-z])\\}')

    def repl(match: re.Match[str]) -> str:
        letter = match.group(2) or match.group(3)
        return unicodedata.normalize('NFC', letter + ACCENT_COMBINING[match.group(1)])
    value = punctuation_pattern.sub(repl, value)
    value = named_pattern.sub(lambda match: unicodedata.normalize('NFC', match.group(2) + ACCENT_COMBINING[match.group(1)]), value)
    letter_commands = {'\\ae{}': 'æ', '\\AE{}': 'Æ', '\\oe{}': 'œ', '\\OE{}': 'Œ', '\\o{}': 'ø', '\\O{}': 'Ø', '\\aa{}': 'å', '\\AA{}': 'Å', '\\ss{}': 'ß', '\\l{}': 'ł', '\\L{}': 'Ł', '\\i{}': 'ı', '\\texttimes{}': '×'}
    for (command, replacement) in letter_commands.items():
        value = value.replace(command, replacement)
    return value

def clean_value(value: object) -> str:
    if value is None:
        return ''
    text = str(value).replace('\ufeff', '')
    for _ in range(3):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    text = re.sub('</?[A-Za-z][^>]*>', '', text)
    for (old, new) in sorted(TEX_SIMPLE.items(), key=lambda item: -len(item[0])):
        text = text.replace(old, new)
    text = decode_tex_accents(text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub('\\s+', ' ', text).strip()
    return text

def canonical_field_name(name: str) -> str:
    lookup = {'author keywords': 'author_keywords', 'author_keywords': 'author_keywords', 'keywords-plus': 'keywords_plus', 'item doi': 'doi', 'publication year': 'year', 'document title': 'title', 'item title': 'title', 'publication title': 'journal', 'pdf link': 'url', 'journal volume': 'volume', 'journal issue': 'number', 'book series title': 'series', 'content type': 'type'}
    return lookup.get(name.strip().lower(), name.strip().lower().replace(' ', '_'))

def map_entry_type(raw_type: str, source_type: str='') -> str:
    text = f'{raw_type} {source_type}'.casefold()
    if 'inproceedings' in text or 'conference' in text or 'proceedings paper' in text:
        return 'inproceedings'
    if 'inbook' in text or 'chapter' in text or 'book chapter' in text:
        return 'incollection'
    if 'book' in text and 'booktitle' not in text:
        return 'book'
    if 'article' in text or 'review' in text:
        return 'article'
    if raw_type in {'article', 'book', 'inbook', 'incollection', 'inproceedings', 'proceedings', 'phdthesis', 'mastersthesis', 'techreport', 'unpublished', 'misc'}:
        return raw_type
    return 'misc'

def normalize_pages(fields: dict[str, str]) -> None:
    if not fields.get('pages'):
        start = fields.get('start_page', '')
        end = fields.get('end_page', '')
        if start and end:
            fields['pages'] = f'{start}--{end}'
        elif start:
            fields['pages'] = start

def normalized_standard_fields(fields: dict[str, str], source: str) -> dict[str, str]:
    lower = {canonical_field_name(k): clean_value(v) for (k, v) in fields.items()}
    if source == 'Scopus':
        keyword_parts = [lower.get('author_keywords', ''), lower.get('keywords', '')]
        lower['keywords'] = join_keywords(keyword_parts)
    elif source == 'IEEE Xplore':
        keyword_parts = [lower.get('author_keywords', ''), lower.get('ieee_terms', ''), lower.get('mesh_terms', '')]
        lower['keywords'] = join_keywords(keyword_parts)
    elif source == 'Web of Science':
        lower['keywords'] = join_keywords([lower.get('keywords', ''), lower.get('keywords_plus', '')])
    if source == 'SpringerLink':
        lower['journal'] = lower.get('publication_title', lower.get('journal', ''))
        lower['series'] = lower.get('book_series_title', lower.get('series', ''))
        lower['volume'] = lower.get('journal_volume', lower.get('volume', ''))
        lower['number'] = lower.get('journal_issue', lower.get('number', ''))
        lower['url'] = lower.get('url', '')
        lower['author'] = lower.get('authors', '')
    if source == 'IEEE Xplore':
        lower['journal'] = lower.get('publication_title', lower.get('journal', ''))
        lower['isbn'] = lower.get('isbns', lower.get('isbn', ''))
    normalize_pages(lower)
    return {name: lower.get(name, '') for name in STANDARD_FIELDS if lower.get(name, '')}

def join_keywords(values: Iterable[str]) -> str:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        for part in re.split('\\s*;\\s*|\\s*\\|\\s*', value or ''):
            item = clean_value(part)
            key = normalize_for_match(item)
            if item and key and (key not in seen):
                seen.add(key)
                result.append(item)
    return '; '.join(result)

def read_bib_file(path: Path, source: str) -> tuple[list[Record], Counter[str]]:
    raw = path.read_text(encoding='utf-8-sig', errors='replace')
    entries = split_bib_entries(raw)
    records: list[Record] = []
    field_counts: Counter[str] = Counter()
    errors: list[dict[str, object]] = []
    for (idx, text) in enumerate(entries, 1):
        try:
            (entry_type, key, raw_fields) = parse_bib_entry(text)
            field_counts.update((k.lower() for (k, v) in raw_fields.items() if clean_value(v)))
            fields = normalized_standard_fields(raw_fields, source)
            records.append(Record(source=source, input_file=path.name, input_index=idx, entry_type=map_entry_type(entry_type, raw_fields.get('type', '')), original_key=key, fields=fields, raw_fields=set(raw_fields)))
        except Exception as exc:
            errors.append({'file': path.name, 'entry': idx, 'error': str(exc)})
    if errors:
        raise RuntimeError(f'Could not conservatively parse {len(errors)} entries in {path.name}: ' + json.dumps(errors[:5], ensure_ascii=False))
    return (records, field_counts)

def read_csv_file(path: Path, source: str) -> tuple[list[Record], Counter[str]]:
    records: list[Record] = []
    field_counts: Counter[str] = Counter()
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as handle:
        reader = csv.DictReader(handle)
        for (idx, row) in enumerate(reader, 1):
            field_counts.update((k for (k, v) in row.items() if clean_value(v)))
            canonical = {canonical_field_name(k): clean_value(v) for (k, v) in row.items()}
            fields = normalized_standard_fields(canonical, source)
            records.append(Record(source=source, input_file=path.name, input_index=idx, entry_type=map_entry_type('', canonical.get('type', '')), original_key='', fields=fields, raw_fields={k for k in row if row[k]}))
    return (records, field_counts)
DOI_EXTRACT_RE = re.compile('(?i)(10\\.\\d{4,9}/[^\\s<>\\"{}]+)')

def normalize_doi(raw: str) -> str:
    text = html.unescape(raw or '').strip()
    text = re.sub('(?i)^(?:doi\\s*:\\s*|https?://(?:dx\\.)?doi\\.org/)', '', text)
    match = DOI_EXTRACT_RE.search(text)
    if not match:
        return ''
    doi = match.group(1).casefold().strip()
    doi = doi.split('?', 1)[0].split('#', 1)[0]
    doi = doi.rstrip('.,;:')
    for (opening, closing) in [('(', ')'), ('[', ']')]:
        while doi.endswith(closing) and doi.count(closing) > doi.count(opening):
            doi = doi[:-1]
    return doi if re.fullmatch('10\\.\\d{4,9}/\\S+', doi) else ''

def normalize_for_match(text: str) -> str:
    value = clean_value(text).casefold()
    value = ''.join((char for char in unicodedata.normalize('NFKD', value) if not unicodedata.combining(char)))
    value = value.replace('ß', 'ss')
    value = re.sub('[^a-z0-9]+', ' ', value)
    return re.sub('\\s+', ' ', value).strip()

def parse_year(value: str) -> int | None:
    match = re.search('\\b(1[5-9]\\d{2}|20\\d{2}|2100)\\b', value or '')
    return int(match.group(1)) if match else None

def prepare_records(records: list[Record]) -> None:
    for (idx, record) in enumerate(records, 1):
        record.raw_id = f'RAW{idx:06d}'
        explicit = normalize_doi(record.fields.get('doi', ''))
        from_url = normalize_doi(record.fields.get('url', '')) if not explicit else ''
        record.doi_normalized = explicit or from_url
        record.doi_origin = 'doi_field' if explicit else 'doi_url' if from_url else ''
        if record.doi_normalized:
            record.fields['doi'] = record.doi_normalized
        record.title_normalized = normalize_for_match(record.fields.get('title', ''))
        record.year_int = parse_year(record.fields.get('year', ''))

def classify_text(fields: dict[str, str]) -> Classification:
    return classify_contextual(fields.get('title', ''), fields.get('abstract', ''), fields.get('keywords', ''))

def set_classification(record: Record, result: Classification) -> None:
    record.wm_class = result.label
    record.wm_evidence = result.evidence
    record.wm_confidence = result.confidence
    record.wm_classifier_version = CLASSIFIER_VERSION
    record.intrinsic_hits = list(result.intrinsic_hits)
    record.extrinsic_hits = list(result.extrinsic_hits)
    record.weak_intrinsic_hits = list(result.weak_intrinsic_hits)
    record.weak_extrinsic_hits = list(result.weak_extrinsic_hits)
    record.negative_hits = list(result.negative_hits)

def apply_classifier(records: list[Record]) -> None:
    for record in records:
        set_classification(record, classify_text(record.fields))

def component_metadata(members: Iterable[int], records: list[Record]) -> tuple[set[str], set[str], int | None, int | None]:
    dois = {records[i].doi_normalized for i in members if records[i].doi_normalized}
    sources = {records[i].source for i in members}
    years = [records[i].year_int for i in members if records[i].year_int is not None]
    return (dois, sources, min(years) if years else None, max(years) if years else None)

def deduplicate(records: list[Record]) -> tuple[list[list[int]], list[dict[str, object]]]:
    uf = UnionFind(len(records))
    reasons: list[dict[str, object]] = []
    doi_groups: dict[str, list[int]] = defaultdict(list)
    for (idx, record) in enumerate(records):
        if record.doi_normalized:
            doi_groups[record.doi_normalized].append(idx)
    for (doi, members) in doi_groups.items():
        first = members[0]
        for idx in members[1:]:
            if uf.find(first) != uf.find(idx):
                uf.union(first, idx)
                reasons.append({'left_raw_id': records[first].raw_id, 'right_raw_id': records[idx].raw_id, 'rule': 'DOI', 'match_value': doi})
    title_groups: dict[str, list[int]] = defaultdict(list)
    for (idx, record) in enumerate(records):
        if record.title_normalized and record.year_int is not None:
            title_groups[record.title_normalized].append(idx)
    for (title, members) in title_groups.items():
        members = sorted(members, key=lambda i: (records[i].year_int or 0, i))
        for (pos, left) in enumerate(members):
            for right in members[pos + 1:]:
                left_year = records[left].year_int
                right_year = records[right].year_int
                assert left_year is not None and right_year is not None
                if right_year - left_year > 1:
                    break
                (rl, rr) = (uf.find(left), uf.find(right))
                if rl == rr:
                    continue
                left_members = uf.members[rl]
                right_members = uf.members[rr]
                (left_dois, left_sources, left_min, left_max) = component_metadata(left_members, records)
                (right_dois, right_sources, right_min, right_max) = component_metadata(right_members, records)
                if left_sources & right_sources:
                    continue
                if left_dois and right_dois and (left_dois != right_dois):
                    continue
                years = [year for year in [left_min, left_max, right_min, right_max] if year is not None]
                if years and max(years) - min(years) > 1:
                    continue
                uf.union(rl, rr)
                reasons.append({'left_raw_id': records[left].raw_id, 'right_raw_id': records[right].raw_id, 'rule': 'TITLE_YEAR_CROSS_SOURCE', 'match_value': f'{title} | {left_year}/{right_year}'})
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(records)):
        groups[uf.find(idx)].append(idx)
    ordered = sorted(groups.values(), key=lambda g: min(g))
    return (ordered, reasons)

def source_rank(source: str) -> int:
    ranking = {'Scopus': 0, 'Web of Science': 1, 'IEEE Xplore': 2, 'ScienceDirect': 3, 'ACM Digital Library': 4, 'SpringerLink': 5}
    return ranking.get(source, 99)

def merge_group(group: list[int], records: list[Record], unique_index: int) -> Record:
    members = [records[i] for i in group]
    ordered = sorted(members, key=lambda r: (-sum((bool(r.fields.get(f)) for f in STANDARD_FIELDS)), -len(r.fields.get('abstract', '')), source_rank(r.source), r.input_file, r.input_index))
    fields: dict[str, str] = {}
    for name in STANDARD_FIELDS:
        values = [r.fields.get(name, '') for r in ordered if r.fields.get(name, '')]
        if not values:
            continue
        if name == 'abstract':
            fields[name] = max(values, key=len)
        elif name == 'keywords':
            fields[name] = join_keywords(values)
        elif name == 'year':
            years = sorted({r.year_int for r in members if r.year_int is not None})
            if years:
                fields[name] = str(years[0])
            else:
                fields[name] = values[0]
        elif name == 'doi':
            dois = sorted({r.doi_normalized for r in members if r.doi_normalized})
            fields[name] = dois[0] if dois else values[0]
        else:
            fields[name] = values[0]
    origins = sorted({r.source for r in members}, key=lambda s: SOURCE_ORDER.index(s))
    entry_type = ordered[0].entry_type
    merged = Record(source='; '.join(origins), input_file='; '.join(sorted({r.input_file for r in members})), input_index=unique_index, entry_type=entry_type, original_key='', fields=fields, raw_id=f'WM{unique_index:06d}', doi_normalized=normalize_doi(fields.get('doi', '')), title_normalized=normalize_for_match(fields.get('title', '')), year_int=parse_year(fields.get('year', '')))
    set_classification(merged, classify_text(fields))
    return merged

def escape_bibtex(value: str) -> str:
    output: list[str] = []
    for char in clean_value(value):
        if char == '\\':
            output.append('\\textbackslash{}')
        elif char in '#%&_$':
            output.append('\\' + char)
        elif char == '{':
            output.append('\\textbraceleft{}')
        elif char == '}':
            output.append('\\textbraceright{}')
        elif char == '~':
            output.append('\\textasciitilde{}')
        elif char == '^':
            output.append('\\textasciicircum{}')
        else:
            output.append(char)
    return ''.join(output)

def record_bib_fields(record: Record, origins: str | None=None) -> dict[str, str]:
    fields = {k: v for (k, v) in record.fields.items() if k in STANDARD_FIELDS and v}
    fields['database'] = origins if origins is not None else record.source
    fields['wm_classifier_version'] = record.wm_classifier_version
    fields['wm_class'] = record.wm_class
    fields['wm_evidence'] = record.wm_evidence
    fields['wm_confidence'] = record.wm_confidence
    if record.intrinsic_hits:
        fields['wm_intrinsic_terms'] = '; '.join(record.intrinsic_hits)
    if record.extrinsic_hits:
        fields['wm_extrinsic_terms'] = '; '.join(record.extrinsic_hits)
    if record.weak_intrinsic_hits:
        fields['wm_weak_intrinsic_terms'] = '; '.join(record.weak_intrinsic_hits)
    if record.weak_extrinsic_hits:
        fields['wm_weak_extrinsic_terms'] = '; '.join(record.weak_extrinsic_hits)
    if record.negative_hits:
        fields['wm_negative_patterns'] = '; '.join(record.negative_hits)
    return fields

def write_bib(path: Path, records: list[Record], key_prefix: str) -> None:
    custom_order = STANDARD_FIELDS + ['database', 'wm_classifier_version', 'wm_class', 'wm_evidence', 'wm_confidence', 'wm_intrinsic_terms', 'wm_extrinsic_terms', 'wm_weak_intrinsic_terms', 'wm_weak_extrinsic_terms', 'wm_negative_patterns']
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for (idx, record) in enumerate(records, 1):
            key = f'{key_prefix}_{idx:06d}'
            handle.write(f'@{record.entry_type}{{{key},\n')
            fields = record_bib_fields(record)
            for name in custom_order:
                value = fields.get(name, '')
                if value:
                    handle.write(f'  {name} = {{{escape_bibtex(value)}}},\n')
            handle.write('}\n\n')

def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def field_inventory(file_records: dict[str, list[Record]], raw_counts: dict[str, Counter[str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (file_name, records) in file_records.items():
        source = records[0].source if records else ''
        counter = raw_counts[file_name]
        rows.append({'file': file_name, 'source': source, 'format': Path(file_name).suffix.lower().lstrip('.'), 'records': len(records), 'title_present': sum((bool(r.fields.get('title')) for r in records)), 'year_present': sum((r.year_int is not None for r in records)), 'doi_present_field': sum((bool(normalize_doi(r.fields.get('doi', ''))) for r in records)), 'doi_recovered_url': sum((r.doi_origin == 'doi_url' for r in records)), 'abstract_present': sum((bool(r.fields.get('abstract')) for r in records)), 'keywords_present': sum((bool(r.fields.get('keywords')) for r in records)), 'raw_fields_present': '; '.join((f'{name} ({count})' for (name, count) in sorted(counter.items())))})
    return rows

def validate_generated_bib(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding='utf-8')
    entries = split_bib_entries(raw)
    keys: list[str] = []
    errors: list[str] = []
    for (idx, entry) in enumerate(entries, 1):
        try:
            (_, key, _) = parse_bib_entry(entry)
            keys.append(key)
            if not re.fullmatch('[A-Za-z0-9_:+.-]+', key):
                errors.append(f'non-ASCII/invalid key at entry {idx}: {key}')
        except Exception as exc:
            errors.append(f'entry {idx}: {exc}')
    external_entries: int | None = None
    external_error = ''
    try:
        import bibtexparser
        with path.open('r', encoding='utf-8') as handle:
            external_entries = len(bibtexparser.load(handle).entries)
        if external_entries != len(entries):
            errors.append(f'bibtexparser recovered {external_entries}/{len(entries)} entries')
    except Exception as exc:
        external_error = str(exc)
        errors.append(f'bibtexparser validation error: {exc}')
    return {'file': str(path.relative_to(OUT_DIR)), 'entries': len(entries), 'unique_keys': len(set(keys)), 'bibtexparser_entries': external_entries, 'bibtexparser_error': external_error, 'errors': errors, 'valid': not errors and len(entries) == len(set(keys))}

def annual_table(records: list[Record]) -> pd.DataFrame:
    rows = []
    for record in records:
        if record.year_int is None:
            continue
        rows.append({'year': record.year_int, 'class': record.wm_class, 'evidence': record.wm_evidence})
    data = pd.DataFrame(rows)
    if data.empty:
        return pd.DataFrame()
    years = range(int(data.year.min()), int(data.year.max()) + 1)
    result = pd.DataFrame(index=years)
    result.index.name = 'year'
    result['total'] = data.groupby('year').size()
    for (label, column) in [('INTRINSIC', 'intrinsic'), ('EXTRINSIC', 'extrinsic'), ('BOTH_CONTEXTUAL', 'ambiguous_both'), ('INSUFFICIENT_EVIDENCE', 'insufficient_evidence'), ('NO_EVIDENCE', 'no_evidence')]:
        if label in {'INTRINSIC', 'EXTRINSIC'}:
            series = data[data['class'] == label].groupby('year').size()
        else:
            series = data[data['evidence'] == label].groupby('year').size()
        result[column] = series
    result = result.fillna(0).astype(int)
    result['classifiable'] = result['intrinsic'] + result['extrinsic']
    denominator = result['classifiable'].replace(0, np.nan)
    result['intrinsic_pct'] = 100 * result['intrinsic'] / denominator
    result['extrinsic_pct'] = 100 * result['extrinsic'] / denominator
    result['coverage_pct'] = 100 * result['classifiable'] / result['total'].replace(0, np.nan)
    result['intrinsic_roll3'] = result['intrinsic'].rolling(3, center=True, min_periods=2).sum()
    result['extrinsic_roll3'] = result['extrinsic'].rolling(3, center=True, min_periods=2).sum()
    roll_den = result['intrinsic_roll3'] + result['extrinsic_roll3']
    result['intrinsic_pct_roll3'] = 100 * result['intrinsic_roll3'] / roll_den
    result['extrinsic_pct_roll3'] = 100 * result['extrinsic_roll3'] / roll_den
    result['intrinsic_cumulative'] = result['intrinsic'].cumsum()
    result['extrinsic_cumulative'] = result['extrinsic'].cumsum()
    return result.reset_index()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-classifiable', type=int, default=10)
    args = parser.parse_args()
    ensure_dirs()
    files = input_files()
    all_records: list[Record] = []
    file_records: dict[str, list[Record]] = {}
    raw_field_counts: dict[str, Counter[str]] = {}
    hashes: list[dict[str, object]] = []
    for (path, source) in files:
        if path.suffix.lower() == '.csv':
            (records, counts) = read_csv_file(path, source)
        else:
            (records, counts) = read_bib_file(path, source)
        file_records[path.name] = records
        raw_field_counts[path.name] = counts
        all_records.extend(records)
        hashes.append({'file': path.name, 'source': source, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'records': len(records)})
    prepare_records(all_records)
    apply_classifier(all_records)
    inventory = field_inventory(file_records, raw_field_counts)
    write_csv(AUDIT_DIR / 'sources.csv', inventory, ['file', 'source', 'format', 'records', 'title_present', 'year_present', 'doi_present_field', 'doi_recovered_url', 'abstract_present', 'keywords_present', 'raw_fields_present'])
    write_csv(AUDIT_DIR / 'hashes.csv', hashes, ['file', 'source', 'bytes', 'records', 'sha256'])
    source_counts = Counter((r.source for r in all_records))
    (groups, _) = deduplicate(all_records)
    unique_records = [merge_group(group, all_records, idx) for (idx, group) in enumerate(groups, 1)]
    write_bib(CORPUS_DIR / 'classified_corpus.bib', unique_records, 'WM')
    group_rows: list[dict[str, object]] = []
    for (unique_idx, group) in enumerate(groups, 1):
        unique_id = f'WM{unique_idx:06d}'
        years = sorted({all_records[i].year_int for i in group if all_records[i].year_int is not None})
        dois = sorted({all_records[i].doi_normalized for i in group if all_records[i].doi_normalized})
        sources = sorted({all_records[i].source for i in group}, key=lambda s: SOURCE_ORDER.index(s))
        for idx in group:
            record = all_records[idx]
            group_rows.append({'unique_id': unique_id, 'raw_id': record.raw_id, 'source': record.source, 'input_file': record.input_file, 'input_index': record.input_index, 'original_key': record.original_key, 'doi_normalized': record.doi_normalized, 'title_normalized': record.title_normalized, 'year_original': record.year_int or '', 'group_sources': '; '.join(sources), 'group_years': '; '.join(map(str, years)), 'group_dois': '; '.join(dois), 'group_size': len(group)})
    write_csv(AUDIT_DIR / 'deduplication.csv', group_rows, ['unique_id', 'raw_id', 'source', 'input_file', 'input_index', 'original_key', 'doi_normalized', 'title_normalized', 'year_original', 'group_sources', 'group_years', 'group_dois', 'group_size'])
    classification_rows = [{'record_id': r.raw_id, 'year': r.year_int or '', 'title': r.fields.get('title', ''), 'origins': r.source, 'wm_class': r.wm_class, 'wm_evidence': r.wm_evidence, 'wm_confidence': r.wm_confidence, 'wm_classifier_version': r.wm_classifier_version, 'intrinsic_hits': '; '.join(r.intrinsic_hits), 'extrinsic_hits': '; '.join(r.extrinsic_hits), 'weak_intrinsic_hits': '; '.join(r.weak_intrinsic_hits), 'weak_extrinsic_hits': '; '.join(r.weak_extrinsic_hits), 'negative_hits': '; '.join(r.negative_hits), 'abstract_present': bool(r.fields.get('abstract')), 'keywords_present': bool(r.fields.get('keywords'))} for r in unique_records]
    write_csv(AUDIT_DIR / 'method_a_classifications.csv', classification_rows, ['record_id', 'year', 'title', 'origins', 'wm_class', 'wm_evidence', 'wm_confidence', 'wm_classifier_version', 'intrinsic_hits', 'extrinsic_hits', 'weak_intrinsic_hits', 'weak_extrinsic_hits', 'negative_hits', 'abstract_present', 'keywords_present'])
    annual = annual_table(unique_records)
    annual.to_csv(AUDIT_DIR / 'method_a_annual.csv', index=False, encoding='utf-8-sig', lineterminator='\n')
    validation = validate_generated_bib(CORPUS_DIR / 'classified_corpus.bib')
    if not validation['valid']:
        raise RuntimeError('Generated corpus failed BibTeX validation')
    print(json.dumps({'raw_records': len(all_records), 'unique_records': len(unique_records), 'duplicates_removed': len(all_records) - len(unique_records)}, indent=2))
if __name__ == '__main__':
    main()
