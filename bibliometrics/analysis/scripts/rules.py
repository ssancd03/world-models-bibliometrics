# Copyright 2026 Sergio Sánchez de la Fuente
# SPDX-License-Identifier: Apache-2.0
"""Contextual lexical rules for world-model terminology."""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
CLASSIFIER_VERSION = 'v2_contextual_rules_2026-08-10'
INTRINSIC_WEAK = ['learned', 'learning', 'latent', 'predictive', 'prediction', 'generative', 'neural', 'variational', 'transformer', 'self-supervised', 'dynamics model', 'foundation model']
EXTRINSIC_WEAK = ['map', 'mapping', 'symbolic', 'sensor fusion', 'knowledge base', 'world representation', 'environment model']
INTRINSIC_STRONG_PHRASES: dict[str, tuple[str, ...]] = {'model-based reinforcement learning': ('model based reinforcement learning', 'mbrl'), 'recurrent state-space model': ('recurrent state space model', 'rssm'), 'Dreamer': ('dreamer',), 'latent dynamics': ('latent dynamics', 'latent dynamical model'), 'action-conditioned prediction': ('action conditioned video prediction', 'action conditioned prediction', 'action conditional video prediction'), 'next-state prediction': ('next state prediction', 'next state predictor', 'predict next state', 'predicting next state'), 'learned predictive world model': ('learned world model', 'learn a world model', 'learning a world model', 'world model learning', 'predictive world model', 'latent world model', 'generative world model', 'neural world model'), 'predictive sensorimotor knowledge': ('predictive sequence learning', 'predictive sensorimotor', 'predictive sensory motor', 'general value function', 'general value functions')}
EXTRINSIC_STRONG_PHRASES: dict[str, tuple[str, ...]] = {'occupancy/grid map': ('occupancy grid', 'occupancy map', 'grid map', 'gridmap', 'grid based environment', 'grid based world model'), 'metric/geometric map': ('metric map', 'geometric map', 'geometry map'), 'topological map': ('topological map', 'topological path map', 'topological world model'), 'semantic/object map': ('semantic map', 'semantic world model', 'object map', 'object based world model', 'object based world representation', 'object based world representations'), '3D/HD map': ('3d map', 'three dimensional map', 'high definition map', 'hd map', 'high resolution map'), 'localization/prior map': ('localization map', 'localisation map', 'a priori map', 'prior map', 'landmark map', 'fingerprint map', 'map maintenance', 'large scale map', 'mapping in the marine environment', 'marine environment mapping', '3d organ map'), 'SLAM/OctoMap': ('slam', 'octomap'), 'CAD/spatial database': ('cad model', 'spatial database', 'spatial data base', 'geographic information system', 'gis database', 'electronic chart'), 'explicit environment representation': ('environment representation', 'representation of the environment', 'representation of its environment', 'representation of the surroundings', 'spatial representation of the environment'), 'world-model module/database': ('world modeling module', 'world modelling module', 'world model module', 'world model database', 'robotic world model', 'robot world model'), 'distributed/shared world model': ('distributed world model', 'shared world model', 'central world model')}
NEGATIVE_WORLD_MODEL_PATTERNS = ('\\bmodel free\\b', '\\bwithout (?:an? )?(?:explicit )?(?:world model|model of the world)\\b', '\\b(?:world model|model of the world) (?:is|was|are|were)? ?(?:not|unavailable|unobtainable)\\b', '\\bno (?:explicit )?(?:world model|model of the world)\\b')

@dataclass(frozen=True)
class Classification:
    label: str
    evidence: str
    confidence: str
    intrinsic_hits: tuple[str, ...]
    extrinsic_hits: tuple[str, ...]
    weak_intrinsic_hits: tuple[str, ...]
    weak_extrinsic_hits: tuple[str, ...]
    negative_hits: tuple[str, ...]

def normalize(text: str) -> str:
    value = (text or '').casefold()
    value = ''.join((char for char in unicodedata.normalize('NFKD', value) if not unicodedata.combining(char)))
    value = value.replace('ß', 'ss')
    value = re.sub('[^a-z0-9]+', ' ', value)
    return re.sub('\\s+', ' ', value).strip()

def split_units(text: str) -> list[str]:
    units = [normalize(part) for part in re.split('[.!?;:\\n]+', text or '')]
    return [unit for unit in units if unit]

def has_phrase(text: str, phrase: str) -> bool:
    return f' {normalize(phrase)} ' in f' {text} '

def phrase_hits(normalized: str, groups: dict[str, tuple[str, ...]]) -> list[str]:
    return [name for (name, aliases) in groups.items() if any((has_phrase(normalized, alias) for alias in aliases))]

def add_once(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)

def classify(title: str, abstract: str, keywords: str) -> Classification:
    raw = '\n'.join((part for part in (title, abstract, keywords) if part))
    normalized = normalize(raw)
    units = split_units(raw)
    robotics_domain = bool(re.search('\\b(?:robot|robotic|autonomous|unmanned|uav|uavs|auv|auvs|vehicle|vehicles|slam|mobile platform|artificial agent|intelligent agent|embodied agent)\\w*\\b', normalized))
    intrinsic = phrase_hits(normalized, INTRINSIC_STRONG_PHRASES)
    extrinsic = phrase_hits(normalized, EXTRINSIC_STRONG_PHRASES)
    if not robotics_domain:
        extrinsic = [hit for hit in extrinsic if hit != 'CAD/spatial database']
    weak_i = [term for term in INTRINSIC_WEAK if has_phrase(normalized, term)]
    weak_e = [term for term in EXTRINSIC_WEAK if has_phrase(normalized, term)]
    negative = [pattern for pattern in NEGATIVE_WORLD_MODEL_PATTERNS if re.search(pattern, normalized)]
    for unit in units:
        unit_negative = any((re.search(pattern, unit) for pattern in NEGATIVE_WORLD_MODEL_PATTERNS))
        human_biomedical = bool(re.search('\\b(?:human|humans|subject|subjects|patient|patients|animal|animals|motor learning|motor skills|neurorehabilitation|visuomotor|brain|cns|spinal|neuron|neurons|psychology|cognitive neuroscience)\\b', unit))
        artificial_agent = bool(re.search('\\b(?:robot|robotic|agent|autonomous vehicle|self driving car)\\w*\\b', unit))
        world_anchor = bool(re.search('\\b(?:world models?|models? of the world|environment models?|world representations?)\\b', unit))
        learned = bool(re.search('\\b(?:learn|learns|learned|learning|train|trained)\\w*\\b', unit))
        predictive = bool(re.search('\\b(?:predict|predicts|predicted|predicting|prediction|predictive|forecast)\\w*\\b', unit))
        latent_or_generative = bool(re.search('\\b(?:latent|generative|variational)\\b', unit))
        action = bool(re.search('\\b(?:action|actions|control|policy|sensorimotor|motor command)\\w*\\b', unit))
        future = bool(re.search('\\b(?:future|next state|next observation|consequence|effect|outcome)\\w*\\b', unit))
        robot_context = bool(re.search('\\b(?:robot|robotic|agent|autonomous|vehicle|navigation|planning|environment)\\w*\\b', unit))
        if world_anchor and (not unit_negative) and (not human_biomedical or artificial_agent):
            if latent_or_generative or (learned and predictive) or (predictive and action):
                add_once(intrinsic, 'context: learned/predictive world model')
            elif learned and re.search('\\b(?:internal|neural|dynamics?|state|representation)\\w*\\b', unit):
                add_once(intrinsic, 'context: learned internal world model')
            elif predictive and re.search('\\b(?:future|dynamics?|state|observation|simulation)\\w*\\b', unit):
                add_once(intrinsic, 'context: predictive world model')
            if re.search('\\b(?:maintain|maintains|maintained|maintenance|update|updated|query|queries|database|module|interface|shared|distributed)\\w*\\b', unit) and re.search('\\b(?:object|pose|position|scene|spatial|geometry|geometric|map|symbol|ontology|knowledge|discrete)\\w*\\b', unit):
                add_once(extrinsic, 'context: maintained/queryable explicit world model')
        if learned and re.search('\\b(?:dynamics?|transition) models?\\b', unit) and (action or future or robot_context):
            add_once(intrinsic, 'context: learned dynamics model')
        if predictive and future and action and robot_context:
            add_once(intrinsic, 'context: predicts action consequences/future state')
        if (not human_biomedical or artificial_agent) and re.search('\\b(?:mental simulation|mental simulations|imagination|imagined|rollout|rollouts)\\b', unit) and re.search('\\b(?:model|models|planning|predict|prediction|action|control)\\w*\\b', unit):
            add_once(intrinsic, 'context: model-based mental simulation/rollout')
        if re.search('\\b(?:internal model|internal models)\\b', unit) and re.search('\\b(?:learn|learned|learning|neural|predict|simulation|forward)\\w*\\b', unit) and (not human_biomedical or artificial_agent):
            add_once(intrinsic, 'context: learned/predictive internal model')
        if re.search('\\b(?:general value function|general value functions|gvf|gvfs)\\b', unit) and predictive:
            add_once(intrinsic, 'context: predictive knowledge/GVF')
        if (not human_biomedical or artificial_agent) and re.search('\\b(?:sensorimotor|sensory motor|sensor events?)\\b', unit) and re.search('\\b(?:predict|prediction|predictions|predictive|expected)\\w*\\b', unit) and artificial_agent:
            add_once(intrinsic, 'context: predicts sensorimotor action consequences')
        if re.search('\\b(?:video prediction|predict future frames|predicting future frames)\\b', unit) and (action or robot_context):
            add_once(intrinsic, 'context: robotic video prediction')
        if latent_or_generative and learned and re.search('\\b(?:state|states|factor|factors|representation|representations|dynamics)\\w*\\b', unit) and robot_context:
            add_once(intrinsic, 'context: learned latent/generative state representation')
        if re.search('\\bforward models?\\b', unit) and (learned or predictive or action):
            add_once(intrinsic, 'context: learned/predictive forward model')
        if re.search('\\b(?:3d|three dimensional) (?:geometric )?models?\\b', unit) and re.search('\\b(?:scene|world|environment|room|indoor|object|workpiece|organ|terrain)\\w*\\b', unit):
            add_once(extrinsic, 'context: explicit 3D environment/scene model')
        if robotics_domain and re.search('\\b(?:map|maps|mapping)\\b', unit) and re.search('\\b(?:navigation|localization|localisation|path planning|route|road|obstacle|scene|spatial|marine|indoor|outdoor|terrain|environment|world|star sensor)\\w*\\b', unit):
            add_once(extrinsic, 'context: map used for navigation/localization')
        if re.search('\\b(?:visual memory|world representation|scene representation|perceptual world)\\b', unit) and re.search('\\b(?:object|objects|position|positions|pose|poses|spatial|room|environment)\\w*\\b', unit):
            add_once(extrinsic, 'context: explicit objects/poses scene representation')
        if robotics_domain and re.search('\\b(?:ontology|symbolic representation|symbolic world|symbolic model)\\b', unit) and re.search('\\b(?:robot|world|environment|object|objects|planning|scene)\\w*\\b', unit):
            add_once(extrinsic, 'context: symbolic/ontological environment model')
        if robotics_domain and re.search('\\b(?:virtual environment|cad)\\b', unit) and re.search('\\b(?:robot|workpiece|scene|spatial|registration|planning|model)\\w*\\b', unit):
            add_once(extrinsic, 'context: explicit CAD/virtual environment')
    if re.search('\\bgeneral value functions?\\b|\\bgvfs?\\b', normalized) and re.search('\\b(?:predict|prediction|predictions|predictive|forecast)\\w*\\b', normalized) and re.search('\\b(?:robot|agent|control|world)\\w*\\b', normalized):
        add_once(intrinsic, 'context: predictive knowledge/GVF')
    if re.search('\\brobot knowledge\\b', normalized) and re.search('\\b(?:represent|representation|maintain|maintenance|query|querying|planning)\\w*\\b', normalized):
        add_once(extrinsic, 'context: maintained/queryable robot knowledge')
    if re.search('\\b(?:distributed|shared|central) world model\\b', normalized) and re.search('\\b(?:coordination|team|multi robot|robots|architecture|communication)\\w*\\b', normalized):
        add_once(extrinsic, 'context: shared explicit world model')
    if re.search('\\bworld model(?:ing)?\\b', normalized) and re.search('\\b(?:task planner|symbolic planner|planning problem|object states?|robot states?|skill definitions?)\\b', normalized):
        add_once(extrinsic, 'context: explicit state supplied to symbolic planning')
    if re.search('\\bworld representations?\\b', normalized) and re.search('\\b(?:build|built|valid|room|objects?|poses?|grammar|grammatical)\\w*\\b', normalized):
        add_once(extrinsic, 'context: constructed explicit world representation')
    if re.search('\\b(?:perceptually grounded model of the world|inner domain representation)\\b', normalized) and re.search('\\b(?:ontology|rule based|structured knowledge|semantic)\\w*\\b', normalized):
        add_once(extrinsic, 'context: grounded symbolic domain representation')
    if re.search('\\b(?:3d|three dimensional) models?\\b', normalized) and re.search('\\b(?:scene|environment|camera pose|structure from motion|geometry|spatial)\\w*\\b', normalized):
        add_once(extrinsic, 'context: explicit 3D scene/environment model')
    if re.search('\\b(?:low fidelity )?model of the world\\b', normalized) and re.search('\\b(?:planning|replanning|planner|current state|goal)\\w*\\b', normalized):
        add_once(extrinsic, 'context: explicit world state for planning')
    if intrinsic and extrinsic:
        (label, evidence, confidence) = ('AMBIGUOUS', 'BOTH_CONTEXTUAL', 'MEDIUM')
    elif intrinsic:
        (label, evidence) = ('INTRINSIC', 'INTRINSIC_CONTEXTUAL')
        confidence = 'HIGH' if len(intrinsic) >= 2 else 'MEDIUM'
    elif extrinsic:
        (label, evidence) = ('EXTRINSIC', 'EXTRINSIC_CONTEXTUAL')
        confidence = 'HIGH' if len(extrinsic) >= 2 else 'MEDIUM'
    elif weak_i or weak_e or negative:
        (label, evidence, confidence) = ('AMBIGUOUS', 'INSUFFICIENT_EVIDENCE', 'LOW')
    else:
        (label, evidence, confidence) = ('AMBIGUOUS', 'NO_EVIDENCE', 'LOW')
    return Classification(label=label, evidence=evidence, confidence=confidence, intrinsic_hits=tuple(intrinsic), extrinsic_hits=tuple(extrinsic), weak_intrinsic_hits=tuple(weak_i), weak_extrinsic_hits=tuple(weak_e), negative_hits=tuple(negative))

def configuration() -> dict[str, object]:
    return {'version': CLASSIFIER_VERSION, 'intrinsic_strong_phrases': INTRINSIC_STRONG_PHRASES, 'extrinsic_strong_phrases': EXTRINSIC_STRONG_PHRASES, 'intrinsic_weak_terms': INTRINSIC_WEAK, 'extrinsic_weak_terms': EXTRINSIC_WEAK, 'negative_world_model_patterns': NEGATIVE_WORLD_MODEL_PATTERNS, 'decision_rule': {'intrinsic_strong_only': 'INTRINSIC / INTRINSIC_CONTEXTUAL', 'extrinsic_strong_only': 'EXTRINSIC / EXTRINSIC_CONTEXTUAL', 'both_strong': 'AMBIGUOUS / BOTH_CONTEXTUAL', 'weak_or_negative_only': 'AMBIGUOUS / INSUFFICIENT_EVIDENCE', 'no_signal': 'AMBIGUOUS / NO_EVIDENCE'}}
