# Search strategies

Records from the six sources were imported into Parsifal on 15 July 2026. No publication-year, language or document-type filters were applied in the databases. The inclusive 2020 cutoff and the remaining eligibility criteria were applied during selection.

All queries combine three concept groups with AND: ground-robot terms, foundation-model or learned world-model terms, and navigation or planning terms. Synonyms within each group are joined with OR. The syntax was adapted to the fields and operators of each platform.

## 1. Scopus

Field `TITLE-ABS-KEY`, no filters.

```text
TITLE-ABS-KEY(("wheeled mobile robot" OR "wheeled robot" OR "mobile robot" OR "ground robot" OR "terrestrial robot" OR "field robot" OR "agricultural robot" OR "unmanned ground vehicle" OR UGV) AND ("foundation model" OR "foundational model" OR "robotics foundation model" OR "robot foundation model" OR "navigation foundation model" OR "general navigation model" OR "large language model" OR LLM OR MLLM OR "vision-language model" OR "vision language model" OR "visual-language model" OR VLM OR "vision-language-action model" OR "vision language action model" OR "navigation world model" OR "navigation world models" OR "learned world model" OR "predictive world model" OR "latent world model" OR "action-conditioned world model" OR (("world model" OR "world models" OR "world modeling" OR "world modelling") W/5 (navigation OR planning OR learn* OR predict* OR latent OR generative OR action* OR dynamics))) AND ("autonomous navigation" OR "robot navigation" OR "mobile robot navigation" OR "outdoor navigation" OR "language-guided navigation" OR "vision-language navigation" OR "vision-and-language navigation" OR VLN OR "semantic navigation" OR "visual navigation" OR "terrain-aware navigation" OR "path planning" OR "obstacle avoidance"))
```

## 2. Web of Science

Core Collection, field `Topic`, no filters.

```text
("wheeled mobile robot" OR "wheeled robot" OR "mobile robot" OR "ground robot" OR "terrestrial robot" OR "field robot" OR "agricultural robot" OR "unmanned ground vehicle" OR UGV) AND ("foundation model" OR "foundational model" OR "robotics foundation model" OR "robot foundation model" OR "navigation foundation model" OR "general navigation model" OR "large language model" OR LLM OR MLLM OR "vision-language model" OR "vision language model" OR "visual-language model" OR VLM OR "vision-language-action model" OR "vision language action model" OR "navigation world model" OR "navigation world models" OR "learned world model" OR "predictive world model" OR "latent world model" OR "action-conditioned world model" OR (("world model" OR "world models" OR "world modeling" OR "world modelling") NEAR/5 (navigation OR planning OR learn* OR predict* OR latent OR generative OR action* OR dynamics))) AND ("autonomous navigation" OR "robot navigation" OR "mobile robot navigation" OR "outdoor navigation" OR "language-guided navigation" OR "vision-language navigation" OR "vision-and-language navigation" OR VLN OR "semantic navigation" OR "visual navigation" OR "terrain-aware navigation" OR "path planning" OR "obstacle avoidance")
```

## 3. ACM Digital Library

Full-Text Collection, field `Anywhere`. Proximity operators were not used, so the *world model* variants were searched as independent terms.

```text
("wheeled mobile robot" OR "wheeled robot" OR "mobile robot" OR "ground robot" OR "terrestrial robot" OR "field robot" OR "agricultural robot" OR "unmanned ground vehicle" OR UGV) AND ("foundation model" OR "foundational model" OR "robotics foundation model" OR "robot foundation model" OR "navigation foundation model" OR "general navigation model" OR "large language model" OR LLM OR MLLM OR "vision-language model" OR "vision language model" OR "visual-language model" OR VLM OR "vision-language-action model" OR "vision language action model" OR "navigation world model" OR "navigation world models" OR "learned world model" OR "predictive world model" OR "latent world model" OR "action-conditioned world model" OR "world model" OR "world models" OR "world modeling" OR "world modelling") AND ("autonomous navigation" OR "robot navigation" OR "mobile robot navigation" OR "outdoor navigation" OR "language-guided navigation" OR "vision-language navigation" OR "vision-and-language navigation" OR VLN OR "semantic navigation" OR "visual navigation" OR "terrain-aware navigation" OR "path planning" OR "obstacle avoidance")
```

## 4. IEEE Xplore

Command Search, field `All Metadata`, no filters.

```text
((("All Metadata":"wheeled mobile robot") OR ("All Metadata":"wheeled robot") OR ("All Metadata":"mobile robot") OR ("All Metadata":"ground robot") OR ("All Metadata":"terrestrial robot") OR ("All Metadata":"field robot") OR ("All Metadata":"agricultural robot") OR ("All Metadata":"unmanned ground vehicle") OR ("All Metadata":UGV)) AND (("All Metadata":"foundation model") OR ("All Metadata":"foundational model") OR ("All Metadata":"robotics foundation model") OR ("All Metadata":"robot foundation model") OR ("All Metadata":"navigation foundation model") OR ("All Metadata":"general navigation model") OR ("All Metadata":"large language model") OR ("All Metadata":LLM) OR ("All Metadata":MLLM) OR ("All Metadata":"vision-language model") OR ("All Metadata":"vision language model") OR ("All Metadata":"visual-language model") OR ("All Metadata":VLM) OR ("All Metadata":"vision-language-action model") OR ("All Metadata":"vision language action model") OR ("All Metadata":"navigation world model") OR ("All Metadata":"learned world model") OR ("All Metadata":"predictive world model") OR ("All Metadata":"latent world model") OR ("All Metadata":"action-conditioned world model") OR ("All Metadata":"world model") OR ("All Metadata":"world models") OR ("All Metadata":"world modeling") OR ("All Metadata":"world modelling")) AND (("All Metadata":"autonomous navigation") OR ("All Metadata":"robot navigation") OR ("All Metadata":"mobile robot navigation") OR ("All Metadata":"outdoor navigation") OR ("All Metadata":"language-guided navigation") OR ("All Metadata":"vision-language navigation") OR ("All Metadata":"vision-and-language navigation") OR ("All Metadata":VLN) OR ("All Metadata":"semantic navigation") OR ("All Metadata":"visual navigation") OR ("All Metadata":"terrain-aware navigation") OR ("All Metadata":"path planning") OR ("All Metadata":"obstacle avoidance")))
```

## 5. SpringerLink

Field `Keywords`, with an additional outdoor-context block.

```text
("wheeled mobile robot" OR "wheeled robot" OR "mobile robot" OR "ground robot" OR "terrestrial robot" OR "field robot" OR "agricultural robot" OR "unmanned ground vehicle" OR UGV) AND ("foundation model" OR "foundational model" OR "robotics foundation model" OR "robot foundation model" OR "navigation foundation model" OR "general navigation model" OR "large language model" OR LLM OR MLLM OR "vision-language model" OR "vision language model" OR "visual-language model" OR VLM OR "vision-language-action model" OR "vision language action model" OR "navigation world model" OR "learned world model" OR "predictive world model" OR "latent world model" OR "action-conditioned world model" OR "world model" OR "world models" OR "world modeling" OR "world modelling") AND ("autonomous navigation" OR "robot navigation" OR "mobile robot navigation" OR "outdoor navigation" OR "language-guided navigation" OR "vision-language navigation" OR "vision-and-language navigation" OR VLN OR "semantic navigation" OR "visual navigation" OR "terrain-aware navigation" OR "path planning" OR "obstacle avoidance") AND ("outdoor environment" OR "outdoor scene" OR "outdoor scenario" OR "outdoor navigation" OR "semi-structured environment" OR "off-road environment" OR "agricultural field" OR "crop row" OR orchard OR vineyard OR "rural environment" OR "construction site" OR "residential area")
```

## 6. ScienceDirect

The platform limits query length and the number of Boolean operators, so the query was split into four subqueries whose results were merged before deduplication.

```text
SD1: ("mobile robot" OR "ground robot") AND ("foundation model" OR "large language model" OR "vision-language model") AND ("autonomous navigation" OR "path planning")

SD2: ("agricultural robot" OR "field robot" OR "unmanned ground vehicle") AND ("foundation model" OR "large language model" OR "vision-language model") AND navigation

SD3: ("mobile robot" OR "agricultural robot") AND ("vision-language-action" OR "vision-and-language navigation" OR VLN OR MLLM) AND navigation

SD4: ("mobile robot" OR "agricultural robot" OR "unmanned ground vehicle") AND "world model" AND (navigation OR "path planning" OR "obstacle avoidance")
```

## Differences between platforms

- Scopus and Web of Science use a proximity operator to restrict generic *world model* variants. ACM Digital Library, IEEE Xplore and SpringerLink do not.
- SpringerLink adds a mandatory outdoor-context block.
- ScienceDirect required four shorter subqueries.
- ACM Digital Library searches the `Anywhere` field, which is broader than title, abstract and keywords.
- A query for arXiv was prepared but not used, because the protocol excludes preprints.

These differences follow from technical limits of each platform and are reported for transparency.
