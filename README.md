# Automated Academic Citation Verification System

A CLI research prototype for detecting likely hallucinated academic citations and identifying bibliographic metadata errors.

## Scope

Version 1 answers:

> Does this citation correspond to a real publication, and how accurate is its bibliographic metadata?

It does **not** verify whether a paper actually supports a generated claim.

The verifier uses scholarly metadata retrieval, string similarity, author matching, DOI comparison, candidate ranking, and field-level evidence. It does not use an LLM as the core verification mechanism.

## Sources

The initial implementation supports:

- Crossref
- OpenAlex

The source interfaces are modular so Semantic Scholar and arXiv can be added later.

## Project structure

```text
citation-verifier/
├── input.json
├── results.json
├── requirements.txt
├── README.md
├── config.py
├── main.py
├── parser/
│   └── citation_parser.py
├── sources/
│   ├── base.py
│   ├── crossref.py
│   └── openalex.py
├── matching/
│   ├── title_similarity.py
│   ├── author_similarity.py
│   ├── venue_similarity.py
│   └── candidate_ranker.py
├── verification/
│   ├── verifier.py
│   └── classifier.py
├── models/
│   └── schemas.py
├── utils/
│   ├── normalization.py
│   └── logging_config.py
└── tests/
    ├── test_parser.py
    ├── test_matching.py
    └── test_normalization.py
```

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Optional API configuration

Crossref and OpenAlex can be used without an API key.

For the Crossref polite pool, set:

```bash
# Windows PowerShell
$env:CROSSREF_EMAIL="your-email@example.com"
```

For OpenAlex:

```bash
$env:OPENALEX_EMAIL="your-email@example.com"
```

These values are only used for API identification. Do not put secrets in source code.

## Run

```bash
python main.py --input input.json --output results.json
```

The program creates:

- `results.json`
- `verification.log`

## Output classifications

### VERIFIED

A strong corresponding publication was found and the supplied metadata is largely consistent.

### EXISTS_METADATA_ERROR

A corresponding publication exists, but one or more supplied fields such as title, author, year, or venue differ.

### EXISTS_DOI_ERROR

The underlying publication was identified strongly, but the supplied DOI differs from the DOI associated with the matching publication.

### LIKELY_HALLUCINATED

No sufficiently reliable corresponding publication was found in the searched scholarly databases.

This is **not** proof that the publication does not exist.

### UNCERTAIN

There are plausible candidates or insufficient evidence to make a reliable decision.

## Retrieval strategy

For each citation, the verifier can use:

1. DOI lookup
2. Title search
3. First-author + title search
4. Title + venue search

The year is deliberately not used as a hard retrieval filter because the citation may contain an incorrect year.

## Candidate scoring

The baseline score is:

```text
0.45 * title_similarity
+ 0.30 * author_similarity
+ 0.10 * venue_similarity
+ 0.05 * year_similarity
+ 0.10 * doi_similarity
```

The weights are configurable in `config.py`.

They are baseline engineering values, not scientifically established optimal weights. A later benchmark should tune them against labeled data.

## Research evidence

For each citation, `results.json` stores:

- original citation
- parsed metadata
- final status
- confidence score
- best matching publication
- field-level comparisons
- retrieval methods
- search queries
- candidate list and scores

This allows later error analysis and calculation of accuracy, precision, recall, F1, and confusion matrices.

## Tests

Run:

```bash
python -m unittest discover -s tests -v
```

## Important limitation

Metadata quality differs between scholarly databases. A failed lookup is not equivalent to proof that a paper is fabricated. The system therefore uses the classification **LIKELY_HALLUCINATED** rather than claiming absolute non-existence.

## Future extensions

1. Better citation parsing
2. More retrieval sources
3. DOI registration-agency fallback
4. Semantic embeddings
5. Learned candidate ranking
6. Benchmark dataset
7. LLM baseline for comparison
8. Claim-to-citation verification
9. Web UI
