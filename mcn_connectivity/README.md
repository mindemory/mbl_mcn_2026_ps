# MCN Connectivity

**Interactive alumni network graph for the MBL Methods in Computational Neuroscience course.**

Built during MCN 2026 at Woods Hole, MA.

## Overview

This is a D3.js force-directed graph visualizing the network of MCN alumni from 1988–2017. Nodes are people (students, faculty, directors, lecturers, TAs). Edges come in two layers you can toggle between:

- **Co-attendance** — connects everyone who attended the same cohort year.
- **Co-authorship** — connects people who have co-authored papers, weighted by the number of shared publications (resolved from [OpenAlex](https://openalex.org), cross-checked against Semantic Scholar / ORCID).

## Quick Start

```bash
# From this directory
python3 -m http.server 8787
# then open: http://localhost:8787
```

## Features

- **Connections** — toggle between **Co-attendance**, **Co-authorship**, or **Both** edge layers
- **Color by year / role / domain / connections** — including by **research domain** (curated field buckets)
- **Year / Role / Domain filters** — isolate a cohort, a role, or a research area
- **Force / Radial / Domains layout** — physics, ring-by-year, or clustered-by-research-domain
- **Search** — highlight by name or affiliation (searches career data too)
- **Click any node** — detail panel shows research areas, MCN appearances, co-authors (with shared-paper counts), and scholarly-profile links
- **Drag nodes** — fully interactive

## Data pipeline

Two scripts generate the data the site ships with. Both use only the Python 3 standard library.

### 1. Rosters — `scrape_mcn.py`

Scrapes the MBL History Archive and writes `data.js` (the graph data) and `mcn_roster.json` (machine-readable rosters for the enricher).

```bash
python3 scrape_mcn.py
```

### 2. Co-authorship edges — `enrich_collabs.py`

Resolves each person to a scholarly identity and derives weighted co-authorship edges, writing `collabs.js`.

```bash
# Validate on a subset first, then do the full run:
python3 enrich_collabs.py --years 2013,2016      # quick sanity check
python3 enrich_collabs.py --email you@uni.edu    # full run (polite OpenAlex pool)
```

Useful flags: `--no-s2` (skip Semantic Scholar), `--threshold 0.6` (stricter matches), `--refresh` (ignore cache), `--limit N` (debug).

How it resolves people:

1. **Canonicalize** names via `name_aliases.json` (collapses spelling variants so one person is one node).
2. **Resolve** each name against **OpenAlex** (scored on name match, output volume, and topical relevance), take **ORCID** from the chosen author, and **crosscheck** with **Semantic Scholar**.
3. **Override** anything wrong via `author_overrides.json` — this always wins.
4. **Fetch works** (cached under `cache/`) and build a weighted edge for every pair of MCN co-authors.

Outputs:

| File | Committed? | Purpose |
|------|-----------|---------|
| `collabs.js` | yes | co-authorship edges consumed by the graph |
| `domains.js` | yes | per-person research domains consumed by the graph |
| `resolution_review.csv` | no | every person + confidence, **sorted worst-first** — inspect this |
| `resolved_authors.json` | no | machine-readable resolution results |
| `cache/` | no | cached API responses (safe to delete) |

### 3. Research domains — `enrich_domains.py`

Labels each resolved person with a **research-domain bucket** (Computational & Theoretical, Machine Learning & AI, Sensory & Perception, Cellular & Molecular, Cognitive & Behavioral, Motor & Action, Systems & Circuits, Biophysics & Physics, Methods/Stats/Imaging, Clinical & Disease), writing `domains.js`. This powers the Domain color mode, the Domain filter, and the Domains layout.

```bash
python3 enrich_domains.py            # reads resolved_authors.json, writes domains.js
```

Buckets are assigned from each person's OpenAlex `topics`. OpenAlex tags ~half of all neuro people with the same generic top topic ("Neural dynamics and brain function"), so the classifier **skips that filler and keys off the first specific topic** (see `BUCKET_RULES` / `GENERIC_TOPICS` in `enrich_domains.py`); people who are *only* generic dynamics fall to "Computational & Theoretical". Tune `BUCKET_RULES` to taste.

**Free:** it fetches author records by id, and single-record lookups cost 0 OpenAlex credits (only searches/filtered lists are billed). Run it any time after `enrich_collabs.py`.

### ⚠️ Coverage caveat

Author name disambiguation is hard and coverage is uneven. Old cohorts, people who left academia, non-publishers, and the deceased resolve poorly, so the co-authorship layer **over-represents recent, prolific academics**. The absence of an edge does **not** mean two people never collaborated. Curate `author_overrides.json` and `name_aliases.json` from `resolution_review.csv` to improve accuracy over time.

### 💳 OpenAlex credits

OpenAlex bills **10 credits per search** (`/authors?search=`) but only **1 credit per filtered list** (`/works?filter=`), and **single-record GETs are free**. The free plan is $1/day = 10,000 credits. Resolving all ~1,000 names is the expensive step (~10k credits), so **never use `--refresh`** — cached searches and `resolved_authors.json` are the costly asset. A full edges re-run costs only ~1,500 credits; the domains run costs ~0.

## Data sources

- Rosters: [MBL History Archives](https://history.archives.mbl.edu/people-and-courses/course-group/methods-computational-neuroscience) (public), 1988–2017
- Co-authorship: [OpenAlex](https://openalex.org) works, with Semantic Scholar / ORCID crosscheck
- Career info: manually curated from public lab pages / ORCID (see `data.js`)

## Roadmap

- [x] Scrape all years (1988–2017) from MBL archive
- [x] Add OpenAlex co-authorship edges
- [ ] Add 2026 cohort (after course ends)
- [ ] GitHub Pages deploy for community access
- [ ] Cross-link to Neurotree for academic genealogy

## File structure

```
mcn_connectivity/
├── index.html            # Main page + controls
├── style.css             # Dark theme
├── graph.js              # D3 force simulation engine + edge-layer toggle
├── data.js               # MCN roster data + career annotations (generated)
├── mcn_roster.json       # Machine-readable rosters (generated)
├── collabs.js            # Co-authorship edges (generated)
├── domains.js            # Per-person research domains (generated)
├── scrape_mcn.py         # Roster scraper
├── enrich_collabs.py     # Co-authorship resolver / edge builder
├── enrich_domains.py     # Research-domain labeler (free)
├── name_aliases.json     # Spelling-variant → canonical name map (manual)
├── author_overrides.json # Pin/skip scholarly identities (manual)
└── README.md
```

## Contributing

- **Career info:** edit `data.js` → `careerInfo` map. Schema: `"Name": { affiliation: "...", role: "..." }`.
- **Fix a merged/wrong scholar:** add an entry to `author_overrides.json`, then re-run `enrich_collabs.py`.
- **Fix duplicate people:** add the variant to `name_aliases.json`.

---
*Built with D3.js v7 · Data from MBL History Archives · 🐟 MCN 2026*
