#!/usr/bin/env python3
"""
MCN Co-authorship Enricher
===========================

Turns the MCN roster (people who attended the course) into a *co-authorship*
network: two attendees are connected if they have co-authored one or more
papers, weighted by the number of shared papers. The result is written to
``collabs.js`` and rendered by the graph as a toggleable "Co-authorship" edge
layer alongside the existing "Co-attendance" layer.

Pipeline
--------
1. Load the rosters (``mcn_roster.json`` if present, else parse ``data.js``).
2. Canonicalize names via ``name_aliases.json`` so spelling variants collapse
   onto a single person.
3. Resolve each canonical name to a scholarly identity:
     - primary:   OpenAlex author (scored by name match, output volume and
                  topical relevance to neuroscience / computation).
     - anchor:    ORCID, taken from the chosen OpenAlex author when present.
     - crosscheck: Semantic Scholar author search (optional) to raise or lower
                  confidence when its ORCID/affiliation agrees or disagrees.
   ``author_overrides.json`` wins over anything decided automatically.
4. Fetch each resolved author's works from OpenAlex (cached to ``cache/``).
5. For every work, find which MCN people are co-authors and accumulate a
   weighted edge for each pair.
6. Emit ``collabs.js`` and an auditable ``resolution_review.csv``.

All network responses are cached on disk, so re-runs are free and offline.

Usage
-----
    python3 enrich_collabs.py                    # resolve everyone, full run
    python3 enrich_collabs.py --years 2013,2016  # validate on a subset first
    python3 enrich_collabs.py --email you@x.edu  # join OpenAlex's polite pool
    python3 enrich_collabs.py --no-s2            # skip the Semantic Scholar check
    python3 enrich_collabs.py --refresh          # ignore cache, re-fetch

Only the Python 3 standard library is required.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dotenv():
    """Load KEY=VALUE lines from a local, gitignored .env into os.environ.

    Keeps the OpenAlex email/API key out of source control and off the command
    line. Existing environment variables win over the file.
    """
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

OPENALEX_API = "https://api.openalex.org"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

# Concept/topic keywords used to judge whether an OpenAlex candidate plausibly
# works in the fields MCN draws from. Matched against the candidate's concepts.
RELEVANT_TOPIC_KEYWORDS = (
    "neuroscience", "neural", "neuron", "brain", "cognition", "cognitive",
    "computational", "computation", "biophysics", "biology", "physics",
    "machine learning", "artificial intelligence", "psychology", "vision",
    "systems biology", "statistics", "mathematics", "dynamical",
)

CONFIDENCE_THRESHOLD_DEFAULT = 0.5


# ── HTTP with on-disk caching ──────────────────────────────────────────────

# Circuit breaker: OpenAlex rate-limits aggressively (429 with multi-hour
# Retry-After). Once we've been throttled repeatedly we stop making new network
# calls and finish with whatever is already cached, so a run always produces a
# collabs.js instead of stalling for hours. Re-running later resumes from cache.
_NET = {"consecutive_429": 0, "cooldowns": 0, "cache_only": False}
# After this many consecutive 429s we take a longer cooldown and resume.
_MAX_CONSECUTIVE_429 = 3
# Length of that cooldown, and how many we tolerate before finally giving up
# (using whatever is cached) so a persistent block can't hang forever.
_COOLDOWN_SECONDS = 60
_MAX_COOLDOWNS = 10

# OpenAlex authentication, set once in main(). An API key lifts the shared
# rate limit; the email joins the (faster) polite pool.
_AUTH = {"email": None, "api_key": None}


def openalex_url(path, params):
    """Build an OpenAlex URL, attaching mailto + api_key when configured."""
    params = dict(params)
    if _AUTH["email"]:
        params["mailto"] = _AUTH["email"]
    if _AUTH["api_key"]:
        params["api_key"] = _AUTH["api_key"]
    return f"{OPENALEX_API}/{path}?" + urllib.parse.urlencode(params)


def _cache_path(*parts):
    path = os.path.join(HERE, "cache", *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _slugify(text):
    """Filesystem-safe slug for cache filenames."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "_"


def http_get_json(url, cache_key=None, refresh=False, retries=5, delay=1.0,
                  polite_delay=0.25):
    """GET a URL and parse JSON, caching the response under ``cache_key``.

    ``cache_key`` is a tuple of path components (e.g. ("openalex", "A123.json")).
    Returns the parsed object, or None on unrecoverable failure.
    """
    cache_file = _cache_path(*cache_key) if cache_key else None
    if cache_file and not refresh and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # Circuit tripped: don't make further network calls this run.
    if _NET["cache_only"]:
        return None

    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "MCNAlumniNetwork/1.0 (educational, non-commercial)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            if cache_file:
                with open(cache_file, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
            _NET["consecutive_429"] = 0  # a success resets the breaker
            time.sleep(polite_delay)  # be gentle even on cache misses
            return payload
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 404:
                return None
            if e.code == 429:
                # A multi-hour Retry-After means the OpenAlex credit window is
                # exhausted, not a transient burst — stop now and report when it
                # resets rather than waiting/cooling down pointlessly.
                retry_after_hdr = e.headers.get("Retry-After")
                if retry_after_hdr and retry_after_hdr.isdigit() and int(retry_after_hdr) > 300:
                    _NET["cache_only"] = True
                    mins = int(retry_after_hdr) // 60
                    print(f"    ! OpenAlex quota exhausted (x-ratelimit-remaining 0). "
                          f"Resets in ~{mins} min. Finishing with cached data; "
                          f"re-run after the reset to fetch the rest (cache is kept).",
                          file=sys.stderr)
                    return None
                _NET["consecutive_429"] += 1
                if _NET["consecutive_429"] >= _MAX_CONSECUTIVE_429:
                    # Persistent throttling: take a longer cooldown and resume,
                    # rather than abandoning the rest of the run. Only give up
                    # (cache-only) if this keeps happening.
                    _NET["cooldowns"] += 1
                    if _NET["cooldowns"] > _MAX_COOLDOWNS:
                        _NET["cache_only"] = True
                        print("    ! still rate-limited after repeated cooldowns — "
                              "finishing with cached data only. Re-run later to "
                              "fetch the remainder (cache is preserved).",
                              file=sys.stderr)
                        return None
                    print(f"    ! sustained rate-limiting — cooling down "
                          f"{_COOLDOWN_SECONDS}s (cooldown "
                          f"{_NET['cooldowns']}/{_MAX_COOLDOWNS}), then resuming",
                          file=sys.stderr)
                    time.sleep(_COOLDOWN_SECONDS)
                    _NET["consecutive_429"] = 0
                    continue
                # Honour Retry-After when present (capped), else back off.
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if (retry_after and retry_after.isdigit()) \
                    else delay * (attempt + 1) * 2
                print(f"    ! HTTP 429 (rate limited), waiting {min(wait, 30):.0f}s",
                      file=sys.stderr)
                time.sleep(min(wait, 30))
            else:
                print(f"    ! HTTP {e.code}, retrying", file=sys.stderr)
                time.sleep(delay * (attempt + 1) * 2)
        except Exception as e:  # noqa: BLE001 - network is best-effort here
            last_error = e
            print(f"    ! network error ({e}), retrying", file=sys.stderr)
            time.sleep(delay * (attempt + 1))
    print(f"    ! request failed after {retries} tries ({last_error}): "
          f"{_redact(url)}", file=sys.stderr)
    return None


def _redact(url):
    """Strip the api_key value from a URL before it's printed anywhere."""
    return re.sub(r"(api_key=)[^&]+", r"\1***", url)


# ── Roster loading ─────────────────────────────────────────────────────────

ROLE_ARRAY_KEYS = ("directors", "faculty", "lecturers", "tas", "students",
                   "assistants")


def load_rosters():
    """Return {"years": [...], "cohorts": {year: {role: [names]}}}.

    Prefers the machine-readable mcn_roster.json (written by scrape_mcn.py);
    falls back to parsing the arrays out of data.js so this works on the
    committed data without a fresh scrape.
    """
    roster_path = os.path.join(HERE, "mcn_roster.json")
    if os.path.exists(roster_path):
        with open(roster_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # JSON object keys are strings; normalize years to ints.
        cohorts = {int(y): c for y, c in data["cohorts"].items()}
        return {"years": [int(y) for y in data["years"]], "cohorts": cohorts}

    return _parse_data_js(os.path.join(HERE, "data.js"))


def _parse_data_js(path):
    """Extract rosters from the JavaScript in data.js via targeted regex.

    data.js is not valid JSON (unquoted keys, trailing commas), but each cohort
    is a numeric year key whose role arrays contain only quoted name strings, so
    a small regex is robust here.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    # Isolate the cohorts block so careerInfo entries are never matched.
    start = text.index("cohorts:")
    end = text.index("careerInfo", start) if "careerInfo" in text[start:] else len(text)
    cohorts_src = text[start:end]

    cohorts = {}
    for year_match in re.finditer(r"(\d{4})\s*:\s*\{(.*?)\},", cohorts_src, re.DOTALL):
        year = int(year_match.group(1))
        block = year_match.group(2)
        cohort = {}
        for key in ROLE_ARRAY_KEYS:
            arr_match = re.search(rf"{key}\s*:\s*\[(.*?)\]", block, re.DOTALL)
            names = re.findall(r'"([^"]*)"', arr_match.group(1)) if arr_match else []
            cohort[key] = names
        cohorts[year] = cohort

    return {"years": sorted(cohorts.keys()), "cohorts": cohorts}


def load_json_file(name):
    """Load a JSON config file, dropping keys that start with an underscore
    (those are inline documentation, not data)."""
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ── Name handling ──────────────────────────────────────────────────────────

def normalize_tokens(name):
    """Lowercased alphabetic tokens of a name, for matching."""
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", name.lower())
    return [t for t in cleaned.split() if t]


def name_match_score(query_name, candidate_name):
    """0..1 similarity focused on surname + given-name agreement.

    Requires the surname to match; rewards a matching given name over a mere
    shared initial. Middle names/initials are ignored.
    """
    q = normalize_tokens(query_name)
    c = normalize_tokens(candidate_name)
    if not q or not c:
        return 0.0
    if q[-1] != c[-1]:  # surnames must agree
        return 0.0
    score = 0.6  # surname match
    if q[0] == c[0]:
        score += 0.4  # full given-name match
    elif q[0][0] == c[0][0]:
        score += 0.2  # shared first initial
    return min(score, 1.0)


# ── OpenAlex resolution ────────────────────────────────────────────────────

def _short_id(openalex_url):
    """'https://openalex.org/A123' -> 'A123'."""
    if not openalex_url:
        return None
    return openalex_url.rstrip("/").split("/")[-1]


def topic_relevance(candidate):
    """0..1 relevance of an OpenAlex author's concepts to MCN's fields."""
    concepts = candidate.get("x_concepts") or candidate.get("topics") or []
    names = " ".join(
        (c.get("display_name") or "").lower() for c in concepts
    )
    hits = sum(1 for kw in RELEVANT_TOPIC_KEYWORDS if kw in names)
    if hits == 0:
        return 0.0
    return min(hits / 4.0, 1.0)


def prominence(candidate):
    """0..1 scholarly prominence from output + citations, on a log scale.

    Log-scaled (not a hard cap) so a 500-paper professor clearly outranks a
    30-paper namesake — this is what breaks ties when two candidates share a
    name, e.g. the real 'L. F. Abbott' over an obscure 'Larry Abbott'.
    """
    works = candidate.get("works_count", 0) or 0
    cites = candidate.get("cited_by_count", 0) or 0
    # log10(works) maxes ~2.7 at 500; log10(cites) maxes ~4.5 at 30k.
    w = min(math.log10(works + 1) / 2.7, 1.0)
    c = min(math.log10(cites + 1) / 4.5, 1.0)
    return 0.5 * w + 0.5 * c


def resolve_openalex(name, refresh):
    """Search OpenAlex for ``name`` and return (candidate, confidence) or (None, 0).

    Candidates are scored on: name agreement (gating — surname must match),
    topical relevance to MCN's fields, scholarly prominence (tie-breaker for
    same-name people), and whether an ORCID is attached.
    """
    url = openalex_url("authors", {
        "search": name,
        "per-page": "15",
        "select": ("id,display_name,display_name_alternatives,works_count,"
                   "cited_by_count,ids,x_concepts,last_known_institutions"),
    })
    data = http_get_json(url, cache_key=("openalex_authorsearch", _slugify(name) + ".json"),
                         refresh=refresh)
    if not data or not data.get("results"):
        return None, 0.0

    best, best_conf = None, 0.0
    for cand in data["results"]:
        display = cand.get("display_name", "")
        alternatives = cand.get("display_name_alternatives") or []
        nscore = max([name_match_score(name, display)]
                     + [name_match_score(name, alt) for alt in alternatives])
        if nscore == 0.0:
            continue

        relevance = topic_relevance(cand)
        prom = prominence(cand)
        has_orcid = 1.0 if (cand.get("ids") or {}).get("orcid") else 0.0

        # Name gates; prominence + relevance disambiguate same-name candidates.
        confidence = (0.45 * nscore + 0.22 * relevance
                      + 0.25 * prom + 0.08 * has_orcid)
        if confidence > best_conf:
            best_conf, best = confidence, cand

    return best, round(best_conf, 3)


def orcid_from_openalex(candidate):
    orcid_url = (candidate.get("ids") or {}).get("orcid")
    if not orcid_url:
        return None
    return orcid_url.rstrip("/").split("/")[-1]  # bare 0000-... form


# ── Semantic Scholar crosscheck ────────────────────────────────────────────

def semantic_scholar_orcid(name, refresh):
    """Best-effort ORCID for ``name`` from Semantic Scholar, or None."""
    params = {"query": name, "fields": "name,externalIds,paperCount", "limit": "5"}
    url = f"{SEMANTIC_SCHOLAR_API}/author/search?" + urllib.parse.urlencode(params)
    data = http_get_json(url, cache_key=("s2_authorsearch", _slugify(name) + ".json"),
                         refresh=refresh, polite_delay=1.0)
    if not data or not data.get("data"):
        return None
    for cand in data["data"]:
        if name_match_score(name, cand.get("name", "")) >= 0.6:
            orcid = (cand.get("externalIds") or {}).get("ORCID")
            if orcid:
                return orcid
    return None


# ── Works fetch + edge building ────────────────────────────────────────────

def fetch_works(openalex_id, orcid, refresh):
    """All works for a person, following cursor pagination.

    Filters by ORCID when available (this aggregates works across a person's
    possibly-split OpenAlex author records); otherwise by the author id.
    Returns a list of {"id", "year", "title", "author_ids": [...]}.
    """
    if orcid:
        author_filter = f"author.orcid:{orcid}"
        cache_id = orcid.replace("/", "_")
    else:
        author_filter = f"author.id:{openalex_id}"
        cache_id = openalex_id

    collected = []
    cursor = "*"
    page = 0
    while cursor:
        url = openalex_url("works", {
            "filter": author_filter,
            "per-page": "200",
            "cursor": cursor,
            "select": "id,title,publication_year,authorships",
        })
        data = http_get_json(
            url, cache_key=("openalex_works", f"{cache_id}_p{page}.json"),
            refresh=refresh)
        if not data:
            break
        for work in data.get("results", []):
            authors = []
            for a in work.get("authorships", []):
                author = a.get("author") or {}
                aid = _short_id(author.get("id"))
                orc = author.get("orcid")
                orc = orc.rstrip("/").split("/")[-1] if orc else None
                if aid or orc:
                    authors.append({"id": aid, "orcid": orc})
            collected.append({
                "id": _short_id(work.get("id")),
                "year": work.get("publication_year"),
                "title": work.get("title") or "",
                "authors": authors,
            })
        cursor = (data.get("meta") or {}).get("next_cursor")
        page += 1
        if page > 50:  # safety valve (~10k works)
            break
    return collected


def build_edges(resolved, refresh):
    """Build weighted co-authorship edges between resolved MCN people.

    ``resolved`` maps canonical name -> {"openalex_id", "orcid", ...}. Returns
    (edges, duplicates); edges are sorted by shared-paper count (descending).
    """
    # Lookups from an author record to our canonical name, by OpenAlex id and
    # by ORCID. A duplicate id means two roster names resolved to the same
    # scholar — merge onto the first and report it.
    id_to_name, orcid_to_name = {}, {}
    duplicates = []
    for cname, info in resolved.items():
        oid = info["openalex_id"]
        if oid in id_to_name:
            duplicates.append((cname, id_to_name[oid]))
            continue
        id_to_name[oid] = cname
        if info.get("orcid"):
            orcid_to_name.setdefault(info["orcid"], cname)

    def name_of(author):
        return id_to_name.get(author["id"]) or (
            orcid_to_name.get(author["orcid"]) if author.get("orcid") else None)

    # Scan every person's works. For each paper, the set of MCN people on it =
    # the person we're fetching for (always an author) plus any co-author we can
    # match by id or ORCID. Keyed by work id so shared papers count once.
    work_members = {}  # work_id -> {"year", "title", "names": set()}
    total = len(id_to_name)
    for i, (oid, cname) in enumerate(id_to_name.items(), 1):
        info = resolved[cname]
        print(f"  [{i}/{total}] works for {cname} ({oid})", file=sys.stderr)
        for work in fetch_works(oid, info.get("orcid"), refresh):
            names = {cname}
            for author in work["authors"]:
                nm = name_of(author)
                if nm:
                    names.add(nm)
            if len(names) < 2:
                continue  # no MCN co-authorship on this paper
            entry = work_members.setdefault(
                work["id"], {"year": work["year"], "title": work["title"], "names": set()})
            entry["names"].update(names)

    # Turn shared works into weighted pairwise edges. Sample titles are capped
    # (not every paper is kept) to bound collabs.js's size — most pairs have
    # only a handful of shared papers anyway; prolific pairs just show their
    # first MAX_SAMPLE_TITLES alongside the true total `papers` count.
    MAX_SAMPLE_TITLES = 8
    edges = {}
    for work in work_members.values():
        names = sorted(work["names"])
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                key = (names[a], names[b])
                edge = edges.setdefault(key, {"papers": 0, "years": set(), "titles": []})
                edge["papers"] += 1
                if work["year"]:
                    edge["years"].add(work["year"])
                if work["title"] and len(edge["titles"]) < MAX_SAMPLE_TITLES:
                    edge["titles"].append({"title": work["title"], "year": work["year"]})

    result = [{
        "source": a, "target": b,
        "papers": edge["papers"],
        "years": sorted(edge["years"]),
        "sampleTitles": edge["titles"],
    } for (a, b), edge in edges.items()]
    result.sort(key=lambda e: e["papers"], reverse=True)
    return result, duplicates


# ── Emit collabs.js ────────────────────────────────────────────────────────

def write_collabs_js(edges, resolved, aliases, out_path):
    resolved_public = {
        name: {
            "openalex": info["openalex_id"],
            "orcid": info.get("orcid"),
            "worksCount": info.get("works_count"),
            "confidence": info.get("confidence"),
        }
        for name, info in resolved.items()
    }
    # Ship the alias map so the browser collapses roster spelling variants onto
    # the same canonical node these edges are keyed by.
    payload = {"edges": edges, "resolved": resolved_public, "aliases": aliases}
    header = (
        "// MCN Alumni Network — co-authorship edges\n"
        "// Generated by enrich_collabs.py from OpenAlex (works) with a\n"
        "// Semantic Scholar / ORCID crosscheck. Regenerate with:\n"
        "//     python3 enrich_collabs.py\n"
        "// An edge means the two people co-authored `papers` publications.\n"
        "// Coverage is biased toward recent, actively-publishing academics;\n"
        "// absence of an edge does not mean absence of collaboration.\n\n"
    )
    body = "const MCN_COLLABS = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(header + body)


def write_review_csv(rows, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "canonical_name", "confidence", "resolved", "source",
            "openalex_id", "orcid", "openalex_display_name", "works_count",
            "s2_orcid_agrees",
        ])
        for r in rows:
            writer.writerow([
                r["name"], r.get("confidence", ""), r.get("resolved", False),
                r.get("source", ""), r.get("openalex_id", ""), r.get("orcid", ""),
                r.get("display_name", ""), r.get("works_count", ""),
                r.get("s2_orcid_agrees", ""),
            ])


# ── Driver ─────────────────────────────────────────────────────────────────

def collect_canonical_names(rosters, aliases, years_filter):
    """Unique canonical names across the (optionally year-filtered) rosters."""
    names = set()
    for year, cohort in rosters["cohorts"].items():
        if years_filter and year not in years_filter:
            continue
        for key in ROLE_ARRAY_KEYS:
            for raw in cohort.get(key, []):
                names.add(aliases.get(raw, raw))
    return sorted(names)


def main():
    ap = argparse.ArgumentParser(description="Build MCN co-authorship edges.")
    ap.add_argument("--years", help="comma-separated cohort years to restrict to "
                                     "(e.g. 2013,2016) for a validation run")
    ap.add_argument("--limit", type=int, help="resolve at most N people (debugging)")
    ap.add_argument("--email", default=None,
                    help="contact email for OpenAlex's fast 'polite pool' "
                         "(defaults to MCN_CONTACT_EMAIL from .env / env)")
    ap.add_argument("--api-key", default=None,
                    help="OpenAlex API key (defaults to OPENALEX_API_KEY from "
                         ".env / env). Lifts the shared rate limit.")
    ap.add_argument("--no-s2", action="store_true",
                    help="skip the Semantic Scholar crosscheck")
    ap.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD_DEFAULT,
                    help="minimum confidence to use an automatic resolution")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached responses and re-fetch")
    args = ap.parse_args()

    # Credentials: CLI flag wins, else .env / environment.
    load_dotenv()
    _AUTH["email"] = args.email or os.environ.get("MCN_CONTACT_EMAIL")
    _AUTH["api_key"] = args.api_key or os.environ.get("OPENALEX_API_KEY")

    years_filter = None
    if args.years:
        years_filter = {int(y) for y in args.years.split(",") if y.strip()}

    if _AUTH["api_key"]:
        print("Using OpenAlex API key (rate limit lifted).", file=sys.stderr)
    elif _AUTH["email"]:
        print(f"Using OpenAlex polite pool as {_AUTH['email']}.", file=sys.stderr)
    else:
        print("Note: no email/API key found; using OpenAlex's slower common "
              "pool and risking rate limits.\n", file=sys.stderr)

    rosters = load_rosters()
    aliases = load_json_file("name_aliases.json")
    overrides = load_json_file("author_overrides.json")

    names = collect_canonical_names(rosters, aliases, years_filter)
    if args.limit:
        names = names[:args.limit]
    print(f"Resolving {len(names)} unique people"
          + (f" from cohorts {sorted(years_filter)}" if years_filter else "")
          + " ...", file=sys.stderr)

    resolved = {}       # canonical name -> resolution info (>= threshold only)
    review_rows = []    # every person, for the audit CSV

    for i, name in enumerate(names, 1):
        override = overrides.get(name)
        if override and override.get("skip"):
            review_rows.append({"name": name, "resolved": False, "source": "override-skip"})
            continue

        if override and override.get("openalex_id"):
            oid = override["openalex_id"]
            info = {"openalex_id": oid, "orcid": override.get("orcid"),
                    "works_count": None, "confidence": 1.0, "source": "override"}
            resolved[name] = info
            review_rows.append({"name": name, "resolved": True, "source": "override",
                                "openalex_id": oid, "orcid": info["orcid"],
                                "confidence": 1.0})
            print(f"  [{i}/{len(names)}] {name}: override -> {oid}", file=sys.stderr)
            continue

        cand, conf = resolve_openalex(name, args.refresh)
        row = {"name": name, "confidence": conf}
        if not cand:
            row["resolved"] = False
            review_rows.append(row)
            print(f"  [{i}/{len(names)}] {name}: no match", file=sys.stderr)
            continue

        oid = _short_id(cand.get("id"))
        orcid = orcid_from_openalex(cand)
        row.update({
            "openalex_id": oid, "orcid": orcid,
            "display_name": cand.get("display_name"),
            "works_count": cand.get("works_count"),
        })

        # Semantic Scholar crosscheck: nudge confidence on ORCID (dis)agreement.
        if not args.no_s2:
            s2_orcid = semantic_scholar_orcid(name, args.refresh)
            if s2_orcid and orcid:
                agrees = (s2_orcid == orcid)
                row["s2_orcid_agrees"] = agrees
                conf = min(conf + 0.10, 1.0) if agrees else max(conf - 0.20, 0.0)
                row["confidence"] = round(conf, 3)

        keep = conf >= args.threshold
        row["resolved"] = keep
        row["source"] = "auto"
        review_rows.append(row)
        if keep:
            resolved[name] = {"openalex_id": oid, "orcid": orcid,
                              "works_count": cand.get("works_count"),
                              "confidence": round(conf, 3), "source": "auto"}
        print(f"  [{i}/{len(names)}] {name}: {oid} conf={conf:.2f} "
              f"{'kept' if keep else 'below-threshold'}", file=sys.stderr)

    print(f"\nResolved {len(resolved)}/{len(names)} people above threshold "
          f"{args.threshold}.", file=sys.stderr)

    # Fetch works and build the co-authorship edges.
    print("Fetching works and building edges ...", file=sys.stderr)
    edges, duplicates = build_edges(resolved, args.refresh)
    for dup, kept in duplicates:
        print(f"  ! '{dup}' resolved to the same scholar as '{kept}' — merged.",
              file=sys.stderr)

    # Write outputs.
    write_collabs_js(edges, resolved, aliases, os.path.join(HERE, "collabs.js"))
    write_review_csv(sorted(review_rows, key=lambda r: r.get("confidence", 0)),
                     os.path.join(HERE, "resolution_review.csv"))
    with open(os.path.join(HERE, "resolved_authors.json"), "w", encoding="utf-8") as fh:
        json.dump(resolved, fh, ensure_ascii=False, indent=2)

    print(f"\nDone.\n  Edges:            {len(edges)}\n"
          f"  Resolved people:  {len(resolved)}\n"
          f"  Review these:     resolution_review.csv (sorted worst-first)\n"
          f"  Graph data:       collabs.js", file=sys.stderr)


if __name__ == "__main__":
    main()
