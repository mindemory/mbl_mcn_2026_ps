#!/usr/bin/env python3
"""
MCN Research-domain Enricher
============================

Labels each resolved MCN person with their research domain(s) so the graph can
color, filter, and cluster people by field of study.

OpenAlex author records carry a ``topics`` array, each entry pre-classified into
a ``subfield -> field -> domain`` hierarchy and sorted by how many of the
author's works fall under it. We take the top topic's **subfield** as the
person's primary research domain (e.g. "Cognitive Neuroscience", "Artificial
Intelligence", "Cellular and Molecular Neuroscience").

Cost: fetching a single author record by id is FREE on OpenAlex (only search
and filtered-list endpoints cost credits), so enriching everyone costs nothing.

Reads:  resolved_authors.json (produced by enrich_collabs.py)
Writes: domains.js

Usage:
    python3 enrich_domains.py            # all resolved people
    python3 enrich_domains.py --refresh  # ignore cache, re-fetch

Only the Python 3 standard library is required.
"""

import argparse
import json
import os
import sys

# Reuse the HTTP/auth/cache/env plumbing from the co-authorship enricher.
import enrich_collabs as ec

HERE = os.path.dirname(os.path.abspath(__file__))
HOW_MANY_TOPICS = 3  # top-N topics kept per person for the detail panel
CLASSIFY_TOPICS = 6  # top-N topics scanned when assigning a domain bucket

# OpenAlex tags a huge share of neuro people with these generic "filler" topics
# as #1, so they carry no discriminating signal. We skip them and classify on
# the first *specific* topic; someone whose whole profile is only these is, by
# definition, a core dynamics/theory person -> "Computational & Theoretical".
GENERIC_TOPICS = {
    "neural dynamics and brain function",
    "neuroscience and neuropharmacology research",
    "neuroscience and neural engineering",
}
GENERIC_FALLBACK = "Computational & Theoretical"

# Curated research-domain buckets for a computational-neuroscience cohort.
# OpenAlex files ~half of everyone under the single subfield "Cognitive
# Neuroscience", so we classify on the person's granular *topics* (which are
# diverse) instead of that catch-all subfield. Rules are matched top-to-bottom
# against the combined text of a person's top topics + subfield + field; the
# first bucket whose keywords hit wins. Order = specific before general.
BUCKET_RULES = [
    # Matched against the first *specific* (non-generic) topic's name + field.
    ("Machine Learning & AI", [
        "artificial intelligence", "neural network", "deep learning",
        "machine learning", "computer vision", "pattern recognition",
        "reinforcement learning", "natural language", "generative"]),
    ("Computational & Theoretical", [
        "nonlinear dynamics", "pattern formation", "dynamical", "oscillat",
        "computational", "information theory", "network model", "attractor",
        "mean field", "stochastic", "chaos", "mathematical model", "bayesian"]),
    ("Sensory & Perception", [
        "visual", "vision", "retina", "photoreceptor", "auditory", "hearing",
        "olfact", "somatosensory", "perception", "sensory", "pain "]),
    ("Motor & Action", [
        "motor", "movement", "locomotion", "reaching", "muscle", "spinal",
        "oculomotor", "sensorimotor", "vocal", "song"]),
    ("Cognitive & Behavioral", [
        "decision", "memory", "attention", "reward", "cognitive", "behavior",
        "hippocamp", "prefrontal", "emotion", "psychology", "language",
        "navigation", "sleep", "consciousness"]),
    ("Cellular & Molecular", [
        "synap", "ion channel", "molecular", "receptor", "neurotransmitter",
        "gene", "protein", "dendrit", "excitability", "cell", "insect",
        "biochemistry", "development", "stem cell", "glia"]),
    ("Systems & Circuits", [
        "connectom", "microcircuit", "neural circuit", "cortical circuit",
        "circuit", "cortical column", "electrophysiolog", "network"]),
    ("Biophysics & Physics", [
        "biophysic", "physics", "quantum", "statistical mechanics", "fluid",
        "optics", "condensed matter", "photon"]),
    ("Methods, Stats & Imaging", [
        "imaging", "fmri", "eeg", "meg", "signal processing", "statistic",
        "microscopy", "brain-computer", "tomography", "spectroscopy"]),
    ("Clinical & Disease", [
        "disease", "clinical", "disorder", "neurology", "psychiatr", "epilep",
        "parkinson", "alzheimer", "stroke", "tumor", "cancer", "pharmacolog", "therap"]),
]


def fetch_topics(openalex_id, refresh):
    """Return the OpenAlex ``topics`` list for an author id (free single GET)."""
    url = ec.openalex_url(f"authors/{openalex_id}",
                          {"select": "display_name,topics"})
    data = ec.http_get_json(
        url, cache_key=("openalex_authors", f"{openalex_id}.json"), refresh=refresh)
    if not data:
        return []
    return data.get("topics") or []


def classify(topics):
    """Assign a person to a curated domain bucket from their OpenAlex topics.

    Topics arrive sorted by ``count`` (descending). Returns
    (bucket, primary_subfield, [top topic display names]); bucket is None when
    the author has no topics, "Other" when nothing matches a rule.
    """
    if not topics:
        return None, None, []

    subfield = ((topics[0].get("subfield") or {}).get("display_name")) or None
    names = [t.get("display_name") for t in topics[:HOW_MANY_TOPICS] if t.get("display_name")]

    # Pick the first topic that isn't generic filler; classify on its name+field.
    chosen = None
    for t in topics[:CLASSIFY_TOPICS]:
        dn = (t.get("display_name") or "").strip()
        if dn and dn.lower() not in GENERIC_TOPICS:
            chosen = t
            break
    if chosen is None:
        return GENERIC_FALLBACK, subfield, names  # profile is entirely generic dynamics

    text = ((chosen.get("display_name") or "") + " "
            + (chosen.get("field") or {}).get("display_name", "")).lower()
    bucket = "Other"
    for name, keywords in BUCKET_RULES:
        if any(kw in text for kw in keywords):
            bucket = name
            break
    return bucket, subfield, names


def write_domains_js(subfield_counts, by_person, out_path):
    """Emit domains.js — same header/JSON style as enrich_collabs.write_collabs_js."""
    ranked = [{"name": name, "count": count}
              for name, count in sorted(subfield_counts.items(),
                                        key=lambda kv: (-kv[1], kv[0]))]
    payload = {"subfields": ranked, "byPerson": by_person}
    header = (
        "// MCN Alumni Network — research domains\n"
        "// Generated by enrich_domains.py from OpenAlex author topics.\n"
        "// Each person's `domain` is their top topic's subfield; `topics` are\n"
        "// their most-published research areas. Regenerate with:\n"
        "//     python3 enrich_domains.py\n\n"
    )
    body = "const MCN_DOMAINS = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(header + body)


def main():
    ap = argparse.ArgumentParser(description="Label MCN people with research domains.")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached author records and re-fetch")
    args = ap.parse_args()

    ec.load_dotenv()
    ec._AUTH["email"] = os.environ.get("MCN_CONTACT_EMAIL")
    ec._AUTH["api_key"] = os.environ.get("OPENALEX_API_KEY")

    resolved_path = os.path.join(HERE, "resolved_authors.json")
    if not os.path.exists(resolved_path):
        sys.exit("resolved_authors.json not found — run enrich_collabs.py first.")
    with open(resolved_path, "r", encoding="utf-8") as fh:
        resolved = json.load(fh)

    print(f"Fetching topics for {len(resolved)} resolved people "
          f"(single-record GETs are free) ...", file=sys.stderr)

    bucket_counts = {}
    by_person = {}
    no_topics = 0
    for i, (name, info) in enumerate(sorted(resolved.items()), 1):
        oid = info.get("openalex_id")
        if not oid:
            continue
        topics = fetch_topics(oid, args.refresh)
        bucket, subfield, top_names = classify(topics)
        by_person[name] = {"domain": bucket, "subfield": subfield, "topics": top_names}
        if bucket:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        else:
            no_topics += 1
        if i % 100 == 0:
            print(f"  [{i}/{len(resolved)}] ...", file=sys.stderr)

    write_domains_js(bucket_counts, by_person, os.path.join(HERE, "domains.js"))

    print(f"\nDone.\n"
          f"  People labeled:   {len(by_person)}\n"
          f"  Domain buckets:   {len(bucket_counts)}\n"
          f"  No topic data:    {no_topics}\n"
          f"  Output:           domains.js", file=sys.stderr)
    top = sorted(bucket_counts.items(), key=lambda kv: -kv[1])
    print("  Distribution: " + ", ".join(f"{n} ({c})" for n, c in top), file=sys.stderr)


if __name__ == "__main__":
    main()
