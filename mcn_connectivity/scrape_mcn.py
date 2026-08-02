#!/usr/bin/env python3
"""
MCN Archive Scraper
Scrapes all MCN cohort rosters from the MBL History Archive and
outputs a data.js file ready for the mcn_connectivity visualizer.

Usage:
    python3 scrape_mcn.py

Writes: data.js (in the same directory)
"""

import urllib.request
import urllib.error
import re
import time
import json
import sys
from html.parser import HTMLParser

BASE_URL = "https://history.archives.mbl.edu"

# All cohort URLs — discovered from the course-group index page
COHORT_URLS = {
    1988: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-1988",
    1989: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-1989",
    1990: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-1990",
    1991: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-1991",
    1992: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-1992",
    1993: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-1993",
    1994: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-1994",
    1995: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-1995",
    1996: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-1996",
    1997: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-1997",
    1998: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-1998",
    1999: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-1999",
    2000: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2000",
    2001: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2001",
    2002: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2002",
    2003: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2003",
    2004: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2004",
    2005: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2005",
    2006: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2006",
    2007: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2007",
    2008: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2008",
    2009: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2009",
    2010: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2010",
    2011: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2011",
    2012: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2012",
    2013: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2013",
    2014: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2014",
    2015: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2015",
    2016: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience-2016",
    # 2017 uses a different slug
    2017: f"{BASE_URL}/people-and-courses/course/methods-computational-neuroscience",
}

# Role label normalisation — archive uses various capitalizations, spellings,
# AND PLURALIZATIONS across years — a raw exact-match lookup against only the
# singular forms silently miscategorized a lot of people as "student" for
# years whose page happened to use a variant this map didn't cover (found by
# auditing every year's raw HTML: 1990/1991 used "Lecturer/Instructor" for
# what other years call Faculty; 1992 used "Computer Manager"; 2009 pluralized
# every label ("Lecturers", "Course Directors", "Teaching Assistants",
# "Students"); 2017 is served from a different URL/template with its own
# vocabulary entirely). All of those variants are covered below; see
# handle_data()'s warning if a *new* one shows up in a future scrape.
ROLE_MAP = {
    "student":          "student",
    "students":         "student",
    "faculty":          "faculty",
    "lecturer":         "lecturer",
    "lecturers":        "lecturer",
    "teaching assistant": "ta",
    "teaching assistants": "ta",
    "ta":               "ta",
    "director":         "director",
    "course director":  "director",
    "course directors": "director",
    "co-director":      "director",
    "lab instructor":   "lecturer",
    "course assistant": "assistant",
    "course coordinator": "assistant",
    "coordinator":      "assistant",
    "computer manager": "assistant",
    "observer":         "student",   # treat observers as students
    "research assistant": "ta",
    # 1990/1991 used a single combined label for what other years call
    # "Faculty" — confirmed by cross-referencing the same names' roles in
    # adjacent years (Sejnowski, Van Essen, Miller, et al.).
    "lecturer/instructor": "faculty",
    # 2017-page vocabulary:
    "course lecturer":  "lecturer",
    "scientific course consultant": "faculty",
    "chief scientific course consultant (formerly course section leader)": "faculty",
    "research facilitator": "ta",
}


class AttendeeParser(HTMLParser):
    """Parse MBL archive HTML and extract role→[name] mappings."""

    def __init__(self, year=None):
        super().__init__()
        self.year = year
        self.cohort = {
            "directors": [],
            "faculty": [],
            "lecturers": [],
            "tas": [],
            "students": [],
            "assistants": [],
        }
        self._in_li = False
        self._current_role = None
        self._current_role_raw = None
        self._current_name = None
        self._capture_text = False
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "li":
            self._in_li = True
            self._current_role = None
            self._current_role_raw = None
            self._current_name = None
        if tag == "a" and self._in_li:
            href = attrs_dict.get("href", "")
            if "/people-and-courses/person/" in href:
                self._capture_text = True

    def handle_endtag(self, tag):
        if tag == "li":
            self._in_li = False
            self._capture_text = False
        if tag == "a":
            self._capture_text = False

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        if self._in_li:
            # Role labels come before the <a> tag in the li text
            lower = data.lower()
            if lower in ROLE_MAP:
                self._current_role = ROLE_MAP[lower]
                self._current_role_raw = data
                return
            # Name inside <a>
            if self._capture_text:
                self._current_name = data
                self._flush()

    def _flush(self):
        if not self._current_name:
            return
        name = self._current_name.strip()
        if not name:
            return
        if self._current_role is None:
            # An unrecognized role label — falling back to "student" here would
            # silently mislabel people (this happened for the entire 2017 page
            # before its role vocabulary was added to ROLE_MAP). Warn loudly so
            # a future vocabulary change gets noticed immediately instead of
            # baked silently into data.js.
            print(f"  ! unmapped role {self._current_role_raw!r} for {name!r} "
                  f"(year {self.year}) — defaulting to student; add it to ROLE_MAP",
                  file=sys.stderr)
        role = self._current_role or "student"
        bucket = {
            "director":  "directors",
            "faculty":   "faculty",
            "lecturer":  "lecturers",
            "ta":        "tas",
            "student":   "students",
            "assistant": "assistants",
        }.get(role, "students")
        if name not in self.cohort[bucket]:
            self.cohort[bucket].append(name)
        self._current_name = None


def fetch_cohort(year, url, retries=3, delay=1.5):
    """Fetch and parse a single cohort page. Returns the cohort dict."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "MCNAlumniScraper/1.0 (educational, non-commercial)"
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            parser = AttendeeParser(year=year)
            parser.feed(html)
            return parser.cohort
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} for {year}", file=sys.stderr)
            if e.code == 404:
                return None
            time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"  Error fetching {year}: {e}", file=sys.stderr)
            time.sleep(delay * (attempt + 1))
    return None


# Manually curated career info (public sources only: lab pages, ORCID, university directories)
CAREER_INFO = {
    "Greg Corrado":         {"affiliation": "Google DeepMind", "role": "Principal Scientist"},
    "Tatyana Sharpee":      {"affiliation": "Salk Institute", "role": "Professor"},
    "Armen Stepanyants":    {"affiliation": "Northeastern University", "role": "Professor"},
    "Andrea Hasenstaub":    {"affiliation": "UCSF", "role": "Professor"},
    "Nicole Rust":          {"affiliation": "UPenn", "role": "Professor"},
    "Tim Gollisch":         {"affiliation": "University of Göttingen", "role": "Professor"},
    "Thomas Gregor":        {"affiliation": "Princeton / Pasteur", "role": "Professor"},
    "Jesse Goldberg":       {"affiliation": "Cornell University", "role": "Professor"},
    "Yoram Burak":          {"affiliation": "Hebrew University", "role": "Professor"},
    "Gasper Tkacik":        {"affiliation": "IST Austria", "role": "Professor"},
    "Simon Sponberg":       {"affiliation": "Georgia Tech", "role": "Professor"},
    "Robert Froemke":       {"affiliation": "NYU", "role": "Professor"},
    "Andrea Barreiro":      {"affiliation": "SMU", "role": "Professor"},
    "Arseny Finkelstein":   {"affiliation": "Weizmann Institute", "role": "Scientist"},
    "Alex Williams":        {"affiliation": "NYU / Flatiron", "role": "Research Scientist"},
    "Laura Lewis":          {"affiliation": "Boston University", "role": "Professor"},
    "Tommy Blanchard":      {"affiliation": "University of Rochester", "role": "Professor"},
    "Hannah Payne":         {"affiliation": "Columbia University", "role": "Research Scientist"},
    "Marcelo Mattar":       {"affiliation": "NYU", "role": "Professor"},
    "Lorenzo Fontolan":     {"affiliation": "EPFL", "role": "Research Scientist"},
    "Pooja Viswanathan":    {"affiliation": "NYU", "role": "Research Scientist"},
    "Noga Weiss Mosheiff":  {"affiliation": "Hebrew University", "role": "Scientist"},
    "Emily Mackevicius":    {"affiliation": "MIT / Broad Institute", "role": "Research Scientist"},
    "Gregory Wayne":        {"affiliation": "Google DeepMind", "role": "Research Scientist"},
    "B Aguera Y Arcas":     {"affiliation": "Google", "role": "Distinguished Scientist"},
    "Surya Ganguli":        {"affiliation": "Stanford University", "role": "Professor"},
    "Jonathan Pillow":      {"affiliation": "Princeton University", "role": "Professor"},
    "Ila Fiete":            {"affiliation": "MIT", "role": "Professor"},
    "Carlos Brody":         {"affiliation": "Princeton University", "role": "Professor"},
    "Adrienne Fairhall":    {"affiliation": "University of Washington", "role": "Professor"},
    "Peter Dayan":          {"affiliation": "Max Planck Institute", "role": "Director"},
    "Michael Berry":        {"affiliation": "Princeton University", "role": "Professor"},
    "Liam Paninski":        {"affiliation": "Columbia University", "role": "Professor"},
    "Kenneth Miller":       {"affiliation": "Columbia University", "role": "Professor"},
    "Eric Shea-Brown":      {"affiliation": "University of Washington", "role": "Professor"},
    "Sara Solla":           {"affiliation": "Northwestern University", "role": "Professor"},
    "Haim Sompolinsky":     {"affiliation": "Hebrew University / Harvard", "role": "Professor"},
    "Larry Abbott":         {"affiliation": "Columbia University", "role": "Professor"},
    "Eve Marder":           {"affiliation": "Brandeis University", "role": "Professor"},
    "William Bialek":       {"affiliation": "Princeton University", "role": "Professor"},
    "Terrence Sejnowski":   {"affiliation": "Salk Institute", "role": "Professor"},
    "Mark Goldman":         {"affiliation": "UC Davis", "role": "Professor"},
    "Michale Fee":          {"affiliation": "MIT", "role": "Professor"},
    "H Sebastian Seung":    {"affiliation": "Princeton University", "role": "Professor"},
    "Bard Ermentrout":      {"affiliation": "University of Pittsburgh", "role": "Professor"},
    "Nancy Kopell":         {"affiliation": "Boston University", "role": "Professor"},
    "David Tank":           {"affiliation": "Princeton University", "role": "Professor"},
    "Yael Niv":             {"affiliation": "Princeton University", "role": "Professor"},
}


def js_string(s):
    """Safely escape a Python string for embedding in JS source."""
    return json.dumps(s, ensure_ascii=False)


def main():
    all_data = {}
    years_found = []

    print(f"Scraping {len(COHORT_URLS)} cohort years from MBL History Archive...")
    print("(Being polite — 1.0s delay between requests)\n")

    for year in sorted(COHORT_URLS.keys()):
        url = COHORT_URLS[year]
        print(f"  [{year}] {url}")
        cohort = fetch_cohort(year, url)
        if cohort is None:
            print(f"         → SKIPPED (fetch failed)")
            continue

        total = sum(len(v) for v in cohort.values())
        print(f"         → directors:{len(cohort['directors'])} "
              f"faculty:{len(cohort['faculty'])} "
              f"lecturers:{len(cohort['lecturers'])} "
              f"tas:{len(cohort['tas'])} "
              f"students:{len(cohort['students'])} "
              f"  (total: {total})")

        all_data[year] = cohort
        years_found.append(year)
        time.sleep(1.0)  # polite crawl delay

    print(f"\nScraped {len(years_found)} cohort years successfully.")
    print("Writing data.js ...")

    # --- Build JS output ---
    lines = []
    lines.append("// MCN Alumni Network Data")
    lines.append("// Scraped from: https://history.archives.mbl.edu/people-and-courses/course-group/methods-computational-neuroscience")
    lines.append(f"// Cohort years: {min(years_found)}–{max(years_found)}")
    lines.append("// Roles: director, faculty, lecturer, ta, student, assistant")
    lines.append("")
    lines.append("const MCN_DATA = {")

    # Years array
    years_js = ", ".join(str(y) for y in years_found)
    lines.append(f"  years: [{years_js}],")
    lines.append("  cohorts: {")

    for year in years_found:
        cohort = all_data[year]
        lines.append(f"    {year}: {{")

        def arr(lst):
            if not lst:
                return "[]"
            inner = ", ".join(js_string(n) for n in lst)
            return f"[{inner}]"

        lines.append(f"      directors: {arr(cohort['directors'])},")
        lines.append(f"      faculty:   {arr(cohort['faculty'])},")
        lines.append(f"      lecturers: {arr(cohort['lecturers'])},")
        lines.append(f"      tas:       {arr(cohort['tas'])},")
        lines.append(f"      students:  {arr(cohort['students'])},")
        if cohort.get("assistants"):
            lines.append(f"      assistants:{arr(cohort['assistants'])},")
        lines.append("    },")

    lines.append("  },")
    lines.append("")
    lines.append("  // Career info — manually curated from public sources")
    lines.append("  // Add yourself: { affiliation: '...', role: '...' }")
    lines.append("  careerInfo: {")
    for name, info in CAREER_INFO.items():
        lines.append(f"    {js_string(name)}: "
                     f"{{ affiliation: {js_string(info['affiliation'])}, "
                     f"role: {js_string(info['role'])} }},")
    lines.append("  }")
    lines.append("};")

    output = "\n".join(lines) + "\n"

    import os
    here = os.path.dirname(__file__)
    out_path = os.path.join(here, "data.js")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Written: {out_path}")

    # Also emit a pure-JSON roster so downstream tooling (enrich_collabs.py)
    # can read the rosters without having to parse the JavaScript in data.js.
    roster_path = os.path.join(here, "mcn_roster.json")
    roster = {"years": years_found, "cohorts": all_data}
    with open(roster_path, "w", encoding="utf-8") as f:
        json.dump(roster, f, ensure_ascii=False, indent=2)
    print(f"Written: {roster_path}")
    print(f"\nSummary:")
    print(f"  Cohort years:  {len(years_found)}")
    total_people = set()
    for y in years_found:
        c = all_data[y]
        for lst in c.values():
            total_people.update(lst)
    print(f"  Unique people: {len(total_people)}")
    print("\nDone! Refresh the browser to see the full network.")


if __name__ == "__main__":
    main()
