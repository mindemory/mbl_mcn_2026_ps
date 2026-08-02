#!/usr/bin/env bash
#
# build_site.sh — assemble a clean, self-contained static bundle in ./dist
#
# The bundle contains ONLY the files needed to run the graph in a browser,
# with D3 and the web fonts vendored locally so the deployed page makes zero
# third-party requests. Build-time files (Python scripts, JSON config, caches,
# and especially .env with the API key) are NEVER copied — the copy is an
# explicit allow-list, not a wildcard.
#
# Usage:
#   ./build_site.sh
# then copy ./dist into your site, e.g.:
#   rsync -a --delete dist/ /path/to/alexandria/public/mcn-network/
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
DIST="$HERE/dist"

# The URL subpath the bundle will be served at. A <base> tag with this value is
# injected so the page's relative asset paths resolve correctly regardless of
# whether the host serves it at /mcn-network, /mcn-network/, or /.../index.html.
# Change this (keep the trailing slash) if you deploy under a different path.
BASE_PATH="${MCN_BASE_PATH:-/mcn-network/}"

# Only these files are ever shipped.
RUNTIME_FILES=(index.html style.css graph.js data.js collabs.js domains.js)

echo "==> Cleaning $DIST"
rm -rf "$DIST"
mkdir -p "$DIST/vendor/fonts"

echo "==> Copying runtime files (allow-list)"
for f in "${RUNTIME_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "    ! missing required file: $f" >&2; exit 1
  fi
  cp "$f" "$DIST/$f"
  echo "    + $f"
done

echo "==> Injecting <base href=\"$BASE_PATH\">"
perl -i -pe "s{<head>}{<head>\n  <base href=\"$BASE_PATH\" />}" "$DIST/index.html"

# ── Vendor D3 locally ───────────────────────────────────────────────────────
echo "==> Vendoring D3 (d3.v7.min.js)"
if curl -fsSL "https://d3js.org/d3.v7.min.js" -o "$DIST/vendor/d3.v7.min.js"; then
  echo "    + vendor/d3.v7.min.js ($(wc -c < "$DIST/vendor/d3.v7.min.js" | tr -d ' ') bytes)"
  perl -i -pe 's{https://d3js\.org/d3\.v7\.min\.js}{vendor/d3.v7.min.js}g' "$DIST/index.html"
else
  echo "    ! D3 download failed — leaving the CDN <script> in place" >&2
fi

# ── Self-host the web fonts (best-effort) ────────────────────────────────────
# Fetch the Google Fonts CSS with a modern-browser UA (so it serves woff2),
# download each font file, and rewrite the CSS to reference the local copies.
echo "==> Self-hosting fonts"
FONTS_CSS_URL="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
FONTS_OK=0
if curl -fsSL -A "$UA" "$FONTS_CSS_URL" -o "$DIST/vendor/fonts/fonts.css"; then
  # Download each referenced woff2 and point the CSS at the local basename.
  # (Portable to macOS bash 3.2 — no mapfile.)
  count=0
  while IFS= read -r url; do
    [[ -z "$url" ]] && continue
    base="$(basename "$url")"
    curl -fsSL "$url" -o "$DIST/vendor/fonts/$base"
    perl -i -pe "s{\Q$url\E}{$base}g" "$DIST/vendor/fonts/fonts.css"
    count=$((count + 1))
  done < <(grep -oE 'https://[^)]+\.woff2' "$DIST/vendor/fonts/fonts.css" | sort -u)
  if [[ "$count" -gt 0 ]]; then
    echo "    + $count font files -> vendor/fonts/"
    FONTS_OK=1
  fi
fi

# Drop Google's preconnect line (only needed for the CDN).
perl -ni -e 'print unless /preconnect.*fonts\./' "$DIST/index.html"
if [[ "$FONTS_OK" -eq 1 ]]; then
  # Point the stylesheet <link> at the local fonts.css.
  perl -i -pe 's{<link href="https://fonts\.googleapis\.com/css2[^>]*>}{<link rel="stylesheet" href="vendor/fonts/fonts.css" />}g' "$DIST/index.html"
else
  echo "    ! font self-host failed — dropping the webfont link (system fonts will be used)" >&2
  rm -rf "$DIST/vendor/fonts"
  perl -ni -e 'print unless m{fonts\.googleapis\.com/css2}' "$DIST/index.html"
fi

# ── Minify the generated data files ──────────────────────────────────────────
# collabs.js / domains.js are pretty-printed; re-emit them compact (roughly
# halves the bytes before the host's gzip/brotli). data.js is tiny — leave it.
echo "==> Minifying data files"
node -e '
  const fs = require("fs");
  for (const f of ["collabs.js", "domains.js"]) {
    const p = "dist/" + f;
    const src = fs.readFileSync(p, "utf8");
    const m = src.match(/const\s+(\w+)\s*=([\s\S]*);\s*$/);
    if (!m) { console.error("    ! could not parse " + f); continue; }
    const obj = eval("(" + m[2] + ")");
    const before = src.length;
    fs.writeFileSync(p, "const " + m[1] + "=" + JSON.stringify(obj) + ";\n");
    const after = fs.statSync(p).size;
    console.log("    " + f + ": " + before + " -> " + after + " bytes");
  }
'

# ── Safety net: make sure nothing sensitive slipped in ───────────────────────
echo "==> Safety check"
if find "$DIST" \( -name '.env' -o -name '*.py' -o -name '*.csv' \
       -o -name 'resolved_authors.json' -o -name 'author_overrides.json' \
       -o -name 'name_aliases.json' \) | grep -q .; then
  echo "    ! FAIL: sensitive/dev file found in dist — aborting" >&2
  find "$DIST" \( -name '.env' -o -name '*.py' -o -name '*.csv' \) >&2
  exit 1
fi
echo "    OK — no secrets or dev files in dist"

echo ""
echo "Bundle ready: $DIST"
echo "Contents:"; ( cd "$DIST" && find . -type f | sort | sed 's/^/    /' )
echo ""
echo "Next: copy it into your site, then commit & push there:"
echo "  rsync -a --delete \"$DIST/\" /Users/mrugankdake/Documents/Personal/alexandria/public/mcn-network/"
