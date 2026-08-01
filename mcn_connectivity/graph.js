/* =============================================
   MCN CONNECTIVITY — D3 GRAPH ENGINE
   ============================================= */

(function () {
  "use strict";

  // ── Color palettes ─────────────────────────
  // Dynamic year→color: maps any year in the archive to a perceptual hue
  // Uses a modified rainbow that avoids pure green (hard to see on dark bg)
  const YEAR_MIN = 1988;
  const YEAR_MAX = 2030;

  function yearColor(year) {
    const t = (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN);
    // Cycle through a hand-tuned hue arc: blue→cyan→yellow→orange→pink→violet
    const hue = (220 + t * 300) % 360;
    const sat = Math.round(80 + 15 * Math.sin(t * Math.PI));
    const lit = Math.round(58 + 10 * Math.cos(t * Math.PI * 2));
    return `hsl(${Math.round(hue)},${sat}%,${lit}%)`;
  }

  const MULTI_COLOR = "#e2e8f0";

  const ROLE_COLORS = {
    director: "#f97316",
    faculty:  "#60a5fa",
    lecturer: "#a78bfa",
    ta:       "#34d399",
    student:  "#94a3b8",
  };

  const ROLE_RADIUS = {
    director: 14,
    faculty:  12,
    lecturer: 10,
    ta:       8,
    student:  7,
  };

  // Distinct look for the two edge layers.
  const ATTEND_COLOR = "#6366f1"; // indigo — used for multi-year co-attendance
  const COLLAB_COLOR = "#fbbf24"; // amber  — co-authorship edges stand out

  // ── Co-authorship data (optional) ──────────
  // collabs.js defines MCN_COLLABS = { edges, resolved, aliases }. If it wasn't
  // generated yet, the graph gracefully falls back to co-attendance only.
  const COLLABS  = (typeof MCN_COLLABS !== "undefined") ? MCN_COLLABS : null;
  const ALIASES  = (COLLABS && COLLABS.aliases)  || {};
  const RESOLVED = (COLLABS && COLLABS.resolved) || {};
  const HAS_COLLABS = !!(COLLABS && COLLABS.edges && COLLABS.edges.length);

  // Collapse roster spelling variants onto one canonical name so co-authorship
  // edges land on the same node the person is drawn as.
  function canon(name) { return ALIASES[name] || name; }

  // Fast lookup: "canonA|||canonB" (sorted) → { papers, years, titles }.
  const collabPairMap = new Map();
  if (COLLABS && COLLABS.edges) {
    for (const e of COLLABS.edges) {
      const key = [e.source, e.target].sort().join("|||");
      collabPairMap.set(key, e);
    }
  }

  // ── State ──────────────────────────────────
  let state = {
    selectedYears: new Set(["all"]),
    selectedRole:  "all",
    // Edge layer to display: "attend" (co-attendance), "collab" (co-authorship),
    // or "both". Default to showing both when co-authorship data is present.
    edgeMode:      HAS_COLLABS ? "both" : "attend",
    colorBy:       "year",
    layout:        "force",
    searchQuery:   "",
    selectedNode:  null,
  };

  let simulation, allNodes, allLinks, svg, linkSel, nodeSel;
  const width  = () => document.getElementById("graphContainer").clientWidth;
  const height = () => document.getElementById("graphContainer").clientHeight;

  // ── Build graph data from MCN_DATA ─────────
  function buildGraphData(yearFilter = "all", roleFilter = "all") {
    const personMap = new Map(); // name → node

    function getOrCreate(name, role, year) {
      let node = personMap.get(name);
      if (!node) {
        node = { id: name, name, roles: new Set(), years: new Set(), primaryRole: role };
        personMap.set(name, node);
      }
      node.roles.add(role);
      node.years.add(year);
      // Priority: director > faculty > lecturer > ta > student
      const priority = { director:0, faculty:1, lecturer:2, ta:3, student:4 };
      if (priority[role] < priority[node.primaryRole]) node.primaryRole = role;
      return node;
    }

    // Iterate over the years we're showing
    const yearsToShow = yearFilter === "all"
      ? MCN_DATA.years
      : [parseInt(yearFilter)].filter(y => MCN_DATA.years.includes(y));

    for (const year of yearsToShow) {
      const cohort = MCN_DATA.cohorts[year];
      if (!cohort) continue;
      const groups = [
        ["director", cohort.directors || []],
        ["faculty",  cohort.faculty   || []],
        ["lecturer", cohort.lecturers || []],
        ["ta",       cohort.tas       || []],
        ["student",  cohort.students  || []],
      ];
      for (const [role, names] of groups) {
        for (const name of names) {
          getOrCreate(canon(name), role, year);
        }
      }
    }

    // Filter by role if needed
    let nodes = Array.from(personMap.values());
    if (roleFilter !== "all") {
      nodes = nodes.filter(n => n.primaryRole === roleFilter);
    }

    const nodeSet = new Set(nodes.map(n => n.id));

    // ── Co-attendance edges: people who appeared in the same cohort year ──
    const edgeMap = new Map();
    for (const year of yearsToShow) {
      const cohort = MCN_DATA.cohorts[year];
      if (!cohort) continue;
      const all = [
        ...(cohort.directors||[]), ...(cohort.faculty||[]), ...(cohort.lecturers||[]),
        ...(cohort.tas||[]),       ...(cohort.students||[]),
      ].map(canon).filter(n => nodeSet.has(n));

      for (let i = 0; i < all.length; i++) {
        for (let j = i + 1; j < all.length; j++) {
          if (all[i] === all[j]) continue; // aliasing can collapse two names
          const key = [all[i], all[j]].sort().join("|||");
          if (!edgeMap.has(key)) {
            edgeMap.set(key, { source: all[i], target: all[j], type: "attend",
                               years: new Set(), weight: 0 });
          }
          edgeMap.get(key).years.add(year);
          edgeMap.get(key).weight++;
        }
      }
    }
    const attendLinks = Array.from(edgeMap.values()).filter(
      l => nodeSet.has(l.source) && nodeSet.has(l.target)
    );

    // ── Co-authorship edges: from MCN_COLLABS, both endpoints in view ──
    const collabLinks = [];
    if (COLLABS && COLLABS.edges) {
      for (const e of COLLABS.edges) {
        if (!nodeSet.has(e.source) || !nodeSet.has(e.target)) continue;
        collabLinks.push({
          source: e.source, target: e.target, type: "collab",
          papers: e.papers, weight: e.papers,
          years: new Set(e.years || []), titles: e.sampleTitles || [],
        });
      }
    }

    // ── Choose the active layer(s) ──
    let links;
    if (state.edgeMode === "attend")      links = attendLinks;
    else if (state.edgeMode === "collab") links = collabLinks;
    else                                  links = attendLinks.concat(collabLinks);

    // Attach degree to each node
    const degMap = new Map();
    for (const l of links) {
      degMap.set(l.source, (degMap.get(l.source)||0) + 1);
      degMap.set(l.target, (degMap.get(l.target)||0) + 1);
    }
    for (const n of nodes) {
      n.degree = degMap.get(n.id) || 0;
    }

    return { nodes, links };
  }

  // ── Color resolver ─────────────────────────
  function nodeColor(d) {
    if (state.colorBy === "role") {
      return ROLE_COLORS[d.primaryRole] || "#64748b";
    }
    if (state.colorBy === "degree") {
      return d3.interpolateViridis(Math.min(d.degree / 50, 1));
    }
    // By year — use earliest year for people who attended multiple
    const yrs = Array.from(d.years).sort();
    if (yrs.length === 1) return yearColor(yrs[0]);
    return MULTI_COLOR; // multi-year attendees → white
  }

  // ── Legend ─────────────────────────────────
  function renderLegend() {
    const el = document.getElementById("legend");
    el.innerHTML = "";

    if (state.colorBy === "year") {
      // Gradient bar spanning the full year range
      const years = [...MCN_DATA.years].sort();
      const first = years[0], last = years[years.length - 1];
      // Build CSS gradient from sampled year colors
      const stops = years.map((y, i) => `${yearColor(y)} ${(i / (years.length - 1) * 100).toFixed(1)}%`).join(", ");
      const bar = document.createElement("div");
      bar.style.cssText = `width:160px;height:10px;border-radius:5px;background:linear-gradient(to right,${stops});flex-shrink:0`;
      const labelEl = document.createElement("span");
      labelEl.style.cssText = "font-size:11px;color:#8892a4;white-space:nowrap";
      labelEl.textContent = `${first} → ${last}`;
      const wrap = document.createElement("div");
      wrap.className = "legend-item";
      wrap.appendChild(bar);
      wrap.appendChild(labelEl);
      el.appendChild(wrap);
      // Multi-year chip
      const multi = document.createElement("div");
      multi.className = "legend-item";
      multi.innerHTML = `<div class="legend-dot" style="background:${MULTI_COLOR}"></div><span>Multi-year</span>`;
      el.appendChild(multi);
    } else if (state.colorBy === "role") {
      for (const [r, c] of Object.entries(ROLE_COLORS)) {
        const item = document.createElement("div");
        item.className = "legend-item";
        item.innerHTML = `<div class="legend-dot" style="background:${c}"></div><span>${r.charAt(0).toUpperCase() + r.slice(1)}</span>`;
        el.appendChild(item);
      }
    } else {
      for (const [label, t] of [["Low", 0], ["Mid", 0.5], ["High", 1]]) {
        const item = document.createElement("div");
        item.className = "legend-item";
        item.innerHTML = `<div class="legend-dot" style="background:${d3.interpolateViridis(t)}"></div><span>${label} connections</span>`;
        el.appendChild(item);
      }
    }

    // Edge-type key — which layer(s) are currently drawn.
    const edgeItems = [];
    if (state.edgeMode !== "collab") {
      edgeItems.push([ATTEND_COLOR, "Co-attendance"]);
    }
    if (state.edgeMode !== "attend" && HAS_COLLABS) {
      edgeItems.push([COLLAB_COLOR, "Co-authorship"]);
    }
    for (const [color, label] of edgeItems) {
      const item = document.createElement("div");
      item.className = "legend-item";
      item.innerHTML = `<div class="legend-line" style="background:${color}"></div><span>${label}</span>`;
      el.appendChild(item);
    }
  }

  // ── Update header stats ─────────────────────
  function updateStats(nodes, links) {
    document.getElementById("statNodes").textContent = nodes.length;
    document.getElementById("statEdges").textContent = links.length;
    document.getElementById("statYears").textContent = MCN_DATA.years.length;
  }

  // ── Main render ────────────────────────────
  function render() {
    const yearFilter = state.selectedYears.has("all") ? "all" : [...state.selectedYears][0];
    const { nodes, links } = buildGraphData(yearFilter, state.selectedRole);
    allNodes = nodes;
    allLinks = links;
    updateStats(nodes, links);
    renderLegend();

    const container = document.getElementById("graphContainer");
    const W = width(), H = height();

    // Clear old SVG content
    svg.selectAll("*").remove();

    // Arrow marker
    svg.append("defs").append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 -4 8 8")
      .attr("refX", 14).attr("refY", 0)
      .attr("markerWidth", 6).attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
        .attr("d", "M0,-4L8,0L0,4")
        .attr("fill", "rgba(255,255,255,0.2)");

    const g = svg.append("g");

    // Zoom behaviour
    const zoom = d3.zoom()
      .scaleExtent([0.1, 8])
      .on("zoom", e => g.attr("transform", e.transform));
    svg.call(zoom);

    // Links — co-authorship edges are brighter/thicker so they read on top of
    // the fainter co-attendance mesh.
    linkSel = g.append("g").attr("class", "links")
      .selectAll("line")
      .data(links)
      .join("line")
        .attr("class", d => "link " + d.type)
        .attr("stroke", d => {
          if (d.type === "collab") return COLLAB_COLOR;
          const yrs = Array.from(d.years).sort();
          return yrs.length === 1 ? yearColor(yrs[0]) : ATTEND_COLOR;
        })
        .attr("stroke-width", d => d.type === "collab"
          ? Math.sqrt(d.weight) * 1.2 + 0.6
          : Math.sqrt(d.weight) * 0.7 + 0.3)
        .attr("stroke-opacity", d => d.type === "collab" ? 0.6 : 0.15);

    // Nodes
    nodeSel = g.append("g").attr("class", "nodes")
      .selectAll(".node")
      .data(nodes, d => d.id)
      .join("g")
        .attr("class", "node")
        .attr("data-id", d => d.id)
        .call(
          d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended)
        )
        .on("click", (event, d) => {
          event.stopPropagation();
          selectNode(d);
        })
        .on("mouseover", (event, d) => showTooltip(event, d))
        .on("mousemove", (event) => moveTooltip(event))
        .on("mouseout", hideTooltip);

    nodeSel.append("circle")
      .attr("r", d => ROLE_RADIUS[d.primaryRole] || 7)
      .attr("fill", d => nodeColor(d))
      .attr("stroke", d => {
        const c = nodeColor(d);
        return d3.color(c) ? d3.color(c).brighter(1).toString() : "#fff";
      })
      .attr("stroke-width", 1.5)
      .attr("fill-opacity", 0.85);

    nodeSel.append("text")
      .attr("dy", d => (ROLE_RADIUS[d.primaryRole] || 7) + 11)
      .attr("text-anchor", "middle")
      .text(d => {
        // Show last name only if too crowded, full name if director/faculty
        if (d.primaryRole === "director" || d.primaryRole === "faculty") return d.name;
        const parts = d.name.split(" ");
        return parts[parts.length - 1];
      })
      .attr("font-size", d => d.primaryRole === "student" ? "9px" : "10px")
      .attr("fill", "rgba(255,255,255,0.75)");

    // Force simulation
    if (simulation) simulation.stop();
    simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links)
        .id(d => d.id)
        .distance(d => {
          const base = 60;
          // Weaker link distance for same-year connections
          return base - (d.weight - 1) * 5;
        })
        .strength(0.3)
      )
      .force("charge", d3.forceManyBody().strength(d => {
        const r = ROLE_RADIUS[d.primaryRole] || 7;
        return -(r * r * 4);
      }))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide(d => (ROLE_RADIUS[d.primaryRole] || 7) + 6))
      .alpha(1)
      .on("tick", ticked);

    // Radial layout override
    if (state.layout === "radial") {
      applyRadialLayout(nodes, links, W, H);
    }

    // Click outside to deselect
    svg.on("click", () => {
      clearSelection();
      closeDetail();
    });

    applySearch();
  }

  function ticked() {
    if (!linkSel || !nodeSel) return;
    linkSel
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);
    nodeSel.attr("transform", d => `translate(${d.x},${d.y})`);
  }

  // ── Drag ───────────────────────────────────
  function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
  }
  function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
  function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null; d.fy = null;
  }

  // ── Radial layout ──────────────────────────
  function applyRadialLayout(nodes, links, W, H) {
    simulation.stop();
    const R = Math.min(W, H) * 0.38;
    const cx = W / 2, cy = H / 2;
    // Group by primary year
    const yearGroups = {};
    for (const n of nodes) {
      const y = [...n.years][0] || "multi";
      if (!yearGroups[y]) yearGroups[y] = [];
      yearGroups[y].push(n);
    }
    let arcStart = 0;
    for (const [yr, grp] of Object.entries(yearGroups)) {
      const arcLen = (2 * Math.PI * grp.length) / nodes.length;
      grp.forEach((n, i) => {
        const angle = arcStart + (i / grp.length) * arcLen;
        n.x = cx + R * Math.cos(angle);
        n.y = cy + R * Math.sin(angle);
        n.fx = n.x; n.fy = n.y;
      });
      arcStart += arcLen;
    }
    ticked();
    // Release after a tick so dragging still works
    setTimeout(() => {
      for (const n of nodes) { n.fx = null; n.fy = null; }
    }, 100);
  }

  // ── Tooltip ────────────────────────────────
  function showTooltip(event, d) {
    const tip = document.getElementById("tooltip");
    tip.classList.remove("hidden");
    const career = MCN_DATA.careerInfo[d.name] || null;
    const yearChips = [...d.years].sort().map(y => {
      const yc = yearColor(y);
      return `<span class="tooltip-year-chip" style="background:${yc}22;color:${yc};border:1px solid ${yc}44">${y}</span>`;
    }).join("");
    let collabCount = 0;
    if (COLLABS && COLLABS.edges) {
      for (const e of COLLABS.edges) {
        if (e.source === d.id || e.target === d.id) collabCount++;
      }
    }
    tip.innerHTML = `
      <div class="tooltip-name">${d.name}</div>
      <div class="tooltip-meta">
        <b style="color:${ROLE_COLORS[d.primaryRole]}">${d.primaryRole}</b>
        ${career ? ` · ${career.affiliation}` : ""}
      </div>
      <div class="tooltip-meta">${d.degree} connection${d.degree !== 1 ? "s" : ""}${
        HAS_COLLABS ? ` · ${collabCount} co-author${collabCount !== 1 ? "s" : ""}` : ""
      }</div>
      <div class="tooltip-years">${yearChips}</div>
    `;
    moveTooltip(event);
  }

  function moveTooltip(event) {
    const tip = document.getElementById("tooltip");
    const pad = 14;
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    if (x + 250 > window.innerWidth)  x = event.clientX - 250 - pad;
    if (y + 140 > window.innerHeight) y = event.clientY - 140 - pad;
    tip.style.left = x + "px";
    tip.style.top  = y + "px";
  }

  function hideTooltip() {
    document.getElementById("tooltip").classList.add("hidden");
  }

  // ── Node selection & detail panel ──────────
  function selectNode(d) {
    state.selectedNode = d;
    highlightNeighbors(d);
    openDetail(d);
  }

  function highlightNeighbors(d) {
    if (!nodeSel || !linkSel) return;
    const neighborIds = new Set([d.id]);
    allLinks.forEach(l => {
      const sid = typeof l.source === "object" ? l.source.id : l.source;
      const tid = typeof l.target === "object" ? l.target.id : l.target;
      if (sid === d.id) neighborIds.add(tid);
      if (tid === d.id) neighborIds.add(sid);
    });
    nodeSel
      .classed("highlighted", n => n.id === d.id)
      .classed("dimmed",      n => !neighborIds.has(n.id));
    linkSel
      .classed("highlighted", l => {
        const sid = typeof l.source === "object" ? l.source.id : l.source;
        const tid = typeof l.target === "object" ? l.target.id : l.target;
        return sid === d.id || tid === d.id;
      })
      .classed("dimmed", l => {
        const sid = typeof l.source === "object" ? l.source.id : l.source;
        const tid = typeof l.target === "object" ? l.target.id : l.target;
        return sid !== d.id && tid !== d.id;
      });
  }

  function clearSelection() {
    state.selectedNode = null;
    if (!nodeSel || !linkSel) return;
    nodeSel.classed("highlighted dimmed search-hit", false);
    linkSel.classed("highlighted dimmed", false);
  }

  function openDetail(d) {
    const panel = document.getElementById("detailPanel");
    const career = MCN_DATA.careerInfo[d.name] || null;
    const initials = d.name.split(" ").map(p => p[0]).slice(0,2).join("");
    const color = nodeColor(d);

    // Find cohort-mates
    const cohortMates = {};
    for (const year of d.years) {
      const cohort = MCN_DATA.cohorts[year];
      if (!cohort) continue;
      const all = [...(cohort.directors||[]), ...(cohort.faculty||[]),
                   ...(cohort.lecturers||[]), ...(cohort.tas||[]), ...(cohort.students||[])];
      cohortMates[year] = all.filter(n => n !== d.name);
    }

    const yearsRows = [...d.years].sort().map(y => {
      let role = "attendee";
      const cohort = MCN_DATA.cohorts[y];
      if (cohort) {
        if ((cohort.directors||[]).includes(d.name)) role = "Director";
        else if ((cohort.faculty||[]).includes(d.name)) role = "Faculty";
        else if ((cohort.lecturers||[]).includes(d.name)) role = "Lecturer";
        else if ((cohort.tas||[]).includes(d.name)) role = "TA";
        else role = "Student";
      }
      return `<div class="detail-year-row">
        <div class="detail-year-dot" style="background:${yearColor(y)}"></div>
        <span class="detail-year-label">MCN ${y}</span>
        <span class="detail-year-role">${role}</span>
      </div>`;
    }).join("");

    // Co-authors of this person (independent of the active edge layer), with
    // shared-paper counts, sorted by how much they've published together.
    const collaborators = [];
    if (COLLABS && COLLABS.edges) {
      const nameSet = new Set(allNodes.map(n => n.id));
      for (const e of COLLABS.edges) {
        let other = null;
        if (e.source === d.id) other = e.target;
        else if (e.target === d.id) other = e.source;
        if (other && nameSet.has(other)) collaborators.push({ name: other, papers: e.papers });
      }
      collaborators.sort((a, b) => b.papers - a.papers);
    }
    const collabList = collaborators.slice(0, 12).map(c =>
      `<a onclick="window.__mcnSelectById('${c.name.replace(/'/g, "\\'")}')" style="color:${COLLAB_COLOR}">${c.name}</a>` +
      `<span class="detail-collab-count">${c.papers}</span>`
    ).join("");
    const collabExtra = collaborators.length > 12 ? ` + ${collaborators.length - 12} more` : "";

    // Scholarly identity links, when this person was resolved.
    const ident = RESOLVED[d.name] || null;
    const identLinks = [];
    if (ident) {
      if (ident.openalex) identLinks.push(`<a href="https://openalex.org/${ident.openalex}" target="_blank">OpenAlex</a>`);
      if (ident.orcid)    identLinks.push(`<a href="https://orcid.org/${ident.orcid}" target="_blank">ORCID</a>`);
    }

    const neighborNodes = allNodes.filter(n => {
      if (n.id === d.id) return false;
      return allLinks.some(l => {
        const sid = typeof l.source === "object" ? l.source.id : l.source;
        const tid = typeof l.target === "object" ? l.target.id : l.target;
        return (sid === d.id && tid === n.id) || (tid === d.id && sid === n.id);
      });
    });
    // Sort by role priority
    const pri = { director:0, faculty:1, lecturer:2, ta:3, student:4 };
    neighborNodes.sort((a, b) => (pri[a.primaryRole]||5) - (pri[b.primaryRole]||5));
    const neighborList = neighborNodes.slice(0, 8).map(n =>
      `<a onclick="window.__mcnSelectById('${n.id}')" style="color:${ROLE_COLORS[n.primaryRole]||'#aaa'}">${n.name}</a>`
    ).join(", ");
    const extra = neighborNodes.length > 8 ? ` + ${neighborNodes.length - 8} more` : "";

    document.getElementById("detailContent").innerHTML = `
      <div class="detail-avatar" style="background:${color}22;border:1px solid ${color}44;color:${color}">${initials}</div>
      <div class="detail-name">${d.name}</div>
      <span class="detail-role-badge" style="background:${ROLE_COLORS[d.primaryRole]}22;color:${ROLE_COLORS[d.primaryRole]};border:1px solid ${ROLE_COLORS[d.primaryRole]}44">${d.primaryRole}</span>

      ${career ? `
      <div class="detail-section">
        <div class="detail-section-title">Current position</div>
        <div class="detail-career-box">
          <div class="detail-career-institution">${career.affiliation}</div>
          <div class="detail-career-role">${career.role}</div>
        </div>
      </div>` : ""}

      ${identLinks.length ? `
      <div class="detail-section">
        <div class="detail-section-title">Scholarly profile</div>
        <div class="detail-ident-links">${identLinks.join(" · ")}</div>
      </div>` : ""}

      <div class="detail-section">
        <div class="detail-section-title">MCN appearances</div>
        ${yearsRows}
      </div>

      ${collaborators.length > 0 ? `
      <div class="detail-section">
        <div class="detail-section-title">Co-authors (${collaborators.length})</div>
        <div class="detail-collabs">${collabList}${collabExtra}</div>
      </div>` : ""}

      ${neighborNodes.length > 0 ? `
      <div class="detail-section">
        <div class="detail-section-title">Connected to (${neighborNodes.length})</div>
        <div class="detail-connections">${neighborList}${extra}</div>
      </div>` : ""}
    `;

    panel.classList.remove("hidden");
    setTimeout(() => panel.classList.add("visible"), 10);
  }

  function closeDetail() {
    const panel = document.getElementById("detailPanel");
    panel.classList.remove("visible");
  }

  // Global hook for clicking neighbor names
  window.__mcnSelectById = function(id) {
    const node = allNodes.find(n => n.id === id);
    if (node) selectNode(node);
  };

  // ── Search ─────────────────────────────────
  function applySearch() {
    if (!nodeSel) return;
    const q = state.searchQuery.toLowerCase().trim();
    if (!q) {
      nodeSel.classed("search-hit dimmed", false);
      linkSel.classed("dimmed", false);
      return;
    }
    const hits = new Set();
    for (const n of allNodes) {
      const career = MCN_DATA.careerInfo[n.name];
      const haystack = [n.name, career?.affiliation, career?.role].filter(Boolean).join(" ").toLowerCase();
      if (haystack.includes(q)) hits.add(n.id);
    }
    nodeSel
      .classed("search-hit", n => hits.has(n.id))
      .classed("dimmed",     n => !hits.has(n.id));
    linkSel.classed("dimmed", l => {
      const sid = typeof l.source === "object" ? l.source.id : l.source;
      const tid = typeof l.target === "object" ? l.target.id : l.target;
      return !hits.has(sid) && !hits.has(tid);
    });
  }

  // ── Year buttons ────────────────────────────────
  function buildYearButtons() {
    const container = document.getElementById("yearPills");
    const allBtn = document.createElement("button");
    allBtn.className = "year-btn active";
    allBtn.dataset.year = "all";
    allBtn.textContent = "All";
    allBtn.style.background = "#4f46e5";
    allBtn.onclick = () => setYearFilter("all");
    container.appendChild(allBtn);

    for (const y of [...MCN_DATA.years].sort()) {
      const btn = document.createElement("button");
      btn.className = "year-btn";
      btn.dataset.year = String(y);
      btn.textContent = String(y);
      btn.onclick = () => setYearFilter(String(y));
      container.appendChild(btn);
    }
  }

  function setYearFilter(year) {
    document.querySelectorAll(".year-btn").forEach(b => b.classList.remove("active"));
    document.querySelector(`.year-btn[data-year="${year}"]`).classList.add("active");
    state.selectedYears = year === "all" ? new Set(["all"]) : new Set([year]);
    clearSelection(); closeDetail();
    render();
  }

  // ── Role buttons ───────────────────────────
  function initRoleButtons() {
    document.querySelectorAll("#rolePills .role-btn").forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll("#rolePills .role-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.selectedRole = btn.dataset.role;
        clearSelection(); closeDetail();
        render();
      };
    });
  }

  // ── Edge-mode buttons ──────────────────────
  function initEdgeModeButtons() {
    const pills = document.getElementById("edgeModePills");
    if (!pills) return;
    // Reflect the default (and disable the co-authorship option when there's
    // no data for it) before wiring up clicks.
    pills.querySelectorAll(".role-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.edgemode === state.edgeMode);
      if (b.dataset.edgemode !== "attend" && !HAS_COLLABS) {
        b.disabled = true;
        b.title = "Run enrich_collabs.py to generate co-authorship edges";
      }
    });
    pills.querySelectorAll(".role-btn").forEach(btn => {
      btn.onclick = () => {
        if (btn.disabled) return;
        pills.querySelectorAll(".role-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.edgeMode = btn.dataset.edgemode;
        clearSelection(); closeDetail();
        render();
      };
    });
  }

  // ── Color-by buttons ───────────────────────
  function initColorByButtons() {
    document.querySelectorAll("#colorByPills .role-btn").forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll("#colorByPills .role-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.colorBy = btn.dataset.colorby;
        // Re-color nodes in place (no full re-render needed)
        if (nodeSel) {
          nodeSel.select("circle")
            .attr("fill", d => nodeColor(d))
            .attr("stroke", d => {
              const c = nodeColor(d);
              return d3.color(c) ? d3.color(c).brighter(1).toString() : "#fff";
            });
        }
        renderLegend();
      };
    });
  }

  // ── Layout buttons ─────────────────────────
  function initLayoutButtons() {
    document.getElementById("btnForce").onclick = () => {
      state.layout = "force";
      document.getElementById("btnForce").classList.add("active");
      document.getElementById("btnRadial").classList.remove("active");
      // Release all fixed positions
      if (allNodes) allNodes.forEach(n => { n.fx = null; n.fy = null; });
      if (simulation) simulation.alpha(0.8).restart();
    };
    document.getElementById("btnRadial").onclick = () => {
      state.layout = "radial";
      document.getElementById("btnForce").classList.remove("active");
      document.getElementById("btnRadial").classList.add("active");
      if (allNodes && allLinks) applyRadialLayout(allNodes, allLinks, width(), height());
    };
  }

  // ── Search input ───────────────────────────
  function initSearch() {
    const inp = document.getElementById("searchInput");
    inp.addEventListener("input", () => {
      state.searchQuery = inp.value;
      clearSelection(); closeDetail();
      applySearch();
    });
    inp.addEventListener("keydown", e => {
      if (e.key === "Escape") { inp.value = ""; state.searchQuery = ""; applySearch(); }
    });
  }

  // ── Close detail panel ─────────────────────
  document.getElementById("closeDetail").onclick = () => {
    clearSelection(); closeDetail();
  };

  // ── Init ───────────────────────────────────
  function init() {
    svg = d3.select("#graph");

    buildYearButtons();
    initRoleButtons();
    initEdgeModeButtons();
    initColorByButtons();
    initLayoutButtons();
    initSearch();
    render();

    // Re-render on resize
    let resizeTimer;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (simulation) {
          simulation.force("center", d3.forceCenter(width() / 2, height() / 2));
          simulation.alpha(0.3).restart();
        }
      }, 200);
    });
  }

  // Run after DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
