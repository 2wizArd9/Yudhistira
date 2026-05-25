## TODO - Yudhishthira AI Analytics Dashboard Modernization

### Step 1: Understand current analytics rendering
- [x] Inspect `analytics.py`, `templates/index.html`, `static/css/style.css`

### Step 2: Upgrade Plotly figures (analytics.py)
- [x] Basic formatting/health checks for `analytics.py` (py_compile passes)
- [ ] Add shared dark neon Plotly theme helper

- [ ] Improve/replace charts: donut for fake vs real, better gauge, smooth confidence line, richer trusted matches
- [ ] Add animated transitions and hover templates
- [ ] Extend `stats_json` with metrics needed for KPI cards (without removing existing keys)

### Step 3: Upgrade analytics UI (templates/index.html)
- [ ] Replace existing 4 static KPI cards with 8-10 requested KPI cards using `stats_json`
- [ ] Add chart-section layout polish (cards, headings, premium microcopy)

### Step 4: Upgrade CSS (static/css/style.css)
- [ ] Add dedicated analytics section styling: glass panels, neon border accents, chart container polish
- [ ] Add KPI card styling: counters, trend arrows/icons, shimmer
- [ ] Add Plotly container styles: transparent background, modebar/legend/hover improvements
- [ ] Ensure responsiveness matches requested grid behavior

### Step 5: Optional JS polish (static/js/main.js)
- [ ] Add lightweight skeleton/entrance animations for KPI cards and charts
- [ ] Add animated counters if feasible without breaking SSR

### Step 6: Verification
- [ ] Run `python front.py`
- [ ] Confirm charts and KPI cards render correctly after restart
- [ ] Validate responsiveness (mobile/tablet/desktop)

