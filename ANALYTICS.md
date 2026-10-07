# Usage analytics (GitHub Pages explorers)

The published explorers send interaction events to **Google Analytics 4**
through one hand-maintained file, [`docs/analytics.js`](docs/analytics.js).

- Each `assignment-N-ui-explorer/generate_docs_site.py` adds
  `<script src="../analytics.js" defer></script>` to `docs/assignment-N/index.html`.
  `docs/index.html` (the landing page) loads it directly.
- The local Flask explorers (`run_sandbox.py`) never load it, so nothing is
  tracked when students run them on their own machines.
- The explorer code is not modified. Once the page's own scripts have run,
  `analytics.js` wraps their global functions (`switchTab`, `setStudentId`,
  `logRow`, ...) and listens at the document level for clicks, changes,
  typing, scrolling, copying, visibility and errors. If a function is renamed
  in `run_sandbox.py`, only that one event stops; nothing breaks.

## Setup

1. **Create the property.** At <https://analytics.google.com>, go to
   Admin → Create → Property. Pick the Arizona (Phoenix) time zone. Then go to
   Data collection → Data streams → Add stream → Web, and enter the URL
   `https://ai-sandbox-ai-healthcare.github.io/neural-network-assignments-ui-explorer/`.
2. **Paste the Measurement ID** (`G-…`) into `CONFIG.measurementId` at the
   top of `docs/analytics.js`. Set `CONFIG.consent` (see
   [Privacy](#privacy-and-research-ethics)).
3. **Test locally** before pushing:
   `python -m http.server 8000 -d docs`, then open
   <http://localhost:8000/assignment-1/?ga_debug=1>. Every event is logged in
   the browser console as `[ga] <name> {…}` and shows up within seconds under
   GA4 Admin → DebugView.
4. **Commit and push** `docs/`. GitHub Pages redeploys. Check
   Reports → Realtime after you open the live site.
5. **Opt yourself out.** Open any live page once with `?notrack=1` in every
   browser and device you use, so your browsing doesn't mix with student data.
   The opt-out is stored in that browser's localStorage, which every page on
   the site shares; `?notrack=0` undoes it. Don't use GA4's IP-based
   "internal traffic" filter instead: students on the campus network share
   your IP ranges.

### GA4 property settings

| Where (Admin →) | Setting | Why |
|---|---|---|
| Data collection and modification → Data retention | **14 months** | The default (2 months) deletes event-level data used by Explorations. |
| Data collection | Google signals **off** | No cross-device ad identity, which is not needed for research. |
| Product links → BigQuery links | **Daily export on** | Raw, unsampled, event-level rows. This is what you analyse (see below). |
| Data streams → (stream) → Enhanced measurement | Leave page views on; the rest is optional | Its 90% scroll event never fires on the explorers, because each tab scrolls inside its own box. `scroll_depth` replaces it. |
| Custom definitions | Register the dimensions and metrics below | GA4's own reports only show parameters that are registered, and only from the day they are registered. BigQuery gets every parameter regardless. |

**BigQuery.** Create a Google Cloud project and enable the BigQuery API. The
free BigQuery sandbox works but expires tables after 60 days; attach billing to
keep data longer (storage at this scale costs cents). The daily export is free
for standard properties.

### Custom definitions to register

Event-scoped dimensions: `assignment`, `tab_name`, `from_tab`, `seed`,
`concept_title`, `el_id`, `el_text`, `el_action`, `is_optimal`, `cell_type`,
`source`, `scroller`.

Don't also register `seed` as a user-scoped dimension: GA4 refuses a parameter
name that is already registered. The event-scoped one covers it, and BigQuery
still receives the user property in its `user_properties` column.

Custom metrics (unit: milliseconds): `dwell_ms`, `from_tab_ms`,
`from_tab_active_ms`, `active_ms`, `ms_since_seed`. Standard unit: `auc`, `f1`,
`accuracy`, `eval_index`.

Don't register `page_load_id`, `t_ms` or `seq`. They exist for ordering events
in BigQuery, and their high cardinality would only produce "(other)" rows in
GA4 reports.

## Events

Every event carries `assignment` (`a1`/`a2`/`a3`/`landing`), `tab_name` (the
active tab's label), `seed` (once set), `page_load_id`, `t_ms` (ms since page
load) and `seq` (order within the page load). GA4 adds the timestamp, page
URL, browser, OS, device, country/city, referrer and its own
`ga_session_id` / `user_pseudo_id`.

| Event | When | Extra parameters |
|---|---|---|
| `page_view` | page load (automatic) | — |
| `client_info` | page load | `viewport_w`, `viewport_h`, `pixel_ratio`, `touch`, `dark_mode`, `load_ms` |
| `tab_view` | tab switched | `from_tab`, `from_tab_ms` (wall time on the previous tab), `from_tab_active_ms`, `first_time` |
| `student_id_set` | ID entered and seed assigned | `previous_seed` (if the ID was changed) |
| `model_eval` | each evaluation run | `eval_index`, the hyperparameters (A1: `lr`, `steps`, `val_fraction`; A2: `n_estimators`, `max_depth`, `oversample_ratio`; A3: `max_seq_len`, `hidden_units`, `cell_type`, `bidirectional`), `auc`, `accuracy`, `f1`, `loss` (A1), `is_optimal` |
| `optimal_reached` | first optimal evaluation | same as `model_eval` + `ms_since_seed` |
| `exploration_complete` | completion bar appears | `eval_count`, `concepts_opened`, `active_ms`, `ms_since_seed` |
| `concept_open` / `concept_close` | concept card modal | `concept_index`, `concept_title`, `first_time` / `dwell_ms` |
| `params_copied` | "copy params" button | — |
| `keyword_test` | A1 keyword tester | `text_length`, `match_count` |
| `max_seq_pick`, `cell_pick`, `bidir_toggle`, `timeline_view` | A3 controls | `max_seq_len`, `cell_type`, `bidirectional`, `patient_id` |
| `arena_run` | A2 Architecture Arena training | `source` (`preset:<name>` or `custom`), `a_cfg`, `b_cfg`, `a_val_f1`, `b_val_f1`, `a_stopped_epoch`, `b_stopped_epoch` |
| `arena_toggle` | A2 dropout / early-stopping toggle | `el_id`, `value` |
| `ui_click` | any button, link or clickable element | `el_id`, `el_tag`, `el_text`, `el_action` (its `onclick`), `link_url` |
| `rage_click` | 3 clicks on one element within 0.8 s | same as `ui_click` |
| `ui_change` | slider / select / checkbox committed | `el_id`, `el_type`, `from_value`, `to_value`, `adjust_ms` |
| `text_input` | pause in typing in a search/keyword box (never the ID box) | `el_id`, `text_length` (+ `typed_text` if `CONFIG.sendTypedText`) |
| `scroll_depth` | 25/50/75/100% of a tab pane or table, once each per tab | `scroller`, `percent` |
| `text_copy` | student copies text from the page | `text_length` |
| `page_hidden` / `page_visible` | tab hidden / back | running totals `active_ms`, `eval_count`, `tabs_seen`, `concepts_opened`, `click_count` / `away_ms` |
| `js_error` | uncaught error (max 10 per page) | `message`, `source`, `line` |

*Active* time counts 5-second slices in which the page was visible and the
student moved, clicked, typed or scrolled within the last 30 s.

## Getting the data out (BigQuery)

One row per event, with the parameters as columns:

```sql
CREATE TEMP FUNCTION p(params ANY TYPE, k STRING) AS ((
  SELECT COALESCE(value.string_value, CAST(value.int_value AS STRING),
                  CAST(value.double_value AS STRING), CAST(value.float_value AS STRING))
  FROM UNNEST(params) WHERE key = k));

SELECT
  TIMESTAMP_MICROS(event_timestamp)           AS ts,
  user_pseudo_id,                              -- one browser (GA cookie)
  p(event_params, 'ga_session_id')            AS session_id,
  p(event_params, 'page_load_id')             AS page_load_id,
  CAST(p(event_params, 'seq') AS INT64)       AS seq,
  event_name,
  p(event_params, 'assignment')               AS assignment,
  p(event_params, 'tab_name')                 AS tab_name,
  p(event_params, 'seed')                     AS seed,
  p(event_params, 'el_action')                AS el_action,
  SAFE_CAST(p(event_params, 'auc') AS FLOAT64) AS auc,
  p(event_params, 'is_optimal')               AS is_optimal
FROM `YOUR_PROJECT.analytics_YOUR_PROPERTY_ID.events_*`
WHERE _TABLE_SUFFIX BETWEEN '20261001' AND '20261231'
  AND p(event_params, 'debug_mode') IS NULL   -- drop your ?ga_debug=1 test visits
ORDER BY user_pseudo_id, page_load_id, seq;
```

Events a student triggers before answering the consent banner are sent when
they click Allow. Their timestamps are therefore the click time; use `t_ms`
for their real timing.

## Privacy and research ethics

- **NetIDs are never sent.** The ID box is excluded from every listener. Only
  the derived seed (`SHA-256(netid) mod 900 + 100`) is sent, and several
  students share each seed. Don't add the NetID, a name or an email to any
  event: Google Analytics' terms forbid personally identifiable information.
- Free-text boxes send only their length unless `CONFIG.sendTypedText = true`.
- GA4 does not store IP addresses. The `_ga` cookie links one browser's visits
  across all three assignments (same `user_pseudo_id`).
- **Consent banner.** `CONFIG.consent` controls it:
  - `'opt-in'` (default): nothing loads from Google until the student clicks
    Allow.
  - `'notice'`: tracking is on, with an Opt-out button.
  - `'none'`: no banner.

  Point `CONFIG.infoUrl` at your study information sheet.
- Analysing student interaction data for publication is human-subjects
  research. Check with the UArizona IRB (Human Subjects Protection Program)
  before collecting. Match the banner text and consent mode to what they
  approve, and consider FERPA if the data is ever linked to grades or
  identities.
