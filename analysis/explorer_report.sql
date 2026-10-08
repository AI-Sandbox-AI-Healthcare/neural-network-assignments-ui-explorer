-- Explorer usage report -- run in BigQuery (console.cloud.google.com -> BigQuery).
--
-- 1. The data source is `nn-ui-analytics.analytics_557882182` (in CREATE TEMP
--    TABLE ev, the only place it appears) -- change it only if the GA4 export
--    moves to another project or dataset.
-- 2. Set the date range just below.
-- 3. Paste the whole file into a new BigQuery query tab and Run. Each numbered SELECT returns one table: in the Results panel, click
--    "View results" next to each statement. Save any of them with "Save results".
--
-- "User" = one browser (GA's user_pseudo_id), not a named student. `seeds` is
-- the seed(s) entered in that browser; several students share each seed.
-- Test visits (?ga_debug=1 or localhost) are left out.
-- Event and parameter definitions: ANALYTICS.md.

DECLARE start_date STRING DEFAULT '20261001';   -- YYYYMMDD
DECLARE end_date   STRING DEFAULT '20261231';

CREATE TEMP FUNCTION p(params ANY TYPE, k STRING) AS ((
  SELECT COALESCE(value.string_value, CAST(value.int_value AS STRING),
                  CAST(value.double_value AS STRING), CAST(value.float_value AS STRING))
  FROM UNNEST(params) WHERE key = k LIMIT 1));

-- One row per event sent by docs/analytics.js, parameters as columns.
CREATE TEMP TABLE ev AS
SELECT
  user_pseudo_id,
  p(event_params, 'page_load_id')                       AS visit,
  SAFE_CAST(p(event_params, 'seq') AS INT64)            AS seq,
  SAFE_CAST(p(event_params, 't_ms') AS INT64)           AS t_ms,
  TIMESTAMP_MICROS(event_timestamp)                     AS ts,
  event_name,
  p(event_params, 'assignment')                         AS assignment,
  p(event_params, 'tab_name')                           AS tab_name,
  p(event_params, 'seed')                               AS seed,
  -- controls
  p(event_params, 'el_id')                              AS el_id,
  p(event_params, 'from_value')                         AS from_value,
  p(event_params, 'to_value')                           AS to_value,
  p(event_params, 'value')                              AS toggle_value,
  SAFE_CAST(p(event_params, 'adjust_ms') AS INT64)      AS adjust_ms,
  -- concept cards
  p(event_params, 'concept_title')                      AS concept_title,
  SAFE_CAST(p(event_params, 'dwell_ms') AS INT64)       AS dwell_ms,
  -- time and progress
  SAFE_CAST(p(event_params, 'active_ms') AS INT64)      AS active_ms,
  SAFE_CAST(p(event_params, 'eval_index') AS INT64)     AS eval_index,
  SAFE_CAST(p(event_params, 'ms_since_seed') AS INT64)  AS ms_since_seed,
  -- hyperparameters: A1, A2, A3
  SAFE_CAST(p(event_params, 'lr') AS FLOAT64)               AS lr,
  SAFE_CAST(p(event_params, 'steps') AS INT64)              AS steps,
  SAFE_CAST(p(event_params, 'val_fraction') AS FLOAT64)     AS val_fraction,
  SAFE_CAST(p(event_params, 'n_estimators') AS INT64)       AS n_estimators,
  SAFE_CAST(p(event_params, 'max_depth') AS INT64)          AS max_depth,
  SAFE_CAST(p(event_params, 'oversample_ratio') AS FLOAT64) AS oversample_ratio,
  SAFE_CAST(p(event_params, 'max_seq_len') AS INT64)        AS max_seq_len,
  SAFE_CAST(p(event_params, 'hidden_units') AS INT64)       AS hidden_units,
  p(event_params, 'cell_type')                              AS cell_type,
  p(event_params, 'bidirectional')                          AS bidirectional,
  -- scores
  SAFE_CAST(p(event_params, 'auc') AS FLOAT64)          AS auc,
  SAFE_CAST(p(event_params, 'f1') AS FLOAT64)           AS f1,
  SAFE_CAST(p(event_params, 'accuracy') AS FLOAT64)     AS accuracy,
  SAFE_CAST(p(event_params, 'loss') AS FLOAT64)         AS loss,
  p(event_params, 'is_optimal')                         AS is_optimal
FROM `nn-ui-analytics.analytics_557882182.events_*`
WHERE _TABLE_SUFFIX BETWEEN start_date AND end_date
  AND p(event_params, 'page_load_id') IS NOT NULL       -- our events, not GA's automatic ones
  AND p(event_params, 'debug_mode') IS NULL
  AND NOT REGEXP_CONTAINS(IFNULL(p(event_params, 'page_location'), ''),
                          r'ga_debug|//localhost|//127\.0\.0\.1');

-- Time from each event to the next one on the same page = time spent on that
-- event's tab. A gap that starts at page_hidden is time away (not counted);
-- other gaps are capped at 30 min so a tab left open overnight doesn't count.
CREATE TEMP TABLE gaps AS
SELECT *,
  IF(event_name = 'page_hidden', 0,
     LEAST(LEAD(t_ms) OVER (PARTITION BY visit ORDER BY seq) - t_ms, 30 * 60 * 1000)) AS gap_ms
FROM ev;

-- 1. Time per user, per assignment.
--    visible_minutes: page open and on screen (reading counts, idle >30 min doesn't).
--    active_minutes:  moving / clicking / typing / scrolling within the last 30 s.
WITH per_visit AS (
  SELECT user_pseudo_id, assignment, visit,
    MAX(seed) AS seed,
    MIN(ts) AS started,
    SUM(gap_ms) AS visible_ms,
    MAX(IF(event_name = 'page_hidden', active_ms, NULL)) AS active_ms
  FROM gaps
  GROUP BY user_pseudo_id, assignment, visit
)
SELECT
  user_pseudo_id,
  assignment,
  STRING_AGG(DISTINCT seed) AS seeds,
  COUNT(*) AS visits,
  MIN(started) AS first_visit,
  ROUND(SUM(visible_ms) / 60000, 1) AS visible_minutes,
  ROUND(SUM(active_ms) / 60000, 1) AS active_minutes
FROM per_visit
GROUP BY user_pseudo_id, assignment
ORDER BY assignment, visible_minutes DESC;

-- 2. Time per tab, per user.
SELECT
  user_pseudo_id,
  assignment,
  tab_name,
  ROUND(SUM(gap_ms) / 60000, 1) AS visible_minutes,
  COUNTIF(event_name = 'tab_view') AS times_switched_to
FROM gaps
WHERE assignment != 'landing'
GROUP BY user_pseudo_id, assignment, tab_name
ORDER BY user_pseudo_id, assignment, visible_minutes DESC;

-- 3. Every parameter change, in order.
--    control ids -- A1: lr-sl, steps-sl, vf-sl (learning rate, steps, validation fraction)
--                   A2: rb-sl (rebalance, Dataset tab), ne-sl, md-sl, or-sl (n_estimators,
--                       max_depth, oversample ratio); arena a-/b-preset, a-/b-act,
--                       a-/b-drop, a-/b-early
--                   A3: hu-sl (hidden units), max_seq_pick, cell_pick, bidir_toggle
SELECT
  user_pseudo_id,
  assignment,
  ts,
  tab_name,
  COALESCE(el_id, event_name) AS control,
  from_value,
  COALESCE(to_value, CAST(max_seq_len AS STRING), cell_type, bidirectional, toggle_value) AS to_value,
  adjust_ms
FROM ev
WHERE event_name IN ('ui_change', 'max_seq_pick', 'cell_pick', 'bidir_toggle', 'arena_toggle')
ORDER BY user_pseudo_id, ts, seq;

-- 4a. Every model trained (one row per evaluation), with its settings and scores.
SELECT
  user_pseudo_id,
  assignment,
  ts,
  eval_index,
  lr, steps, val_fraction,                          -- A1
  n_estimators, max_depth, oversample_ratio,        -- A2
  max_seq_len, hidden_units, cell_type, bidirectional,  -- A3
  auc, f1, accuracy, loss, is_optimal
FROM ev
WHERE event_name = 'model_eval'
ORDER BY user_pseudo_id, ts, seq;

-- 4b. Optimal performance and activity complete, per user.
--     reached_optimal:   a model met the student's personal oracle target.
--     activity_complete: the green "complete" bar appeared -- optimal reached
--                        AND every concept card opened (A2 also needs the
--                        Architecture Arena goals; A3 a max_seq_len choice).
--     minutes_to_*:      from entering the student ID, in that same visit.
SELECT
  user_pseudo_id,
  assignment,
  STRING_AGG(DISTINCT seed) AS seeds,
  COUNTIF(event_name = 'model_eval') AS models_trained,
  LOGICAL_OR(event_name = 'optimal_reached') AS reached_optimal,
  MIN(IF(event_name = 'optimal_reached', ts, NULL)) AS first_optimal_at,
  MIN(IF(event_name = 'optimal_reached', eval_index, NULL)) AS models_until_optimal,
  ROUND(MIN(IF(event_name = 'optimal_reached', ms_since_seed, NULL)) / 60000, 1) AS minutes_to_optimal,
  LOGICAL_OR(event_name = 'exploration_complete') AS activity_complete,
  MIN(IF(event_name = 'exploration_complete', ts, NULL)) AS completed_at,
  ROUND(MIN(IF(event_name = 'exploration_complete', ms_since_seed, NULL)) / 60000, 1) AS minutes_to_complete
FROM ev
WHERE assignment != 'landing'
GROUP BY user_pseudo_id, assignment
ORDER BY assignment, user_pseudo_id;

-- 5. Concept cards: which ones each user opened, how often, and for how long.
--    Cards a user never opened don't appear. A card still open when the
--    browser closed counts as opened, with no time.
SELECT
  user_pseudo_id,
  assignment,
  concept_title,
  COUNTIF(event_name = 'concept_open') AS times_opened,
  ROUND(SUM(IF(event_name = 'concept_close', dwell_ms, 0)) / 1000) AS seconds_open
FROM ev
WHERE event_name IN ('concept_open', 'concept_close')
GROUP BY user_pseudo_id, assignment, concept_title
ORDER BY user_pseudo_id, assignment, seconds_open DESC;

-- 6. Class summary per assignment: how many users reached optimal performance
--    and completed the activity. Percentages are of users who entered an ID
--    (nobody can train a model before that).
WITH per_user AS (
  SELECT user_pseudo_id, assignment,
    LOGICAL_OR(event_name = 'student_id_set') AS entered_id,
    LOGICAL_OR(event_name = 'optimal_reached') AS reached_optimal,
    LOGICAL_OR(event_name = 'exploration_complete') AS completed,
    MIN(IF(event_name = 'optimal_reached', ms_since_seed, NULL)) / 60000 AS minutes_to_optimal,
    MIN(IF(event_name = 'exploration_complete', ms_since_seed, NULL)) / 60000 AS minutes_to_complete
  FROM ev
  WHERE assignment != 'landing'
  GROUP BY user_pseudo_id, assignment
)
SELECT
  assignment,
  COUNT(*) AS users,
  COUNTIF(entered_id) AS entered_id,
  COUNTIF(reached_optimal) AS reached_optimal,
  ROUND(100 * SAFE_DIVIDE(COUNTIF(reached_optimal), COUNTIF(entered_id)), 1) AS pct_reached_optimal,
  COUNTIF(completed) AS activity_complete,
  ROUND(100 * SAFE_DIVIDE(COUNTIF(completed), COUNTIF(entered_id)), 1) AS pct_activity_complete,
  ROUND(APPROX_QUANTILES(minutes_to_optimal, 2)[OFFSET(1)], 1) AS median_minutes_to_optimal,
  ROUND(APPROX_QUANTILES(minutes_to_complete, 2)[OFFSET(1)], 1) AS median_minutes_to_complete
FROM per_user
GROUP BY assignment
ORDER BY assignment;
