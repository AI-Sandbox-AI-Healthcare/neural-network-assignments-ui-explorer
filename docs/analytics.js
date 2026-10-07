/*
 * Google Analytics 4 interaction tracking for the GitHub Pages explorers.
 *
 * Hand-maintained -- NOT generated. Each assignment's generate_docs_site.py
 * adds <script src="../analytics.js" defer></script> to its docs page, and
 * docs/index.html loads it directly. The local Flask explorers
 * (run_sandbox.py) never load it, so nothing is tracked when students run
 * them locally.
 *
 * The explorer code itself is untouched: `defer` runs this file after the
 * page's own scripts, so it can wrap their global functions (switchTab,
 * setStudentId, logRow, openConcept, ...) to emit semantic events, and add
 * document-level listeners for generic clicks, control changes, typing,
 * scrolling, copying, visibility and errors. Every hook first checks that its
 * function exists -- renaming one in run_sandbox.py only silences that event.
 * The full event list lives in ANALYTICS.md at the repo root.
 *
 * Privacy: the NetID typed into the sidebar is never sent -- only the seed
 * derived from it (100-999, shared by several students). Free-text boxes send
 * their length only, unless CONFIG.sendTypedText is turned on.
 *
 * Debugging: add ?ga_debug=1 to a page URL to log every event to the console
 * and show it in GA4's DebugView (works on localhost too). ?notrack=1 opts
 * this browser out for good (use it on your own machines); ?notrack=0 undoes it.
 */
(() => {
'use strict';

const CONFIG = {
  // GA4 -> Admin -> Data streams -> (your web stream) -> Measurement ID
  measurementId: 'G-T41SB4WQ70',
  // Only these hostnames send data (any host does with ?ga_debug=1).
  hosts: ['ai-sandbox-ai-healthcare.github.io'],
  // 'opt-in' : nothing is sent until the student clicks Allow.
  // 'notice' : tracking is on by default; the banner offers an opt-out.
  // 'none'   : no banner -- only if your IRB / course policy allows it.
  consent: 'none',
  noticeText: 'This explorer can record anonymous usage data (tabs, clicks, slider '
    + 'settings and model results) with Google Analytics cookies, to improve the '
    + 'course and for education research. Your NetID is never sent.',
  infoUrl: '',            // optional "Learn more" link, e.g. an IRB information sheet
  sendTypedText: false,   // true -> also send what is typed into search / keyword boxes
};

const ID = CONFIG.measurementId;
const qs = new URLSearchParams(location.search);
const DEBUG = qs.has('ga_debug');
const KEY = 'nnui_ga_consent';
const m = location.pathname.match(/assignment-(\d+)/);
const ASSIGNMENT = m ? 'a' + m[1] : 'landing';
const now = () => Math.round(performance.now());
const warn = e => { if (DEBUG) console.warn('[ga] hook failed', e); };

const store = {
  get() { try { return localStorage.getItem(KEY); } catch (e) { return null; } },
  set(v) { try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch (e) {} },
};
if (qs.has('notrack')) store.set(qs.get('notrack') === '0' ? null : 'denied');

const live = /^G-[A-Z0-9]+$/.test(ID) && ID !== 'G-XXXXXXXXXX'
  && (DEBUG || CONFIG.hosts.includes(location.hostname));
if (!live && !DEBUG) return;
if (!live) console.info('[ga] measurementId not set -- events are logged here, not sent');

// 'granted' | 'denied' | null (undecided: events wait in `buffer`)
const stored = store.get();
let consent = stored === 'granted' || stored === 'denied' ? stored
  : CONFIG.consent === 'opt-in' ? null : 'granted';
const buffer = [];
const userProps = {};

// ============================================================
// gtag + consent
// ============================================================
let started = false;
function startGtag() {
  if (started || !live) return;
  started = true;
  window.dataLayer = window.dataLayer || [];
  window.gtag = function () { dataLayer.push(arguments); };
  gtag('js', new Date());
  gtag('config', ID, Object.assign({
    assignment: ASSIGNMENT,
    allow_google_signals: false,
    allow_ad_personalization_signals: false,
  }, DEBUG ? { debug_mode: true } : {}));
  const s = document.createElement('script');
  s.async = true;
  s.src = 'https://www.googletagmanager.com/gtag/js?id=' + ID;
  document.head.appendChild(s);
}

function decide(choice) {
  store.set(choice);
  consent = choice;
  if (choice === 'granted') {
    startGtag();
    if (started) {
      if (Object.keys(userProps).length) gtag('set', 'user_properties', userProps);
      buffer.forEach(([name, ev]) => gtag('event', name, ev));
    }
  } else if (started) {
    window['ga-disable-' + ID] = true;
  }
  buffer.length = 0;
}

function banner() {
  const optIn = CONFIG.consent === 'opt-in';
  const box = document.createElement('div');
  box.id = 'ga-consent';
  box.setAttribute('role', 'region');
  box.setAttribute('aria-label', 'Usage data notice');
  box.style.cssText = 'position:fixed;right:16px;bottom:16px;z-index:2147483000;'
    + 'max-width:min(400px,calc(100vw - 32px));background:#1e293b;color:#f1f5f9;'
    + 'font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
    + 'padding:14px 16px;border-radius:10px;box-shadow:0 10px 30px rgba(0,0,0,.3)';
  const btn = 'border:0;border-radius:6px;padding:6px 14px;font:inherit;font-weight:600;cursor:pointer;';
  box.innerHTML = '<div>' + CONFIG.noticeText
    + (CONFIG.infoUrl ? ' <a href="' + CONFIG.infoUrl + '" target="_blank" rel="noopener" style="color:#93c5fd">Learn more</a>' : '')
    + '</div><div style="display:flex;gap:8px;justify-content:flex-end;margin-top:10px">'
    + '<button type="button" data-choice="denied" style="' + btn
    + 'background:transparent;color:#cbd5e1;box-shadow:inset 0 0 0 1px #475569">'
    + (optIn ? 'No thanks' : 'Opt out') + '</button>'
    + '<button type="button" data-choice="granted" style="' + btn + 'background:#6366f1;color:#fff">'
    + (optIn ? 'Allow' : 'OK') + '</button></div>';
  box.addEventListener('click', e => {
    const choice = e.target.getAttribute && e.target.getAttribute('data-choice');
    if (!choice) return;
    decide(choice);
    box.remove();
  });
  document.body.appendChild(box);
}

// ============================================================
// send()
// ============================================================
function appState() {
  try { return typeof state !== 'undefined' && state ? state : {}; } catch (e) { return {}; }
}
function currentTab() {
  const b = document.querySelector('.tab-btn.active');
  return b ? b.textContent.replace(/\s+/g, ' ').trim() : ASSIGNMENT;
}

// Random per page load: groups one visit's events (seq orders them) in BigQuery.
const PAGE_LOAD_ID = Math.random().toString(36).slice(2, 10);
let seq = 0;
function send(name, params) {
  const ev = Object.assign({
    assignment: ASSIGNMENT,
    tab_name: currentTab(),
    seed: appState().seed,
    page_load_id: PAGE_LOAD_ID,
    t_ms: now(),          // ms since page load -- survives buffering
    seq: ++seq,
  }, params);
  // GA4 limits: 25 params per event, 100-character string values.
  for (const k of Object.keys(ev)) {
    const v = ev[k];
    if (v === undefined || v === null || v === '' || (typeof v === 'number' && !isFinite(v))) delete ev[k];
    else if (typeof v === 'number') ev[k] = Number.isInteger(v) ? v : +v.toFixed(4);
    else ev[k] = String(v).slice(0, 100);
  }
  if (DEBUG) { ev.debug_mode = true; console.debug('[ga]', name, ev); }
  if (!live || consent === 'denied') return;
  if (consent === null) { if (buffer.length < 200) buffer.push([name, ev]); return; }
  gtag('event', name, ev);
}

function setUserProps(p) {
  Object.assign(userProps, p);
  if (started && consent === 'granted') gtag('set', 'user_properties', userProps);
}

// Wrap a global function of the explorer page. `before(args)` runs first and
// its result is handed to `after(args, returnValue, ctx)`, which runs once the
// original has finished (awaited if it is async).
function hook(name, after, before) {
  const orig = window[name];
  if (typeof orig !== 'function') return;
  window[name] = function (...args) {
    let ctx;
    try { ctx = before && before(args); } catch (e) { warn(e); }
    const r = orig.apply(this, args);
    if (after) Promise.resolve(r).then(v => { try { after(args, v, ctx); } catch (e) { warn(e); } }, () => {});
    return r;
  };
}

// ============================================================
// Active time (visible + input within the last 30 s), overall and per tab
// ============================================================
let lastInput = now(), activeMs = 0;
const tabActive = {};
['pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart', 'scroll'].forEach(t =>
  document.addEventListener(t, () => { lastInput = now(); }, { capture: true, passive: true }));
setInterval(() => {
  if (document.hidden || now() - lastInput > 30000) return;
  activeMs += 5000;
  const t = currentTab();
  tabActive[t] = (tabActive[t] || 0) + 5000;
}, 5000);

// ============================================================
// Semantic events -- explorer functions
// ============================================================
let tab = currentTab(), tabSince = now(), tabActiveAtEnter = 0;
const tabsSeen = new Set([tab]);
hook('switchTab', () => {
  const to = currentTab();
  if (to === tab) return;
  send('tab_view', {
    from_tab: tab,
    from_tab_ms: now() - tabSince,
    from_tab_active_ms: (tabActive[tab] || 0) - tabActiveAtEnter,
    first_time: !tabsSeen.has(to),
  });
  tab = to; tabSince = now(); tabActiveAtEnter = tabActive[to] || 0;
  tabsSeen.add(to);
});

let seedAt = null;
hook('setStudentId', (args, r, prevSeed) => {
  const seed = appState().seed;
  if (!seed || seed === prevSeed) return;
  seedAt = now();
  setUserProps({ seed: String(seed) });
  send('student_id_set', { previous_seed: prevSeed });
}, () => appState().seed);

// One event per model evaluation. A1 logs via logInteraction(params, metrics),
// A2 via logRow(ne, md, or, d), A3 via logRow(body, d).
let evals = 0, optimalAt = null;
function evalParams(name, a) {
  if (name === 'logInteraction') {
    const [p, mt] = a;
    return { lr: p.lr, steps: p.steps, val_fraction: p.val_fraction, auc: mt.auc,
      accuracy: mt.accuracy, f1: mt.f1, loss: mt.final_loss, is_optimal: mt.is_optimal };
  }
  if (typeof a[0] === 'number') {
    const [ne, md, or, d] = a;
    return { n_estimators: ne, max_depth: md, oversample_ratio: or, auc: d.auc,
      accuracy: d.accuracy, f1: d.f1, is_optimal: d.is_optimal };
  }
  const [b, d] = a, r = d.recurrent || d;
  return { max_seq_len: b.max_seq_len, hidden_units: b.hidden_units, cell_type: b.cell_type,
    bidirectional: b.bidirectional, auc: r.auc, accuracy: r.accuracy, f1: r.f1, is_optimal: d.is_optimal };
}
['logInteraction', 'logRow'].forEach(name => hook(name, args => {
  const p = evalParams(name, args);
  evals++;
  send('model_eval', Object.assign({ eval_index: evals }, p));
  if (p.is_optimal && optimalAt === null) {
    optimalAt = now();
    send('optimal_reached', Object.assign({
      eval_index: evals, ms_since_seed: seedAt === null ? undefined : optimalAt - seedAt }, p));
  }
}));

let concept = null;
const conceptsOpened = new Set();
const conceptTitle = i => {
  try { return CONCEPTS[i].title.replace(/<[^>]*>/g, ''); } catch (e) { return String(i); }
};
hook('openConcept', ([i]) => {
  concept = { i, title: conceptTitle(i), at: now() };
  send('concept_open', { concept_index: i, concept_title: concept.title, first_time: !conceptsOpened.has(i) });
  conceptsOpened.add(i);
});
hook('closeModal', () => {
  const overlay = document.getElementById('modal-overlay');
  if (!concept || (overlay && overlay.classList.contains('open'))) return;   // close was blocked
  send('concept_close', { concept_index: concept.i, concept_title: concept.title, dwell_ms: now() - concept.at });
  concept = null;
});

hook('copyParams', () => send('params_copied'));
hook('copyOptimalParams', () => send('params_copied'));
hook('dismissCompleteFooter', () => send('complete_footer_dismissed'));
hook('testKeywords', () => send('keyword_test', {
  text_length: (document.getElementById('kw-input') || {}).value?.length,
  match_count: document.querySelectorAll('#kw-result .kw-chip').length,
}));
// A3
hook('pickMaxSeq', ([n]) => send('max_seq_pick', { max_seq_len: n }));
hook('pickCell', ([v]) => send('cell_pick', { cell_type: v }));
hook('toggleBidir', () => send('bidir_toggle', { bidirectional: appState().bidir }));
hook('showTimeline', ([pid]) => send('timeline_view', { patient_id: pid }));
// A2 Architecture Arena
let arenaSource = 'custom';
hook('runPreset', null, ([name]) => { arenaSource = 'preset:' + name; });
hook('trainCustom', null, () => { arenaSource = 'custom'; });
hook('tgl', ([btn]) => send('arena_toggle', { el_id: btn.id, value: btn.textContent.trim() }));
hook('trainBoth', (args, d) => {
  if (!d || !d.a || !d.b) return;
  const desc = side => {
    try {
      const c = window.cfg(side);
      return [c.preset, c.activation, c.dropout ? 'dropout' : 'no-dropout',
        c.early_stopping ? 'early-stop' : 'no-early-stop'].join('/');
    } catch (e) { return undefined; }
  };
  send('arena_run', {
    source: arenaSource, a_cfg: desc('a'), b_cfg: desc('b'),
    a_val_f1: d.a.val_f1, b_val_f1: d.b.val_f1,
    a_stopped_epoch: d.a.stopped_epoch, b_stopped_epoch: d.b.stopped_epoch,
  });
});

// Every explorer shows #top-complete-bar once all its tasks are done.
const bar = document.getElementById('top-complete-bar');
if (bar && window.MutationObserver) {
  let done = false;
  new MutationObserver(() => {
    if (done || bar.style.display !== 'flex') return;
    done = true;
    send('exploration_complete', {
      eval_count: evals, concepts_opened: conceptsOpened.size, active_ms: activeMs,
      ms_since_seed: seedAt === null ? undefined : now() - seedAt,
    });
  }).observe(bar, { attributes: true, attributeFilter: ['style'] });
}

// ============================================================
// Generic events -- any element, including ones added later
// ============================================================
const CLICKABLE = 'button, a[href], [onclick], [role="button"], summary, input[type="checkbox"], input[type="radio"]';
const FIELD = 'input[type="range"], input[type="number"], input[type="checkbox"], input[type="radio"], select';
const TEXT = 'input[type="text"], input[type="search"], input:not([type]), textarea';
const NEVER_SEND = new Set(['sid-input']);   // NetID box

let clicks = 0, lastClick = { el: null, t: 0, n: 0 };
document.addEventListener('click', e => {
  const el = e.target.closest && e.target.closest(CLICKABLE);
  if (!el || el.closest('#ga-consent')) return;
  if ((el.getAttribute('onclick') || '').trim() === 'event.stopPropagation()') return;   // modal body
  const t = now();
  lastClick = el === lastClick.el && t - lastClick.t < 800
    ? { el, t, n: lastClick.n + 1 } : { el, t, n: 1 };
  const d = {
    el_id: el.id,
    el_tag: el.tagName.toLowerCase(),
    el_text: (el.getAttribute('aria-label') || el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 80),
    el_action: (el.getAttribute('onclick') || '').trim(),
    link_url: el.href,
  };
  clicks++;
  send('ui_click', d);
  if (lastClick.n === 3) send('rage_click', d);
}, true);

// Sliders / selects / checkboxes: one event per committed change, with the
// value before the drag or keypress and how long the adjustment took.
const fieldValue = el => (el.type === 'checkbox' || el.type === 'radio' ? String(el.checked) : el.value);
const armed = new WeakMap();
const arm = e => {
  const el = e.target;
  if (el.matches && el.matches(FIELD)) armed.set(el, { v: fieldValue(el), t: now() });
};
document.addEventListener('pointerdown', arm, true);
document.addEventListener('keydown', arm, true);
document.addEventListener('change', e => {
  const el = e.target;
  if (!el.matches || !el.matches(FIELD)) return;
  const a = armed.get(el);
  armed.delete(el);
  send('ui_change', {
    el_id: el.id || el.name, el_type: el.type, from_value: a && a.v,
    to_value: fieldValue(el), adjust_ms: a ? now() - a.t : undefined,
  });
}, true);

// Search / keyword boxes: one event per pause in typing.
const typing = new WeakMap();
document.addEventListener('input', e => {
  const el = e.target;
  if (!el.matches || !el.matches(TEXT) || NEVER_SEND.has(el.id)) return;
  const tabAtInput = currentTab();
  clearTimeout(typing.get(el));
  typing.set(el, setTimeout(() => send('text_input', Object.assign(
    { tab_name: tabAtInput, el_id: el.id, text_length: el.value.length },
    CONFIG.sendTypedText ? { typed_text: el.value.trim() } : {})), 1500));
}, true);

// Scroll depth per tab and per scrolling box (tab pane, patient table, ...).
// Each .tab-pane scrolls on its own, so GA4's built-in scroll event (window
// at 90%) never fires on the explorers.
const depthSent = {};
const scrollTimers = new Map();
document.addEventListener('scroll', e => {
  const el = e.target === document ? document.scrollingElement : e.target;
  if (!el || !el.classList || scrollTimers.has(el)) return;
  scrollTimers.set(el, setTimeout(() => {
    scrollTimers.delete(el);
    const room = el.scrollHeight - el.clientHeight;
    if (room < 50) return;
    const pct = el.scrollTop / room * 100;
    const scroller = el === document.scrollingElement ? 'page' : el.id || el.classList[0] || el.tagName.toLowerCase();
    const key = currentTab() + '|' + scroller;
    const sent = depthSent[key] || (depthSent[key] = new Set());
    [25, 50, 75, 100].forEach(p => {
      if (pct >= p - 1 && !sent.has(p)) { sent.add(p); send('scroll_depth', { scroller, percent: p }); }
    });
  }, 250));
}, { capture: true, passive: true });

document.addEventListener('copy', () => send('text_copy', { text_length: String(getSelection()).length }));

// page_hidden doubles as the running summary of the visit: the last one sent
// is the final tally when the student closes the tab.
let hiddenAt = null;
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    hiddenAt = now();
    send('page_hidden', {
      active_ms: activeMs, eval_count: evals, tabs_seen: tabsSeen.size,
      concepts_opened: conceptsOpened.size, click_count: clicks,
    });
  } else if (hiddenAt !== null) {
    send('page_visible', { away_ms: now() - hiddenAt });
  }
});

let errors = 0;
window.addEventListener('error', e => {
  if (errors++ < 10) send('js_error', {
    message: e.message, source: (e.filename || '').split('?')[0].split('/').pop() || 'index.html', line: e.lineno });
});
window.addEventListener('unhandledrejection', e => {
  if (errors++ < 10) send('js_error', { message: (e.reason && e.reason.message) || e.reason, source: 'promise' });
});

function clientInfo() {
  const nav = performance.getEntriesByType ? performance.getEntriesByType('navigation')[0] : null;
  send('client_info', {
    viewport_w: innerWidth, viewport_h: innerHeight, pixel_ratio: devicePixelRatio,
    touch: navigator.maxTouchPoints > 0,
    dark_mode: matchMedia('(prefers-color-scheme: dark)').matches,
    load_ms: nav && nav.loadEventEnd ? Math.round(nav.loadEventEnd) : undefined,
  });
}

// ============================================================
// Start
// ============================================================
if (consent === 'granted') startGtag();
if (CONFIG.consent !== 'none' && !stored) banner();
if (document.readyState === 'complete') setTimeout(clientInfo, 0);
else window.addEventListener('load', () => setTimeout(clientInfo, 0));
})();
