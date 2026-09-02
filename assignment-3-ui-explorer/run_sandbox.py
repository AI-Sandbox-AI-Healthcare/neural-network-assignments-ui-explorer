#!/usr/bin/env python3
"""
Assignment 3 -- AI-Sandbox Interactive Explorer
Run from anywhere:  python assignment-3-ui-explorer/run_sandbox.py
Opens http://localhost:3003 in your browser.
"""
import json
import logging
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_LOG_FILE = ROOT / "server.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
logging.getLogger("werkzeug").handlers = logging.getLogger().handlers
logging.getLogger("werkzeug").propagate = False

FEATURES_FILE = ROOT / "data" / "patient_features.csv"
VISITS_FILE = ROOT / "data" / "patient_visits.csv"
ORACLE_FILE = ROOT / "oracle_table.json"

try:
    import pandas as pd
    _fdf = pd.read_csv(FEATURES_FILE)
    _vdf = pd.read_csv(VISITS_FILE)
    _vdf["patient_id"] = _vdf["patient_id"].astype(str)
    log.info("Loaded %d patients / %d visit rows", len(_fdf), len(_vdf))
except Exception as exc:  # pragma: no cover
    log.error("Failed to load data: %s", exc, exc_info=True)
    sys.exit(1)

import reference as ref  # noqa: E402

for _c in ref.VISIT_FEATURE_COLS:
    _vdf[_c] = _vdf[_c].astype(float)

if ORACLE_FILE.exists():
    ORACLE_TABLE = json.loads(ORACLE_FILE.read_text(encoding="utf-8"))
    log.info("Oracle table: %d seeds loaded", len(ORACLE_TABLE))
else:
    log.warning("oracle_table.json not found -- seeds computed on demand. "
                "Run: python assignment-3-ui-explorer/generate_oracle.py")
    ORACLE_TABLE = {}

app = Flask(__name__)


def _oracle_for(seed):
    entry = ORACLE_TABLE.get(str(seed))
    if entry is None:
        entry = ref.oracle_metrics(_fdf, _vdf, seed)
        ORACLE_TABLE[str(seed)] = entry
        log.info("Computed oracle for seed %d on demand (AUC %.3f)", seed, entry["auc"])
    return entry


def _good_seed(base_seed):
    if ORACLE_TABLE:
        return ref.good_seed(base_seed, ORACLE_TABLE)
    return base_seed


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/data")
def api_data():
    labels = ref._labels(_fdf["condition_text"].tolist())
    n_pos = int(labels.sum())
    counts = _vdf.groupby("patient_id", sort=False).size()
    counts = counts.reindex(_fdf["id"].astype(str)).fillna(0).astype(int)
    hist = [int((counts == k).sum()) for k in range(1, ref.MAX_VISITS + 1)]
    visits = [
        [str(r.patient_id), int(r.visit_number), int(r.days_since_first_visit),
         float(r.pain_score_at_visit), int(r.medications_at_visit), int(r.visit_type_code)]
        for r in _vdf.itertuples(index=False)
    ]
    return jsonify({
        "patients": _fdf.to_dict(orient="records"),
        "columns": list(_fdf.columns),
        "labels": labels.tolist(),
        "visits": visits,
        "seq_len_hist": hist,
        "n_total": len(labels),
        "n_positive": n_pos,
        "n_negative": len(labels) - n_pos,
        "positive_rate": round(n_pos / len(labels), 3),
        "max_visits": ref.MAX_VISITS,
        "visit_feature_cols": ref.VISIT_FEATURE_COLS,
    })


@app.route("/api/assign", methods=["POST"])
def api_assign():
    data = request.get_json()
    student_id = (data.get("student_id") or "").strip()
    if not student_id:
        return jsonify({"error": "student_id required"}), 400
    seed = _good_seed(ref.student_to_seed(student_id))
    try:
        oracle = _oracle_for(seed)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 500
    return jsonify({"seed": seed, "oracle": oracle})


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    data = request.get_json()
    try:
        seed = int(data["seed"])
        max_seq_len = int(data["max_seq_len"])
        hidden_units = int(data["hidden_units"])
        cell_type = str(data["cell_type"])
        bidirectional = bool(data.get("bidirectional", False))
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        out = ref.evaluate_seed(_fdf, _vdf, seed, max_seq_len, hidden_units,
                                cell_type, bidirectional)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 500
    return jsonify(out)


# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI-Sandbox -- Assignment 3</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html{font-size:18px}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f1f5f9;color:#1e293b;display:flex;min-height:100vh;font-size:14px}
#sidebar{width:264px;min-width:264px;background:#e8dfd0;color:#2e1f10;display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto;border-right:1px solid #cfc0a4}
#sidebar h1{font-size:.95rem;font-weight:800;padding:16px 16px 2px;color:#1e1208;letter-spacing:.03em}
#sidebar .subtitle{font-size:.7rem;color:#7a6450;padding:0 16px 14px;border-bottom:1px solid #cfc0a4;font-weight:500}
.sid-s{padding:12px 14px;border-bottom:1px solid #cfc0a4}
.sid-lbl{font-size:.68rem;text-transform:uppercase;letter-spacing:.09em;color:#6b5440;margin-bottom:5px;font-weight:700}
.sid-s input{width:100%;padding:7px 9px;border-radius:6px;border:1px solid #c4ae8c;background:#e0d6c4;color:#2e1f10;font-size:.82rem}
.sid-s input:focus{outline:none;border-color:#4f46e5}
.sid-s button{width:100%;margin-top:7px;padding:7px;background:#4f46e5;color:white;border:none;border-radius:6px;cursor:pointer;font-size:.8rem;font-weight:700}
.sid-s button:hover{background:#4338ca}
#seed-disp{font-size:.74rem;color:#4f46e5;margin-top:6px;display:none;font-weight:600}
.oracle-s{padding:12px 14px;border-bottom:1px solid #cfc0a4;display:none}
.oracle-row{display:flex;justify-content:space-between;margin-bottom:6px}
.oracle-row .nm{font-size:.8rem;color:#5c4430;font-weight:600}
.oracle-row .vl{font-size:.88rem;font-weight:800;color:#4f46e5}
.oracle-row .vl.hit{color:#10b981}
.fc-prog-s{padding:12px 14px;border-bottom:1px solid #cfc0a4}
.fc-prog-bar-wrap{background:#cfc0a4;border-radius:99px;height:6px;overflow:hidden;margin:6px 0 4px}
.fc-prog-bar-fill{height:100%;background:#4f46e5;border-radius:99px;transition:width .4s}
.fc-prog-txt{font-size:.72rem;color:#6b5440;font-weight:500}
#top-complete-bar{position:fixed;top:0;left:0;right:0;height:46px;z-index:500;display:none;align-items:center;justify-content:center;gap:10px;
  background:linear-gradient(90deg,#053b2c,#065f46 45%,#0d9668 85%,#10b981);
  color:#ecfdf5;font-weight:700;font-size:.86rem;box-shadow:0 6px 20px rgba(2,6,23,.3);border-bottom:1px solid rgba(236,253,245,.28)}
#top-complete-bar .tcb-check{flex:none;width:22px;height:22px;border-radius:999px;background:#ecfdf5;color:#065f46;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:900}
#top-complete-bar b{color:#fff}
body.has-completion-bar{padding-top:46px}
body.has-completion-bar #sidebar{height:calc(100vh - 46px)}
.pipeline{display:flex;align-items:center;flex-wrap:wrap;gap:6px;padding:14px 18px;background:white;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.pipe-step{flex:1 1 0;display:flex;flex-direction:column;align-items:center;gap:3px;min-width:88px}
.pipe-step .p-icon{font-size:1.9rem}
.pipe-step .p-name{font-size:.68rem;font-weight:700;color:#334155;text-align:center}
.pipe-step .p-sub{font-size:.58rem;color:#94a3b8;text-align:center}
.pipe-arrow{font-size:1.1rem;color:#94a3b8;padding:0 2px}
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.tab-nav{background:white;border-bottom:2px solid #e2e8f0;display:flex;padding:0 20px;position:sticky;top:0;z-index:20;gap:4px;flex-wrap:wrap}
.tab-btn{padding:12px 16px;font-size:.82rem;font-weight:600;color:#64748b;cursor:pointer;border:none;background:none;border-bottom:3px solid transparent;margin-bottom:-2px;white-space:nowrap}
.tab-btn.active{color:#4f46e5;border-bottom-color:#4f46e5}
.tab-btn:hover:not(.active){color:#334155}
.tab-pane{display:none;padding:22px;overflow-y:auto;flex:1}
.tab-pane.active{display:block}
.card{background:white;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:18px;margin-bottom:16px}
.card-title{font-size:.9rem;font-weight:700;color:#1e293b;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.badge{font-size:.67rem;background:#ede9fe;color:#4f46e5;padding:2px 7px;border-radius:99px;font-weight:700}
.info-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px}
@media(max-width:900px){.info-grid{grid-template-columns:1fr}}
.info-card{background:white;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.08);border-top:3px solid #4f46e5}
.info-card.g{border-top-color:#10b981}.info-card.a{border-top-color:#f59e0b}
.info-card h3{font-size:.82rem;font-weight:700;color:#1e293b;margin-bottom:8px}
.info-card p{font-size:.78rem;color:#475569;line-height:1.55}
.concepts-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px}
.prog-info{font-size:.8rem;color:#64748b}
.complete-banner{background:#ecfdf5;border:1.5px solid #10b981;border-radius:8px;padding:10px 14px;font-size:.82rem;color:#065f46;display:none;margin-bottom:12px;font-weight:600}
.concept-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px}
.concept-card{background:white;border-radius:10px;padding:14px;box-shadow:0 1px 4px rgba(0,0,0,.08);cursor:pointer;transition:transform .15s,box-shadow .15s;border:2px solid transparent;position:relative;text-align:center}
.concept-card:hover{transform:translateY(-2px);box-shadow:0 4px 14px rgba(0,0,0,.12)}
.concept-card.done{border-color:#10b981}
.concept-card.done::after{content:'\2713';position:absolute;top:6px;right:8px;color:#10b981;font-size:.85rem;font-weight:800}
.concept-card .c-icon{font-size:1.8rem;margin-bottom:7px}
.concept-card .c-title{font-size:.75rem;font-weight:700;color:#1e293b;line-height:1.3}
.concept-card .c-tap{font-size:.63rem;color:#94a3b8;margin-top:5px}
#modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:100;align-items:center;justify-content:center}
#modal-overlay.open{display:flex}
#modal-box{background:white;border-radius:14px;width:min(780px,95vw);max-height:90vh;overflow-y:auto;box-shadow:0 20px 60px rgba(0,0,0,.25)}
.modal-header{display:flex;justify-content:space-between;align-items:center;padding:18px 22px;border-bottom:1px solid #e2e8f0;position:sticky;top:0;background:white;z-index:1}
.modal-header h2{font-size:1.05rem;font-weight:800;color:#1e293b}
.modal-close{background:none;border:none;font-size:1.3rem;cursor:pointer;color:#94a3b8;padding:4px 8px;border-radius:6px}
.modal-close:hover{background:#f1f5f9;color:#1e293b}
.modal-body{padding:22px;display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:640px){.modal-body{grid-template-columns:1fr}}
.modal-left .section-lbl{font-size:.65rem;text-transform:uppercase;letter-spacing:.09em;color:#94a3b8;font-weight:700;margin:14px 0 6px}
.modal-left .section-lbl:first-child{margin-top:0}
.modal-left p{font-size:.82rem;color:#334155;line-height:1.6;margin-bottom:6px}
.modal-left .formula{font-family:monospace;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px 12px;font-size:.78rem;color:#4f46e5;margin:8px 0;white-space:pre-line}
.modal-left .example-box{background:#fef9e7;border-left:3px solid #f59e0b;border-radius:0 7px 7px 0;padding:10px 12px;font-size:.78rem;color:#78350f;line-height:1.5;margin-top:10px}
.modal-right svg{width:100%;height:auto}
.modal-right .caption{font-size:.72rem;color:#64748b;text-align:center;margin-top:6px}
.stats-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.stat-chip{background:white;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.07);padding:12px 16px;text-align:center;flex:1 1 0;min-width:120px}
.stat-chip .n{font-size:1.4rem;font-weight:800;color:#4f46e5}
.stat-chip .l{font-size:.68rem;color:#64748b;margin-top:2px}
.stat-chip.green .n{color:#10b981}.stat-chip.amber .n{color:#f59e0b}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:stretch}
@media(max-width:820px){.two-col{grid-template-columns:1fr}}
.tbl-controls{display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap}
.tbl-controls input{flex:1;min-width:180px;padding:6px 10px;border:1px solid #e2e8f0;border-radius:6px;font-size:.78rem}
.tbl-wrap{overflow:auto;border:1px solid #e2e8f0;border-radius:8px;flex:1 1 0;min-height:260px}
table{border-collapse:collapse;font-size:.75rem;white-space:nowrap;width:100%}
thead th{background:#f8fafc;padding:7px 10px;text-align:left;font-weight:600;color:#64748b;position:sticky;top:0;border-bottom:1px solid #e2e8f0;z-index:1}
tbody tr:nth-child(even){background:#fafafa}
tbody td{padding:5px 10px;border-bottom:1px solid #f1f5f9;max-width:280px;overflow:hidden;text-overflow:ellipsis}
tbody tr.sel{background:#ede9fe!important}
tbody tr[role=button]{cursor:pointer}
.lbl-pos{background:#ede9fe;color:#4f46e5;padding:1px 7px;border-radius:99px;font-size:.68rem;font-weight:700}
.lbl-neg{background:#f1f5f9;color:#94a3b8;padding:1px 7px;border-radius:99px;font-size:.68rem}
.reminder-banner{background:#fef9e7;border:1.5px solid #f59e0b;border-radius:8px;padding:10px 14px;font-size:.8rem;color:#92400e;margin-bottom:16px;display:none}
input[type=range]{width:100%;accent-color:#4f46e5;cursor:pointer}
.slider-row{margin-bottom:12px}
.slider-row label{display:flex;justify-content:space-between;font-size:.8rem;font-weight:600;color:#334155;margin-bottom:4px}
.slider-row label span{color:#4f46e5;font-weight:700}
.slider-hints{display:flex;justify-content:space-between;font-size:.63rem;color:#94a3b8;margin-top:1px}
/* --- timeline --- */
#tl-detail{min-height:352px;display:flex;flex-direction:column;justify-content:center}
.tl-empty{margin:auto;color:#94a3b8;font-size:.8rem;text-align:center;padding:20px}
.tl-top{flex:1 1 0;min-height:0;display:flex;align-items:center;overflow-x:auto;border-bottom:1px dashed #e2e8f0}
.tl-bot{flex:1 1 0;min-height:0;display:flex;flex-direction:column;padding-top:12px}
.tl-bot-lbl{flex:none;font-size:.72rem;font-weight:700;color:#475569;margin-bottom:6px}
.tl-wrap{position:relative;padding:24px 6px 6px;width:100%}
.tl-line{position:relative;height:140px}
.tl-axis{position:absolute;left:0;right:0;top:46px;height:2px;background:#e2e8f0}
.tl-visit{position:absolute;transform:translateX(-50%);text-align:center;top:0;width:88px}
.tl-dot{width:18px;height:18px;border-radius:50%;background:#4f46e5;margin:28px auto 0;border:2px solid #fff;box-shadow:0 0 0 1px #4f46e5}
.tl-dot.hi{background:#dc2626;box-shadow:0 0 0 1px #dc2626}
.tl-cap{font-size:.64rem;color:#64748b;margin-top:6px;line-height:1.35}
.tl-day{font-size:.6rem;color:#94a3b8;font-weight:600}
.tl-bot .spark{flex:1 1 0;min-height:0;width:100%;height:auto;max-height:100%;display:block}
/* --- sequence explorer slots --- */
.seq-rows{max-height:560px;overflow-y:auto;border:1px solid #e2e8f0;border-radius:8px}
.seq-row{display:flex;align-items:center;gap:12px;padding:8px 14px;border-bottom:1px solid #f1f5f9;font-size:.72rem}
.seq-row .sid{width:56px;color:#475569;font-weight:700;flex:none}
.seq-row .slots{display:flex;gap:6px;flex:1 1 auto}
.slot{flex:1 1 0;min-width:22px;height:26px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:.62rem;font-weight:600;color:#fff}
.slot.real{background:#4f46e5}
.slot.pad{background:repeating-linear-gradient(45deg,#e2e8f0,#e2e8f0 4px,#cbd5e1 4px,#cbd5e1 8px);color:#64748b}
.slot.slot-empty{background:transparent;box-shadow:none;pointer-events:none}
.slot.trunc{background:#fca5a5;color:#7f1d1d}
.seq-row .trunc-col{width:140px;flex:none;color:#b91c1c;font-weight:600;white-space:nowrap;font-size:.68rem}
.seq-hint{color:#94a3b8;font-size:.8rem;text-align:center;padding:28px 10px}
.mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:680px){.mini-grid{grid-template-columns:1fr}}
.mono{font-family:monospace;font-size:.72rem;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;white-space:pre;overflow-x:auto;color:#334155}
.mini-h{font-size:.7rem;font-weight:700;color:#475569;margin-bottom:5px}
.mini-p{font-size:.72rem;color:#64748b;line-height:1.5;margin-top:6px}
.pad-frac{font-size:1.5rem;font-weight:800;color:#f59e0b}
.arena-layout{display:grid;grid-template-columns:330px 1fr;gap:16px;align-items:stretch}
@media(max-width:820px){.arena-layout{grid-template-columns:1fr}}
.arch-scroll{overflow-x:auto}
.arch-table{width:100%;border-collapse:collapse;font-size:.74rem;min-width:560px}
.arch-table th,.arch-table td{border:1px solid #e2e8f0;padding:8px 11px;vertical-align:top;text-align:left;line-height:1.5;
  white-space:normal;overflow:visible;text-overflow:clip;max-width:none;word-break:normal;overflow-wrap:anywhere}
.arch-table thead th{background:#f8fafc;font-weight:800;color:#334155;position:sticky;top:0}
.arch-table thead th.rec{color:#4338ca}
.arch-table thead th.conv{color:#047857}
.arch-table td.sec,.arch-table th.sec{width:140px;font-weight:700;color:#475569;background:#fbfbfd}
.arch-table tbody tr:nth-child(even) td:not(.sec){background:#fafafa}
.arch-table code{font-size:.92em}
.afig{display:flex;flex-direction:column;gap:5px}
.afig-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:.68rem;color:#334155}
.afig .chip{border:1px solid #cbd5e1;border-radius:5px;padding:2px 6px;background:#fff;white-space:nowrap}
.afig .chip.cell{background:#eef2ff;border-color:#c7d2fe;color:#3730a3;font-weight:600}
.afig .chip.scan{background:#ecfdf5;border-color:#a7f3d0;color:#065f46;font-weight:600}
.afig .chip.pred{background:#f1f5f9;font-weight:700}
.afig .mem{color:#7c3aed;letter-spacing:2px;font-weight:700}
.afig .note{font-size:.66rem;color:#94a3b8;font-style:italic}
.afig .cap{font-size:.67rem;color:#475569;margin-top:3px;line-height:1.45}
.afig .strip{display:inline-flex;gap:2px}
.afig .sq{width:11px;height:15px;border-radius:2px;background:#d1fae5;border:1px solid #a7f3d0}
.afig .sq.on{background:#10b981;border-color:#059669}
.arena-layout>.card{margin-bottom:0}
.arena-layout>div{display:flex;flex-direction:column;gap:16px}
#not-set-msg,#metrics-panel{flex:1 1 0;display:flex;flex-direction:column}
#not-set-msg{align-items:center;justify-content:center}
#metric-bars{flex:1 1 0;display:flex;flex-direction:column;justify-content:space-evenly;padding:12px 0 4px}
.arena-layout .m-row{margin-bottom:0}
.tgl{padding:5px 12px;border-radius:6px;border:1px solid #cbd5e1;background:#f8fafc;font-size:.76rem;cursor:pointer;font-weight:700;color:#475569}
.tgl.on{background:#4f46e5;color:#fff;border-color:#4f46e5}
.seg{display:flex;gap:6px}
.seg button{flex:1;padding:6px;border:1px solid #cbd5e1;background:#f8fafc;border-radius:6px;font-size:.76rem;font-weight:700;color:#475569;cursor:pointer}
.seg button.on{background:#4f46e5;color:#fff;border-color:#4f46e5}
.m-row{margin-bottom:12px}
.m-row .ml{display:flex;justify-content:space-between;font-size:.75rem;margin-bottom:3px}
.m-row .ml .mn{color:#64748b;font-weight:600}
.m-row .ml .mv .cur{color:#1e293b;font-weight:700}
.m-row .ml .mv .tgt{color:#94a3b8;margin-left:8px}
.prog-track{height:8px;background:#f1f5f9;border-radius:99px;overflow:hidden}
.prog-fill{height:100%;background:#4f46e5;border-radius:99px;transition:width .4s}
.prog-fill.good{background:#10b981}.prog-fill.warn{background:#f59e0b}
.hint-box{background:#fef9e7;border-left:3px solid #f59e0b;border-radius:0 7px 7px 0;padding:9px 12px;font-size:.78rem;color:#92400e;margin:16px 0;line-height:1.5;display:none}
.optimal-banner{background:#ecfdf5;border:2px solid #10b981;border-radius:10px;padding:16px;text-align:center;display:none;margin:16px 0}
.optimal-banner h3{color:#065f46;font-size:.95rem;margin-bottom:4px}
.optimal-banner p{color:#047857;font-size:.78rem}
.optimal-banner button{margin-top:8px;background:#10b981;color:#fff;border:none;padding:7px 14px;border-radius:8px;font-weight:700;cursor:pointer}
.vis-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:16px 0}
@media(max-width:860px){.vis-grid{grid-template-columns:1fr}}
.vis-card{background:white;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:14px}
.vis-title{font-size:.8rem;font-weight:700;color:#334155;margin-bottom:8px}
.vis-caption{font-size:.71rem;color:#64748b;margin-top:6px;line-height:1.45;min-height:26px}
.cm-cells{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.cm-cell{border-radius:7px;padding:8px 5px;text-align:center}
.cm-cell .cnt{font-size:1.4rem;font-weight:800}
.cm-cell .lbl{font-size:.62rem;margin-top:3px;opacity:.9}
.cm-tp,.cm-tn{background:#dcfce7;color:#166534}
.cm-fn{background:#fee2e2;color:#991b1b}.cm-fp{background:#fef3c7;color:#92400e}
.baseline-card{border-top:3px solid #f59e0b}
.baseline-card .bl-metrics{display:flex;gap:16px;flex-wrap:wrap;font-size:.8rem;margin-top:6px}
.baseline-card .bl-metrics b{color:#f59e0b}
.arena-caption{background:#faf5ff;border-left:3px solid #a78bfa;border-radius:0 6px 6px 0;padding:10px 12px;font-size:.8rem;color:#5b21b6;line-height:1.55;margin-top:12px;display:none}
.legend{font-size:.7rem;color:#64748b;margin-top:6px;text-align:center}
.log-wrap{max-height:220px;overflow-y:auto;border:1px solid #e2e8f0;border-radius:7px;font-size:.72rem}
.opt-y{color:#10b981;font-weight:700}.opt-n{color:#94a3b8}
.hist-bars{display:flex;align-items:flex-end;gap:14px;height:210px;padding:10px 6px 0}
.hist-col{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%}
.hist-bar{width:100%;max-width:52px;background:linear-gradient(180deg,#6366f1,#4f46e5);border-radius:5px 5px 0 0;min-height:3px;display:flex;align-items:flex-start;justify-content:center;color:#fff;font-size:.66rem;font-weight:700;padding-top:3px}
.hist-lbl{font-size:.64rem;color:#64748b;margin-top:6px;line-height:1.42;text-align:center}
code{background:#f1f5f9;padding:1px 4px;border-radius:4px;font-size:.9em}
</style>
</head>
<body>
<div id="top-complete-bar" role="status">
  <div class="tcb-check">&#10003;</div>
  <div>Exploration complete &mdash; <b>optimal parameters</b> ready to copy in the Recurrent Arena tab</div>
</div>
<aside id="sidebar">
  <h1>ISTA 457 / INFO 557 &middot; Neural Networks</h1>
  <p class="subtitle">Assignment 3 &mdash; Model the Patient Journey</p>
  <div class="sid-s">
    <div class="sid-lbl">Your Student ID</div>
    <input type="text" id="sid-input" placeholder="e.g. jdoe" />
    <button onclick="setStudentId()">Set ID &rarr;</button>
    <div id="seed-disp"></div>
  </div>
  <div class="oracle-s" id="oracle-s">
    <div class="sid-lbl">Your Target (Oracle)</div>
    <div class="oracle-row"><span class="nm">F1 Score</span><span class="vl" id="o-f1">--</span></div>
    <div class="oracle-row"><span class="nm">Accuracy</span><span class="vl" id="o-acc">--</span></div>
    <div class="oracle-row"><span class="nm">AUC</span><span class="vl" id="o-auc">--</span></div>
  </div>
  <div class="fc-prog-s" style="border-bottom:none">
    <div class="sid-lbl">Concept Progress</div>
    <div class="fc-prog-bar-wrap"><div class="fc-prog-bar-fill" id="fc-fill" style="width:0%"></div></div>
    <div class="fc-prog-txt" id="fc-txt">0 / 8 concepts explored</div>
  </div>
</aside>

<div id="main">
  <nav class="tab-nav">
    <button class="tab-btn active" onclick="switchTab(0)">Overview &amp; Concepts</button>
    <button class="tab-btn" onclick="switchTab(1)">Patient Timeline Explorer</button>
    <button class="tab-btn" onclick="switchTab(2)">Sequence Explorer</button>
    <button class="tab-btn" onclick="switchTab(3)">Recurrent Arena</button>
  </nav>

  <!-- TAB 1 -->
  <div class="tab-pane active" id="tab-0">
    <div class="info-grid">
      <div class="info-card">
        <h3>&#127973; The Problem: Variable-Length Histories</h3>
        <p>Identify patients with chronic pain from their electronic health records &mdash; the
        same clinical question as Assignments 1 and 2. What changes: a patient is no longer one
        row of numbers but a <strong>sequence of 1&ndash;6 clinical visits</strong>. A fixed-size
        input layer can't read a variable-length history, so you have to reshape it first.</p>
      </div>
      <div class="info-card g">
        <h3>&#128202; The Dataset: Same 320 Patients, Now With Visits</h3>
        <p>The exact Assignment 1&ndash;2 patients, label, and personal seed. A new file
        <code>patient_visits.csv</code> adds 1&ndash;6 visits per patient, each with
        <code>days_since_first_visit</code>, <code>pain_score_at_visit</code>,
        <code>medications_at_visit</code> and <code>visit_type_code</code>.</p>
      </div>
      <div class="info-card a">
        <h3>&#128300; The Approach: Pad &rarr; Mask &rarr; Recur or Convolve</h3>
        <p>Pad every history to a common length, build a boolean mask of real vs. padded steps,
        split <em>patients</em> (not visits) into train/validation, pool per-visit outputs while
        ignoring padding, then build an LSTM/GRU classifier &mdash; and compare it against a
        Conv1D baseline.</p>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Pipeline Stages</div>
      <div class="pipeline">
        <div class="pipe-step"><div class="p-icon">&#128203;</div><div class="p-name">Long Visits Table</div><div class="p-sub">patient_visits.csv</div></div>
        <div class="pipe-arrow">&#8594;</div>
        <div class="pipe-step"><div class="p-icon">&#128207;</div><div class="p-name">Pad / Truncate</div><div class="p-sub">pad_sequence</div></div>
        <div class="pipe-arrow">&#8594;</div>
        <div class="pipe-step"><div class="p-icon">&#127917;</div><div class="p-name">Mask</div><div class="p-sub">create_sequence_mask</div></div>
        <div class="pipe-arrow">&#8594;</div>
        <div class="pipe-step"><div class="p-icon">&#9986;&#65039;</div><div class="p-name">Patient Split</div><div class="p-sub">patient_level_split</div></div>
        <div class="pipe-arrow">&#8594;</div>
        <div class="pipe-step"><div class="p-icon">&#129504;</div><div class="p-name">LSTM / GRU</div><div class="p-sub">build_sequence_classifier</div></div>
        <div class="pipe-arrow">&#8594;</div>
        <div class="pipe-step"><div class="p-icon">&#127919;</div><div class="p-name">Evaluate</div><div class="p-sub">F1 &middot; AUC</div></div>
      </div>
    </div>
    <div class="card">
      <div class="concepts-header">
        <div class="card-title" style="margin-bottom:0">Key Concepts <span class="badge" id="concepts-badge">0/8</span></div>
        <div class="prog-info">Open all 8 to unlock the completion banner.</div>
      </div>
      <div class="complete-banner" id="complete-banner">All 8 concepts explored! Move on to the Patient Timeline Explorer.</div>
      <div class="concept-grid" id="concept-grid"></div>
    </div>
  </div>

  <!-- TAB 2 -->
  <div class="tab-pane" id="tab-1">
    <div class="stats-row">
      <div class="stat-chip"><div class="n" id="s-total">--</div><div class="l">Patients</div></div>
      <div class="stat-chip green"><div class="n" id="s-pos">--</div><div class="l">Pain=1</div></div>
      <div class="stat-chip amber"><div class="n" id="s-neg">--</div><div class="l">Pain=0</div></div>
      <div class="stat-chip"><div class="n" id="s-visits">--</div><div class="l">Visit rows</div></div>
    </div>
    <div class="card" style="background:#fef3c7;border:1px solid #fcd34d">
      <div style="font-size:.84rem;color:#92400e;line-height:1.65">
        <strong>&#128073; Click a patient in the list on the left</strong> to open their
        <strong>Visit Timeline</strong> on the right.
        <strong>Explore several patients with different visit counts</strong> &mdash; try a
        <strong>1-visit</strong> patient, a <strong>3-visit</strong> patient and a
        <strong>6-visit</strong> patient (the <em>visits</em> column shows the count). Notice how the
        pain trajectory, and how much history the model gets to see, changes from patient to patient.
      </div>
    </div>
    <div class="two-col">
      <div class="card" style="display:flex;flex-direction:column">
        <div class="card-title">Patients <span class="badge" id="tbl-badge"></span></div>
        <div class="tbl-controls"><input type="text" id="tl-search" placeholder="Search condition text or ID..." oninput="renderPatientList()"></div>
        <div class="tbl-wrap"><table><thead><tr><th>ID</th><th>label</th><th>visits</th><th>condition_text</th></tr></thead>
          <tbody id="tl-body"></tbody></table></div>
      </div>
      <div class="card">
        <div class="card-title">Visit Timeline <span class="badge" id="tl-pid">select a patient</span></div>
        <div id="tl-detail"><div class="tl-empty">Click a patient on the left to see their visit history.</div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Sequence-Length Histogram</div>
      <div style="font-size:.68rem;color:#94a3b8;font-weight:700;margin-bottom:2px">&uarr; number of patients</div>
      <div class="hist-bars" id="hist-bars"></div>
      <div style="font-size:.68rem;color:#94a3b8;text-align:center;margin-top:2px">visits in the patient's history &rarr;</div>
      <div style="font-size:.78rem;color:#475569;line-height:1.7;margin-top:16px" id="hist-note"></div>
    </div>
  </div>

  <!-- TAB 3 -->
  <div class="tab-pane" id="tab-2">
    <div class="card" style="background:#ede9fe;border:1px solid #c4b5fd">
      <div style="font-size:.8rem;color:#3730a3;line-height:1.6"><strong>Sequence Explorer.</strong>
      Every patient's history has to be forced to the same length before a model can batch it.
      <strong>Choose a maximum number of visits to keep (1&ndash;6)</strong> with the buttons below and
      watch the padding / truncation trade-off across all 320 patients: lower values waste less
      memory on padding but throw away real visits for longer patients; higher values keep every
      visit but most slots become padding a mask has to ignore.
      <strong>Picking a value here is part of completing this activity</strong> &mdash; it becomes
      your <code>max_seq_len</code> in <code>get_sandbox_params()</code>.</div>
    </div>
    <div class="two-col">
      <div class="card">
        <div class="card-title">Choose the maximum number of visits to keep</div>
        <div class="seg" id="msl-seg" style="margin-bottom:10px">
          <button data-v="1" onclick="pickMaxSeq(1)">1</button>
          <button data-v="2" onclick="pickMaxSeq(2)">2</button>
          <button data-v="3" onclick="pickMaxSeq(3)">3</button>
          <button data-v="4" onclick="pickMaxSeq(4)">4</button>
          <button data-v="5" onclick="pickMaxSeq(5)">5</button>
          <button data-v="6" onclick="pickMaxSeq(6)">6</button>
        </div>
        <div style="font-size:.73rem;color:#64748b;line-height:1.55">
          Lower values create <strong>less padding</strong> but remove more patient history.<br>
          Higher values <strong>keep more history</strong> but add more padding.</div>
        <div style="text-align:center;margin-top:14px">
          <div class="pad-frac" id="pad-frac">&ndash;</div>
          <div style="font-size:.72rem;color:#64748b" id="pad-frac-lbl">pick a value above</div>
        </div>
        <div style="font-size:.74rem;color:#64748b;margin-top:10px;line-height:1.5">
          <span id="trunc-note">Choose a value above &mdash; the padding fraction, the truncation
          note and the per-patient preview below all update once you do.</span></div>
        <div style="font-size:.72rem;color:#475569;line-height:1.6;margin-top:14px;padding-top:12px;border-top:1px solid #eef1f5">
          <strong>Note:</strong> in larger real-world datasets a patient may have many more visits
          or notes &mdash; like 30+ &mdash; and when truncating we usually keep the <strong>most recent</strong>
          visits, because recent medical history tends to be more relevant for prediction. There is
          <strong>no single correct value</strong> for <code>max_seq_len</code>: a good choice is
          driven by the sequence-length distribution and validation performance, while avoiding too
          much padding, extra computation, or overfitting.</div>
      </div>
      <div class="card">
        <div class="card-title">Worked Mini-Example &mdash; Padding Patient Visits</div>
        <p style="font-size:.76rem;color:#475569;line-height:1.6;margin-bottom:10px">
        Two patients. Each visit has 4 features <code>[days, pain, meds, visit_type]</code>. We set
        <code>max_len&nbsp;=&nbsp;3</code>, so every patient must end up with <strong>exactly 3
        visit rows</strong>. This is the same toy table checked in
        <code>pipeline/test_pipeline.py</code>.</p>
        <div class="mini-grid">
          <div>
            <div class="mini-h">Patient A &mdash; 3 real visits</div>
            <div class="mono">A = [[ 0, 2, 1, 1],
     [30, 5, 2, 2],
     [61, 7, 3, 3]]</div>
            <p class="mini-p">Already has 3 visits &rarr; <strong>no padding needed</strong>.</p>
          </div>
          <div>
            <div class="mini-h">Patient B &mdash; 2 real visits + 1 padding row</div>
            <div class="mono">B = [[ 0, 1, 0, 1],
     [14, 4, 1, 5],
     [ 0, 0, 0, 0]]  &larr; padding</div>
            <p class="mini-p">Only 2 visits &rarr; <strong>add one zero row</strong> at the end.</p>
          </div>
        </div>
        <div class="mini-grid" style="margin-top:12px">
          <div>
            <div class="mini-h">lengths</div>
            <div class="mono">lengths = [3, 2]</div>
            <p class="mini-p">A has <strong>3</strong> real visits, B has <strong>2</strong>.</p>
          </div>
          <div>
            <div class="mini-h">mask</div>
            <div class="mono">mask = [[ True,  True,  True],
        [ True,  True, False]]</div>
            <p class="mini-p"><strong>True</strong> = real visit, <strong>False</strong> = padding.
            The model uses only the <strong>True</strong> rows and ignores the <strong>False</strong>
            one.</p>
          </div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Per-Patient Padding View <span class="badge" id="seq-badge">first 20 patients</span></div>
      <div class="seq-rows" id="seq-rows"><div class="seq-hint">Choose a value above to preview padding &amp; truncation for the first 20 patients.</div></div>
      <p style="font-size:.72rem;color:#64748b;margin-top:8px">
        <span class="slot real" style="display:inline-flex;flex:none;width:20px;height:14px;vertical-align:middle"></span> real visit &nbsp;
        <span class="slot pad" style="display:inline-flex;flex:none;width:20px;height:14px;vertical-align:middle"></span> padding &nbsp;
        <span class="slot trunc" style="display:inline-flex;flex:none;width:20px;height:14px;vertical-align:middle"></span> truncated (dropped)</p>
    </div>
  </div>

  <!-- TAB 4 -->
  <div class="tab-pane" id="tab-3">
    <div class="reminder-banner" id="reminder-banner">&#128218; You haven't opened all 8 concept cards yet. <span id="reminder-count"></span></div>
    <div class="card" style="background:#ede9fe;border:1px solid #c4b5fd">
      <div style="font-size:.8rem;color:#3730a3;line-height:1.6"><strong>Recurrent Arena.</strong>
      Set your Student ID, then tune the recurrent classifier &mdash; <code>cell_type</code>,
      <code>hidden_units</code>, and <code>bidirectional</code> &mdash; against your personal
      oracle. A fixed <strong>Conv1D baseline</strong> runs on the same padded/masked sequences
      every time, so you can see when reading visits <em>in order</em> actually helps. For this
      small 320-patient dataset, <strong>one hidden layer is enough</strong> because a deeper
      network can easily overfit. The goal is to find a <strong>simple</strong> recurrent model
      that improves validation F1 without adding unnecessary complexity. Make all three bars green
      for the &ldquo;Optimal Performance Reached!&rdquo; banner, then copy your parameters. The
      <code>max_seq_len</code> you copy comes from the Sequence Explorer tab.</div>
    </div>
    <div class="arena-layout">
      <div class="card">
        <div class="card-title">&#129504; Recurrent Model</div>
        <div class="slider-row">
          <label>cell_type <span id="ct-val">lstm</span></label>
          <div class="seg" id="ct-seg">
            <button class="on" data-v="lstm" onclick="pickCell('lstm')">LSTM</button>
            <button data-v="gru" onclick="pickCell('gru')">GRU</button>
          </div>
        </div>
        <div class="slider-row">
          <label>hidden_units <span id="hu-val">24</span></label>
          <input type="range" id="hu-sl" min="4" max="64" step="4" value="24"
            oninput="document.getElementById('hu-val').textContent=this.value;scheduleEval()">
          <div class="slider-hints"><span>4</span><span>32</span><span>64</span></div>
        </div>
        <div class="slider-row">
          <label>bidirectional <span id="bd-val">off</span></label>
          <button class="tgl" id="bd-tgl" onclick="toggleBidir()">off</button>
        </div>
        <div class="slider-row" style="margin-bottom:0">
          <label>max_seq_len (from Sequence Explorer) <span id="arena-msl">not set</span></label>
        </div>
        <div id="arena-msl-warn" style="font-size:.7rem;margin-top:6px;line-height:1.4">
          Set this on the Sequence Explorer tab first.</div>
      </div>
      <div>
        <div class="card" id="not-set-msg"><p style="color:#94a3b8;text-align:center;padding:20px 0">Enter your Student ID in the sidebar to begin.</p></div>
        <div class="card" id="metrics-panel" style="display:none">
          <div class="card-title">Current vs Target &mdash; recurrent model vs your oracle</div>
          <div id="metric-bars"></div>
        </div>
        <div class="card baseline-card" id="baseline-card" style="display:none">
          <div class="card-title">&#128248; Conv1D Baseline (fixed, always on)</div>
          <div style="font-size:.74rem;color:#64748b">Conv1D &rarr; GlobalMaxPooling1D &rarr; Dense on the same
          padded/masked sequences. Not tunable &mdash; it's the yardstick.</div>
          <div class="bl-metrics" id="bl-metrics"></div>
        </div>
      </div>
    </div>
    <div class="hint-box" id="hint-box"></div>
    <div class="optimal-banner" id="opt-banner">
      <h3>Optimal Performance Reached!</h3>
      <p id="opt-msg">Your recurrent model matches the oracle for your seed.</p>
      <p id="reveal-params" style="font-size:.73rem;margin-top:4px"></p>
      <button onclick="copyParams()">Copy params for get_sandbox_params()</button>
    </div>
    <div class="vis-grid" id="vis-panels" style="display:none">
      <div class="vis-card">
        <div class="vis-title">&#128201; Loss Curves &mdash; solid train, dashed val</div>
        <canvas id="loss-canvas" width="320" height="180" style="width:100%;height:auto"></canvas>
        <div class="legend">Recurrent = indigo &middot; Conv1D = amber</div>
      </div>
      <div class="vis-card">
        <div class="vis-title">&#128208; ROC Curve</div>
        <canvas id="roc-canvas" width="300" height="180" style="width:100%;height:auto"></canvas>
        <div class="vis-caption" id="roc-caption"></div>
      </div>
      <div class="vis-card">
        <div class="vis-title">&#128290; Confusion Matrix (recurrent)</div>
        <div class="cm-cells">
          <div class="cm-cell cm-tp" id="cm-tp"><div class="cnt">--</div><div class="lbl">TP pain caught</div></div>
          <div class="cm-cell cm-fn" id="cm-fn"><div class="cnt">--</div><div class="lbl">FN pain missed</div></div>
          <div class="cm-cell cm-fp" id="cm-fp"><div class="cnt">--</div><div class="lbl">FP false alarm</div></div>
          <div class="cm-cell cm-tn" id="cm-tn"><div class="cnt">--</div><div class="lbl">TN healthy cleared</div></div>
        </div>
        <div class="vis-caption" id="cm-caption"></div>
      </div>
    </div>
    <div class="arena-caption" id="arena-caption"></div>

    <div class="card">
      <div class="card-title">&#129504; Recurrent (LSTM / GRU) vs. &#128248; Conv1D &mdash; architecture side by side</div>
      <p style="font-size:.76rem;color:#64748b;line-height:1.6;margin-bottom:12px">Both models read the
      same padded, masked <code>(max_seq_len &times; 4)</code> sequence and end in
      <code>Dense(1) &rarr; sigmoid</code>. What differs is the middle &mdash; how they turn the
      sequence into one vector.</p>
      <div class="arch-scroll"><table class="arch-table">
        <thead><tr>
          <th class="sec">Section</th>
          <th class="rec">&#129504; LSTM / GRU &mdash; Memory-Based Model</th>
          <th class="conv">&#128248; Conv1D &mdash; Pattern-Finding Model</th>
        </tr></thead>
        <tbody>
          <tr>
            <td class="sec">How it works</td>
            <td>
              <div class="afig">
                <div class="afig-row"><span class="chip">Visit 1</span>&rarr;<span class="chip cell">LSTM cell</span>&rarr;<span class="mem">&#8226;&#8226;&#8226;</span>memory</div>
                <div class="afig-row"><span class="chip">Visit 2</span>&rarr;<span class="chip cell">LSTM cell</span>&rarr;<span class="mem">&#8226;&#8226;&#8226;</span>memory</div>
                <div class="afig-row note">one at a time&hellip;</div>
                <div class="afig-row"><span class="chip">Visit 6</span>&rarr;<span class="chip cell">LSTM cell</span>&rarr;<span class="chip pred">prediction</span></div>
                <div class="cap">Each cell passes its memory forward &mdash; visit&nbsp;1 still influences the final answer.</div>
              </div>
            </td>
            <td>
              <div class="afig">
                <div class="afig-row">All 6 visits at once:</div>
                <div class="afig-row"><span class="strip"><span class="sq on"></span><span class="sq on"></span><span class="sq on"></span><span class="sq"></span><span class="sq"></span><span class="sq"></span></span>&rarr;<span class="chip scan">window scan</span></div>
                <div class="afig-row note">slides across &rarr;</div>
                <div class="afig-row"><span class="strip"><span class="sq"></span><span class="sq"></span><span class="sq on"></span><span class="sq on"></span><span class="sq on"></span><span class="sq"></span></span>&rarr;<span class="chip scan">window scan</span></div>
                <div class="afig-row"><span class="chip">keep strongest signal</span>&rarr;<span class="chip pred">prediction</span></div>
                <div class="cap">All windows run in parallel &mdash; fast, but forgets the overall order.</div>
              </div>
            </td>
          </tr>
          <tr><td class="sec">Main idea</td>
            <td>Reads the patient history like a <b>story</b>.</td>
            <td>Reads the patient history like small <b>snapshots</b>.</td></tr>
          <tr><td class="sec">How it reads visits</td>
            <td>Processes visits <b>one by one, in order</b>.</td>
            <td>Looks at <b>small groups of nearby visits</b> at the same time.</td></tr>
          <tr><td class="sec">Example</td>
            <td><code>Visit 1 &rarr; Visit 2 &rarr; Visit 3 &rarr; Visit 4 &rarr; Prediction</code></td>
            <td><code>[Visit 1, Visit 2] &rarr; [Visit 2, Visit 3] &rarr; [Visit 3, Visit 4] &rarr; Prediction</code></td></tr>
          <tr><td class="sec">What it remembers</td>
            <td>Can carry information from <b>earlier visits to later visits</b>.</td>
            <td>Mainly focuses on <b>short local patterns</b>.</td></tr>
          <tr><td class="sec">Best for</td>
            <td>Long-term patterns, trends, or changes over time.</td>
            <td>Short-term patterns, sudden changes, or nearby-visit signals.</td></tr>
          <tr><td class="sec">Strengths</td>
            <td>Understands visit order; useful when early visits still matter later; good for slow changes over time.</td>
            <td>Faster to train; cheaper to run; good at finding repeated local patterns.</td></tr>
          <tr><td class="sec">Weaknesses</td>
            <td>Slower and more computationally expensive; may overfit on small datasets.</td>
            <td>May miss long-range history; depends on the window size; less natural memory.</td></tr>
          <tr><td class="sec">Runtime</td>
            <td>Usually <b>slower</b> because it reads visits step by step.</td>
            <td>Usually <b>faster</b> because it can process visit windows in parallel.</td></tr>
          <tr><td class="sec">Simple takeaway</td>
            <td>Use <b>LSTM/GRU</b> when the order and full patient history matter.</td>
            <td>Use <b>Conv1D</b> when nearby visits contain enough useful signal and speed matters.</td></tr>
        </tbody>
      </table></div>
    </div>

    <div class="card" id="log-card" style="display:none">
      <div class="card-title">Interaction Log <span class="badge" id="log-cnt">0</span></div>
      <div class="log-wrap"><table>
        <thead><tr><th>#</th><th>max_len</th><th>hidden</th><th>cell</th><th>bidir</th><th>AUC</th><th>F1</th><th>Optimal</th></tr></thead>
        <tbody id="log-body"></tbody>
      </table></div>
    </div>
  </div>
</div>

<div id="modal-overlay" onclick="closeModal()">
  <div id="modal-box" onclick="event.stopPropagation()">
    <div class="modal-header"><h2 id="modal-title"></h2>
      <button class="modal-close" onclick="closeModal()">&#10005;</button></div>
    <div class="modal-body"><div class="modal-left" id="modal-left"></div><div class="modal-right" id="modal-right"></div></div>
  </div>
</div>

<script>
const state={studentId:null,seed:null,oracle:null,current:null,isOptimal:false,
  evalTimer:null,explored:new Set(),log:[],cell:'lstm',bidir:false,seqTouched:false,maxSeqLen:null};
const VTYPE=['','routine','urgent','specialist','procedure','telehealth'];
let DATA=null, VISITS_BY_PID={};

function svg(inner){return '<svg viewBox="0 0 280 170" xmlns="http://www.w3.org/2000/svg"><rect width="280" height="170" fill="#f8fafc" rx="8"/>'+inner+'</svg>';}
const CONCEPTS=[
 {icon:"&#128337;",title:"Sequential Clinical Data",
  explanation:`A patient's record is a list of visits in time order, not one snapshot. The order of the visits tells a story that a single row of averages would hide.\n\nHere each patient has 1 to 6 visits, and every visit records four numbers: days since the first visit, a pain score, how many medications were given, and the visit type.`,
  formula:`patient  =  [visit 1, visit 2, ..., visit T],   1 <= T <= 6\nvisit    =  [days, pain, meds, visit_type]`,
  example:`build_patient_sequences() turns the long visits table into an (N, max_len, 4) array, one patient at a time.`,
  visual:svg('<line x1="20" y1="120" x2="260" y2="120" stroke="#cbd5e1"/><circle cx="45" cy="120" r="7" fill="#4f46e5"/><circle cx="110" cy="120" r="7" fill="#4f46e5"/><circle cx="150" cy="120" r="7" fill="#4f46e5"/><circle cx="235" cy="120" r="7" fill="#4f46e5"/><text x="140" y="60" text-anchor="middle" font-size="11" fill="#475569">visits are spaced unevenly in time</text>')},
 {icon:"&#128207;",title:"Padding",
  explanation:`A batch of data is a rectangle, so every patient in it must have the same number of visit rows.\n\nIf a patient has fewer visits than max_len, we add rows of zeros at the end. If a patient has more, we drop the oldest visits and keep the most recent ones.`,
  formula:`fewer than max_len visits  ->  add zero rows at the end\nmore than max_len visits    ->  keep the last max_len visits`,
  example:`pad_sequence(visits, max_len) returns a fixed (max_len, 4) array -- keep the END of the sequence when trimming.`,
  visual:svg('<text x="56" y="16" text-anchor="middle" font-size="9" fill="#475569">2 visits, pad to 3</text><rect x="26" y="24" width="60" height="20" rx="3" fill="#4f46e5"/><text x="56" y="38" text-anchor="middle" font-size="9" fill="#fff">v1</text><rect x="26" y="48" width="60" height="20" rx="3" fill="#4f46e5"/><text x="56" y="62" text-anchor="middle" font-size="9" fill="#fff">v2</text><rect x="26" y="72" width="60" height="20" rx="3" fill="#e2e8f0"/><text x="56" y="86" text-anchor="middle" font-size="9" fill="#64748b">0 0 0 0</text><text x="56" y="108" text-anchor="middle" font-size="8" fill="#94a3b8">added zero row</text><text x="206" y="16" text-anchor="middle" font-size="9" fill="#475569">4 visits, keep last 3</text><rect x="176" y="24" width="62" height="20" rx="3" fill="#fca5a5"/><text x="207" y="38" text-anchor="middle" font-size="9" fill="#7f1d1d">v1 dropped</text><rect x="176" y="48" width="62" height="20" rx="3" fill="#4f46e5"/><text x="207" y="62" text-anchor="middle" font-size="9" fill="#fff">v2</text><rect x="176" y="72" width="62" height="20" rx="3" fill="#4f46e5"/><text x="207" y="86" text-anchor="middle" font-size="9" fill="#fff">v3</text><rect x="176" y="96" width="62" height="20" rx="3" fill="#4f46e5"/><text x="207" y="110" text-anchor="middle" font-size="9" fill="#fff">v4</text>')},
 {icon:"&#127917;",title:"Masking",
  explanation:`After padding, the model needs to know which rows are real visits and which are just filler.\n\nA mask is a grid of True / False: True where there is a real visit, False where there is padding. Pooling and recurrent layers use it to skip the padded rows.`,
  formula:`mask[i, t] = True   when t is before patient i's real visit count`,
  example:`create_sequence_mask(lengths, max_len) builds the whole grid with one broadcast -- no Python loop.`,
  visual:svg('<rect x="40" y="50" width="26" height="26" fill="#10b981"/><rect x="70" y="50" width="26" height="26" fill="#10b981"/><rect x="100" y="50" width="26" height="26" fill="#e2e8f0"/><text x="53" y="68" font-size="12" fill="#fff" text-anchor="middle">T</text><text x="83" y="68" font-size="12" fill="#fff" text-anchor="middle">T</text><text x="113" y="68" font-size="12" fill="#64748b" text-anchor="middle">F</text><text x="140" y="110" font-size="10" fill="#475569">T = real visit, F = padding</text>')},
 {icon:"&#128274;",title:"LSTM Gates",
  explanation:`A plain RNN tends to forget the early visits by the time it reaches the last one.\n\nAn LSTM fixes this by keeping two memory tracks: a long-term cell state that runs straight through every visit, and a hidden state it reads out. Three gates control it at each visit -- forget (erase old memory), input (write new info), and output (what to pass forward).`,
  formula:`forget gate  ->  erase old memory\ninput gate   ->  write new info\noutput gate  ->  what to pass forward`,
  example:`build_sequence_classifier(..., cell_type="lstm", hidden_units=H) creates one nn.LSTM(4, H, batch_first=True).`,
  visual:`<svg viewBox="22 12 400 448" xmlns="http://www.w3.org/2000/svg" font-family="Arial, Helvetica, sans-serif">
<rect width="900" height="500" fill="#ffffff"/>

  <!-- LSTM header -->
  <rect x="30" y="20" width="380" height="38" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="220" y="44" text-anchor="middle" font-size="15" font-weight="bold" fill="#26215C">LSTM &#8212; 3 gates</text>

  <!-- GRU header -->
  <rect x="490" y="20" width="380" height="38" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="1.5"/>
  <text x="680" y="44" text-anchor="middle" font-size="15" font-weight="bold" fill="#04342C">GRU &#8212; 2 gates</text>

  <!-- LSTM outer box -->
  <rect x="30" y="75" width="380" height="370" rx="14" fill="#F8F8FF" stroke="#534AB7" stroke-width="2"/>

  <!-- GRU outer box -->
  <rect x="490" y="75" width="380" height="370" rx="14" fill="#F0FAF5" stroke="#0F6E56" stroke-width="2"/>

  <!-- LSTM dashed flow lines -->
  <line x1="95" y1="112" x2="95" y2="410" stroke="#888780" stroke-width="1.5" stroke-dasharray="5 4"/>
  <line x1="200" y1="112" x2="200" y2="410" stroke="#534AB7" stroke-width="1.5" stroke-dasharray="5 4"/>

  <!-- LSTM flow labels -->
  <text x="95" y="107" text-anchor="middle" font-size="12" fill="#5F5E5A">x (visit)</text>
  <text x="200" y="107" text-anchor="middle" font-size="12" fill="#3C3489">h (memory)</text>

  <!-- Forget gate -->
  <rect x="55" y="125" width="160" height="58" rx="8" fill="#FAECE7" stroke="#993C1D" stroke-width="2"/>
  <text x="135" y="149" text-anchor="middle" font-size="14" font-weight="bold" fill="#712B13">Forget gate</text>
  <text x="135" y="168" text-anchor="middle" font-size="12" fill="#993C1D">Erase old memory</text>

  <!-- Input gate -->
  <rect x="55" y="215" width="160" height="58" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="2"/>
  <text x="135" y="239" text-anchor="middle" font-size="14" font-weight="bold" fill="#3C3489">Input gate</text>
  <text x="135" y="258" text-anchor="middle" font-size="12" fill="#534AB7">Write new info</text>

  <!-- Output gate -->
  <rect x="55" y="305" width="160" height="58" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="2"/>
  <text x="135" y="329" text-anchor="middle" font-size="14" font-weight="bold" fill="#085041">Output gate</text>
  <text x="135" y="348" text-anchor="middle" font-size="12" fill="#0F6E56">Pass forward</text>

  <!-- Cell state box -->
  <rect x="260" y="195" width="128" height="58" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="2"/>
  <text x="324" y="219" text-anchor="middle" font-size="13" font-weight="bold" fill="#2C2C2A">Cell state</text>
  <text x="324" y="238" text-anchor="middle" font-size="11" fill="#5F5E5A">Long-term memory</text>

  <!-- LSTM arrows -->
  <defs>
    <marker id="L_arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#444441" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <marker id="L_arrg" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#0F6E56" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <marker id="L_arrp" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#534AB7" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <marker id="L_arrr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#993C1D" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <line x1="95" y1="112" x2="95" y2="125" stroke="#444441" stroke-width="2" marker-end="url(#L_arr)"/>
  <line x1="95" y1="183" x2="95" y2="215" stroke="#444441" stroke-width="2" marker-end="url(#L_arr)"/>
  <line x1="95" y1="273" x2="95" y2="305" stroke="#444441" stroke-width="2" marker-end="url(#L_arr)"/>
  <line x1="200" y1="112" x2="200" y2="125" stroke="#534AB7" stroke-width="2" marker-end="url(#L_arrp)"/>
  <line x1="200" y1="183" x2="200" y2="215" stroke="#534AB7" stroke-width="2" marker-end="url(#L_arrp)"/>
  <line x1="200" y1="273" x2="200" y2="305" stroke="#534AB7" stroke-width="2" marker-end="url(#L_arrp)"/>

  <line x1="215" y1="148" x2="258" y2="210" stroke="#993C1D" stroke-width="2" marker-end="url(#L_arrr)"/>
  <line x1="215" y1="244" x2="258" y2="232" stroke="#534AB7" stroke-width="2" marker-end="url(#L_arrp)"/>
  <line x1="324" y1="253" x2="324" y2="303" stroke="#444441" stroke-width="2" marker-end="url(#L_arr)"/>

  <line x1="215" y1="334" x2="388" y2="334" stroke="#0F6E56" stroke-width="2.5" marker-end="url(#L_arrg)"/>
  <text x="295" y="326" text-anchor="middle" font-size="11" fill="#0F6E56" font-weight="bold">output h</text>

  <text x="220" y="445" text-anchor="middle" font-size="12" fill="#5F5E5A">2 separate memory tracks (h + cell state)</text>

  <!-- GRU dashed flow lines -->
  <line x1="560" y1="112" x2="560" y2="380" stroke="#888780" stroke-width="1.5" stroke-dasharray="5 4"/>
  <line x1="660" y1="112" x2="660" y2="380" stroke="#0F6E56" stroke-width="1.5" stroke-dasharray="5 4"/>

  <text x="560" y="107" text-anchor="middle" font-size="12" fill="#5F5E5A">x (visit)</text>
  <text x="660" y="107" text-anchor="middle" font-size="12" fill="#085041">h (memory)</text>

  <!-- Reset gate -->
  <rect x="520" y="155" width="160" height="58" rx="8" fill="#FAEEDA" stroke="#854F0B" stroke-width="2"/>
  <text x="600" y="179" text-anchor="middle" font-size="14" font-weight="bold" fill="#633806">Reset gate</text>
  <text x="600" y="198" text-anchor="middle" font-size="12" fill="#854F0B">How much past to use</text>

  <!-- Update gate -->
  <rect x="520" y="255" width="160" height="58" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="2"/>
  <text x="600" y="279" text-anchor="middle" font-size="14" font-weight="bold" fill="#085041">Update gate</text>
  <text x="600" y="298" text-anchor="middle" font-size="12" fill="#0F6E56">Blend old + new</text>

  <!-- Hidden state box -->
  <rect x="720" y="165" width="128" height="58" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="2"/>
  <text x="784" y="189" text-anchor="middle" font-size="13" font-weight="bold" fill="#2C2C2A">Hidden state</text>
  <text x="784" y="208" text-anchor="middle" font-size="11" fill="#5F5E5A">Only memory</text>

  <!-- GRU arrows -->
  <line x1="560" y1="112" x2="560" y2="155" stroke="#444441" stroke-width="2" marker-end="url(#L_arr)"/>
  <line x1="560" y1="213" x2="560" y2="255" stroke="#444441" stroke-width="2" marker-end="url(#L_arr)"/>
  <line x1="660" y1="112" x2="660" y2="155" stroke="#0F6E56" stroke-width="2" marker-end="url(#L_arrg)"/>
  <line x1="660" y1="213" x2="660" y2="255" stroke="#0F6E56" stroke-width="2" marker-end="url(#L_arrg)"/>

  <line x1="680" y1="180" x2="718" y2="188" stroke="#854F0B" stroke-width="2" marker-end="url(#L_arr)"/>
  <line x1="680" y1="284" x2="848" y2="284" stroke="#0F6E56" stroke-width="2.5" marker-end="url(#L_arrg)"/>
  <text x="758" y="276" text-anchor="middle" font-size="11" fill="#0F6E56" font-weight="bold">output h</text>

  <text x="680" y="445" text-anchor="middle" font-size="12" fill="#5F5E5A">1 combined memory track (h only)</text>
</svg>`},
 {icon:"&#9881;&#65039;",title:"GRU Gates",
  explanation:`A GRU is a lighter LSTM. It keeps just one memory track (the hidden state) and only two gates: reset decides how much of the past to use, and update blends the old state with the new one.\n\nFewer gates means fewer parameters and faster training, and on small datasets it usually does just as well as an LSTM.`,
  formula:`reset gate   ->  how much past to use\nupdate gate  ->  blend old state + new state`,
  example:`Pass cell_type="gru" to build_sequence_classifier for an nn.GRU layer instead of nn.LSTM.`,
  visual:`<svg viewBox="482 12 400 448" xmlns="http://www.w3.org/2000/svg" font-family="Arial, Helvetica, sans-serif">
<rect width="900" height="500" fill="#ffffff"/>

  <!-- LSTM header -->
  <rect x="30" y="20" width="380" height="38" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="1.5"/>
  <text x="220" y="44" text-anchor="middle" font-size="15" font-weight="bold" fill="#26215C">LSTM &#8212; 3 gates</text>

  <!-- GRU header -->
  <rect x="490" y="20" width="380" height="38" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="1.5"/>
  <text x="680" y="44" text-anchor="middle" font-size="15" font-weight="bold" fill="#04342C">GRU &#8212; 2 gates</text>

  <!-- LSTM outer box -->
  <rect x="30" y="75" width="380" height="370" rx="14" fill="#F8F8FF" stroke="#534AB7" stroke-width="2"/>

  <!-- GRU outer box -->
  <rect x="490" y="75" width="380" height="370" rx="14" fill="#F0FAF5" stroke="#0F6E56" stroke-width="2"/>

  <!-- LSTM dashed flow lines -->
  <line x1="95" y1="112" x2="95" y2="410" stroke="#888780" stroke-width="1.5" stroke-dasharray="5 4"/>
  <line x1="200" y1="112" x2="200" y2="410" stroke="#534AB7" stroke-width="1.5" stroke-dasharray="5 4"/>

  <!-- LSTM flow labels -->
  <text x="95" y="107" text-anchor="middle" font-size="12" fill="#5F5E5A">x (visit)</text>
  <text x="200" y="107" text-anchor="middle" font-size="12" fill="#3C3489">h (memory)</text>

  <!-- Forget gate -->
  <rect x="55" y="125" width="160" height="58" rx="8" fill="#FAECE7" stroke="#993C1D" stroke-width="2"/>
  <text x="135" y="149" text-anchor="middle" font-size="14" font-weight="bold" fill="#712B13">Forget gate</text>
  <text x="135" y="168" text-anchor="middle" font-size="12" fill="#993C1D">Erase old memory</text>

  <!-- Input gate -->
  <rect x="55" y="215" width="160" height="58" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="2"/>
  <text x="135" y="239" text-anchor="middle" font-size="14" font-weight="bold" fill="#3C3489">Input gate</text>
  <text x="135" y="258" text-anchor="middle" font-size="12" fill="#534AB7">Write new info</text>

  <!-- Output gate -->
  <rect x="55" y="305" width="160" height="58" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="2"/>
  <text x="135" y="329" text-anchor="middle" font-size="14" font-weight="bold" fill="#085041">Output gate</text>
  <text x="135" y="348" text-anchor="middle" font-size="12" fill="#0F6E56">Pass forward</text>

  <!-- Cell state box -->
  <rect x="260" y="195" width="128" height="58" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="2"/>
  <text x="324" y="219" text-anchor="middle" font-size="13" font-weight="bold" fill="#2C2C2A">Cell state</text>
  <text x="324" y="238" text-anchor="middle" font-size="11" fill="#5F5E5A">Long-term memory</text>

  <!-- LSTM arrows -->
  <defs>
    <marker id="G_arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#444441" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <marker id="G_arrg" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#0F6E56" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <marker id="G_arrp" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#534AB7" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <marker id="G_arrr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#993C1D" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <line x1="95" y1="112" x2="95" y2="125" stroke="#444441" stroke-width="2" marker-end="url(#G_arr)"/>
  <line x1="95" y1="183" x2="95" y2="215" stroke="#444441" stroke-width="2" marker-end="url(#G_arr)"/>
  <line x1="95" y1="273" x2="95" y2="305" stroke="#444441" stroke-width="2" marker-end="url(#G_arr)"/>
  <line x1="200" y1="112" x2="200" y2="125" stroke="#534AB7" stroke-width="2" marker-end="url(#G_arrp)"/>
  <line x1="200" y1="183" x2="200" y2="215" stroke="#534AB7" stroke-width="2" marker-end="url(#G_arrp)"/>
  <line x1="200" y1="273" x2="200" y2="305" stroke="#534AB7" stroke-width="2" marker-end="url(#G_arrp)"/>

  <line x1="215" y1="148" x2="258" y2="210" stroke="#993C1D" stroke-width="2" marker-end="url(#G_arrr)"/>
  <line x1="215" y1="244" x2="258" y2="232" stroke="#534AB7" stroke-width="2" marker-end="url(#G_arrp)"/>
  <line x1="324" y1="253" x2="324" y2="303" stroke="#444441" stroke-width="2" marker-end="url(#G_arr)"/>

  <line x1="215" y1="334" x2="388" y2="334" stroke="#0F6E56" stroke-width="2.5" marker-end="url(#G_arrg)"/>
  <text x="295" y="326" text-anchor="middle" font-size="11" fill="#0F6E56" font-weight="bold">output h</text>

  <text x="220" y="445" text-anchor="middle" font-size="12" fill="#5F5E5A">2 separate memory tracks (h + cell state)</text>

  <!-- GRU dashed flow lines -->
  <line x1="560" y1="112" x2="560" y2="380" stroke="#888780" stroke-width="1.5" stroke-dasharray="5 4"/>
  <line x1="660" y1="112" x2="660" y2="380" stroke="#0F6E56" stroke-width="1.5" stroke-dasharray="5 4"/>

  <text x="560" y="107" text-anchor="middle" font-size="12" fill="#5F5E5A">x (visit)</text>
  <text x="660" y="107" text-anchor="middle" font-size="12" fill="#085041">h (memory)</text>

  <!-- Reset gate -->
  <rect x="520" y="155" width="160" height="58" rx="8" fill="#FAEEDA" stroke="#854F0B" stroke-width="2"/>
  <text x="600" y="179" text-anchor="middle" font-size="14" font-weight="bold" fill="#633806">Reset gate</text>
  <text x="600" y="198" text-anchor="middle" font-size="12" fill="#854F0B">How much past to use</text>

  <!-- Update gate -->
  <rect x="520" y="255" width="160" height="58" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="2"/>
  <text x="600" y="279" text-anchor="middle" font-size="14" font-weight="bold" fill="#085041">Update gate</text>
  <text x="600" y="298" text-anchor="middle" font-size="12" fill="#0F6E56">Blend old + new</text>

  <!-- Hidden state box -->
  <rect x="720" y="165" width="128" height="58" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="2"/>
  <text x="784" y="189" text-anchor="middle" font-size="13" font-weight="bold" fill="#2C2C2A">Hidden state</text>
  <text x="784" y="208" text-anchor="middle" font-size="11" fill="#5F5E5A">Only memory</text>

  <!-- GRU arrows -->
  <line x1="560" y1="112" x2="560" y2="155" stroke="#444441" stroke-width="2" marker-end="url(#G_arr)"/>
  <line x1="560" y1="213" x2="560" y2="255" stroke="#444441" stroke-width="2" marker-end="url(#G_arr)"/>
  <line x1="660" y1="112" x2="660" y2="155" stroke="#0F6E56" stroke-width="2" marker-end="url(#G_arrg)"/>
  <line x1="660" y1="213" x2="660" y2="255" stroke="#0F6E56" stroke-width="2" marker-end="url(#G_arrg)"/>

  <line x1="680" y1="180" x2="718" y2="188" stroke="#854F0B" stroke-width="2" marker-end="url(#G_arr)"/>
  <line x1="680" y1="284" x2="848" y2="284" stroke="#0F6E56" stroke-width="2.5" marker-end="url(#G_arrg)"/>
  <text x="758" y="276" text-anchor="middle" font-size="11" fill="#0F6E56" font-weight="bold">output h</text>

  <text x="680" y="445" text-anchor="middle" font-size="12" fill="#5F5E5A">1 combined memory track (h only)</text>
</svg>`},
 {icon:"&#8646;",title:"Bidirectional Processing",
  explanation:`A normal RNN reads visits front to back, so at visit 2 it has not seen visits 3 to 6 yet.\n\nA bidirectional layer adds a second copy that reads back to front, then joins the two. Now every visit is understood with both past and future context. It doubles the parameters and only makes sense when you already have the whole sequence (here you do).`,
  formula:`output = [ forward pass: visit 1 -> T ,  backward pass: visit T -> 1 ]`,
  example:`bidirectional=True sets bidirectional=True on the nn.LSTM / nn.GRU; the Dense head then sees hidden_units * 2 numbers.`,
  visual:svg('<text x="140" y="18" text-anchor="middle" font-size="10" fill="#475569">read the visits both ways</text><circle cx="45" cy="85" r="7" fill="#94a3b8"/><circle cx="83" cy="85" r="7" fill="#94a3b8"/><circle cx="121" cy="85" r="7" fill="#94a3b8"/><circle cx="159" cy="85" r="7" fill="#94a3b8"/><circle cx="197" cy="85" r="7" fill="#94a3b8"/><circle cx="235" cy="85" r="7" fill="#94a3b8"/><path d="M38 58 H244" stroke="#4f46e5" stroke-width="2"/><path d="M244 58 l-8 -4 v8 z" fill="#4f46e5"/><text x="140" y="50" text-anchor="middle" font-size="8" fill="#4f46e5">forward: visit 1 -> 6</text><path d="M244 116 H38" stroke="#f59e0b" stroke-width="2"/><path d="M38 116 l8 -4 v8 z" fill="#f59e0b"/><text x="140" y="132" text-anchor="middle" font-size="8" fill="#f59e0b">backward: visit 6 -> 1</text><text x="140" y="152" text-anchor="middle" font-size="8" fill="#94a3b8">join both, so every visit sees past AND future</text>')},
 {icon:"&#128246;",title:"1D Convolution as a Visit-Pattern Detector",
  explanation:`A Conv1D layer slides a small window (say 3 visits wide) along the sequence, looking for a short local pattern like "pain jumps, then medications jump".\n\nAfter scanning, it keeps only the strongest match it found anywhere. It is fast and good at short patterns, but it has no memory, so it can miss a slow change spread across all 6 visits.`,
  formula:`slide a small window over the visits  ->  score every position\nkeep the single highest score`,
  example:`The Arena's fixed baseline is Conv1D -> GlobalMaxPooling -> Dense on the same sequences.`,
  visual:svg('<rect x="30" y="70" width="24" height="24" fill="#f59e0b"/><rect x="58" y="70" width="24" height="24" fill="#f59e0b"/><rect x="86" y="70" width="24" height="24" fill="#f59e0b"/><rect x="120" y="70" width="24" height="24" fill="#e2e8f0"/><rect x="148" y="70" width="24" height="24" fill="#e2e8f0"/><text x="140" y="120" text-anchor="middle" font-size="10" fill="#475569">window spans 3 adjacent visits</text>')},
 {icon:"&#128293;",title:"Overfitting in Small Sequence Datasets",
  explanation:`With only ~320 training patients, a large recurrent layer can simply memorise them.\n\nYou see the training loss keep dropping while validation performance stalls and then gets worse. Fixes: fewer hidden units, the simpler GRU cell, or stopping training earlier. This is why the Arena suggests keeping the model small.`,
  formula:`overfitting  ->  training loss keeps falling, validation loss starts rising`,
  example:`If a wide model overfits your seed, the Arena's hint tells you to lower hidden_units.`,
  visual:svg('<line x1="36" y1="18" x2="36" y2="140" stroke="#cbd5e1"/><line x1="36" y1="140" x2="264" y2="140" stroke="#cbd5e1"/><text x="15" y="90" font-size="9" fill="#64748b" transform="rotate(-90 15 90)">loss</text><text x="150" y="159" text-anchor="middle" font-size="9" fill="#64748b">training epochs</text><path d="M42 38 C92 88 165 116 258 126" stroke="#4f46e5" stroke-width="2.5" fill="none"/><path d="M42 58 C92 96 128 100 150 100 C192 100 232 66 258 42" stroke="#f59e0b" stroke-width="2.5" fill="none"/><circle cx="150" cy="100" r="3.5" fill="#ef4444"/><line x1="150" y1="103" x2="150" y2="140" stroke="#ef4444" stroke-dasharray="3"/><text x="150" y="92" text-anchor="middle" font-size="8" fill="#ef4444">best point</text><text x="58" y="34" font-size="8" fill="#4f46e5">train</text><text x="232" y="38" font-size="8" fill="#f59e0b">validation</text>')},
];

function switchTab(i){
  document.querySelectorAll('.tab-btn').forEach((b,j)=>b.classList.toggle('active',i===j));
  document.querySelectorAll('.tab-pane').forEach((p,j)=>p.classList.toggle('active',i===j));
  if(i===2) drawSeqExplorer();
  if(i===3){ updateReminder(); setArenaMslNote(); }
}
function setArenaMslNote(){
  const w=document.getElementById('arena-msl-warn'); if(!w) return;
  if(state.seqTouched){
    w.style.color='#94a3b8';
    w.textContent='Update in the Sequence Explorer tab if you want a different value.';
  }else{
    w.style.color='#b45309';
    w.textContent='Set this on the Sequence Explorer tab first.';
  }
}

// ---- concepts ----
function renderConcepts(){
  document.getElementById('concept-grid').innerHTML=CONCEPTS.map((c,i)=>
    `<div class="concept-card" id="cc-${i}" onclick="openConcept(${i})">
      <div class="c-icon">${c.icon}</div><div class="c-title">${c.title}</div>
      <div class="c-tap">Click to explore</div></div>`).join('');
}
function openConcept(i){
  const c=CONCEPTS[i];
  document.getElementById('modal-title').innerHTML=c.icon+'&nbsp; '+c.title;
  document.getElementById('modal-left').innerHTML=
    `<div class="section-lbl">What it is</div><p>${c.explanation.replace(/\n/g,'</p><p>')}</p>`+
    `<div class="section-lbl">The Math</div><div class="formula">${c.formula}</div>`+
    `<div class="section-lbl">In Assignment 3</div><div class="example-box">${c.example}</div>`;
  document.getElementById('modal-right').innerHTML=c.visual+`<div class="caption">Figure: ${c.title}</div>`;
  document.getElementById('modal-overlay').classList.add('open');
  if(!state.explored.has(i)){
    state.explored.add(i);
    document.getElementById('cc-'+i).classList.add('done');
    updateConceptProgress();
  }
}
function closeModal(){document.getElementById('modal-overlay').classList.remove('open');}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal();});
function updateConceptProgress(){
  const n=state.explored.size,t=CONCEPTS.length;
  document.getElementById('fc-fill').style.width=(n/t*100)+'%';
  document.getElementById('fc-txt').textContent=`${n} / ${t} concepts explored`;
  document.getElementById('concepts-badge').textContent=`${n}/${t}`;
  if(n>=t) document.getElementById('complete-banner').style.display='block';
  updateCompletion();
}
function updateReminder(){
  const n=state.explored.size,b=document.getElementById('reminder-banner');
  if(n<CONCEPTS.length){b.style.display='block';
    document.getElementById('reminder-count').textContent=`(${n}/${CONCEPTS.length})`;}
  else b.style.display='none';
}

// ---- data load ----
async function loadData(){
  DATA=await (await fetch('/api/data')).json();
  VISITS_BY_PID={};
  DATA.visits.forEach(v=>{(VISITS_BY_PID[v[0]]=VISITS_BY_PID[v[0]]||[]).push(v);});
  Object.values(VISITS_BY_PID).forEach(a=>a.sort((x,y)=>x[1]-y[1]));
  document.getElementById('s-total').textContent=DATA.n_total;
  document.getElementById('s-pos').textContent=DATA.n_positive;
  document.getElementById('s-neg').textContent=DATA.n_negative;
  document.getElementById('s-visits').textContent=DATA.visits.length;
  renderPatientList(); renderHist(); drawSeqExplorer();
}
function renderHist(){
  const h=DATA.seq_len_hist, mx=Math.max(...h), tot=h.reduce((a,b)=>a+b,0);
  const rows=DATA.visits.length;
  document.getElementById('hist-bars').innerHTML=h.map((c,i)=>{
    const n=i+1, vw=n===1?'visit':'visits', pct=(c/tot*100).toFixed(0);
    return `<div class="hist-col"><div class="hist-bar" title="${c} of ${tot} patients have exactly ${n} ${vw} (${pct}%)" style="height:${Math.max(3,c/mx*130)}px">${c}</div>
     <div class="hist-lbl"><b style="color:#334155">${n} ${vw}</b><br>${c} patients<br><span style="color:#94a3b8">${pct}% of ${tot}</span></div></div>`;
  }).join('');
  const mode=h.indexOf(mx)+1, six=h[5], avg=(rows/tot).toFixed(1);
  const mvw=mode===1?'visit':'visits';
  document.getElementById('hist-note').innerHTML=
    `<p style="margin-bottom:10px">Each bar represents <b>one possible history length</b>, from
     <b>1 visit</b> to <b>6 visits</b>. The <b>height of the bar</b> shows the <b>number of
     patients</b> with exactly that many visits.</p>
     <p style="margin-bottom:10px">The most common history length is <b>${mode} ${mvw}</b> with
     <b>${mx} patients</b>. The average history length is about <b>${avg} visits</b>, and
     <b>${six} patient${six===1?'':'s'}</b> have the full <b>6 visits</b>. Altogether, the dataset
     contains <b>${tot} patients</b> and <b>${rows.toLocaleString()} visit rows</b>.</p>
     <p style="margin-bottom:10px">This matters because neural networks usually need
     <b>fixed-size inputs</b>. But patients have different numbers of visits. To handle this,
     shorter histories are <b>zero-padded</b>, and longer histories may be <b>truncated</b>.</p>
     <p style="margin-bottom:0">For example, if <code>max_seq_len = 6</code>, many patients with
     fewer than 6 visits will need <b>padding</b>. If <code>max_seq_len = 3</code>, less padding is
     needed, but patients with more than 3 visits will be <b>truncated</b>, usually
     <b>keeping the most recent visits</b>.</p>`;
}
function renderPatientList(){
  const q=(document.getElementById('tl-search').value||'').toLowerCase();
  const rows=DATA.patients.map((p,i)=>({p,i,label:DATA.labels[i],
      nv:(VISITS_BY_PID[String(p.id)]||[]).length}))
    .filter(r=>!q||String(r.p.id).toLowerCase().includes(q)||(r.p.condition_text||'').toLowerCase().includes(q));
  document.getElementById('tbl-badge').textContent=rows.length+' of '+DATA.patients.length;
  document.getElementById('tl-body').innerHTML=rows.map(r=>
    `<tr role="button" onclick="showTimeline('${r.p.id}')" id="prow-${r.p.id}">
      <td>${r.p.id}</td>
      <td>${r.label?'<span class="lbl-pos">pain=1</span>':'<span class="lbl-neg">pain=0</span>'}</td>
      <td>${r.nv}</td>
      <td>${(r.p.condition_text||'').slice(0,70)}</td></tr>`).join('')
    ||'<tr><td colspan="4" style="text-align:center;color:#94a3b8;padding:16px">No matches</td></tr>';
}
function showTimeline(pid){
  document.querySelectorAll('#tl-body tr').forEach(t=>t.classList.remove('sel'));
  const row=document.getElementById('prow-'+pid); if(row) row.classList.add('sel');
  document.getElementById('tl-pid').textContent=pid;
  const vs=VISITS_BY_PID[String(pid)]||[];
  if(!vs.length){document.getElementById('tl-detail').innerHTML='<div class="tl-empty">No visits recorded for this patient.</div>';return;}
  const marks=vs.map((v,i)=>{
    // even spacing by visit index so markers never overlap; the day label
    // still shows the real (uneven) timing.
    const left=6+ (vs.length>1? i/(vs.length-1)*88 : 44);
    return `<div class="tl-visit" style="left:${left}%">
      <div class="tl-day">day ${v[2]}</div>
      <div class="tl-dot ${v[3]>=7?'hi':''}"></div>
      <div class="tl-cap">v${v[1]}<br>pain ${v[3].toFixed(1)}<br>${v[4]} meds<br>${VTYPE[v[5]]}</div></div>`;
  }).join('');
  // sparkline of pain over visits -- stretched to fill the lower half
  const W=420,H=150,pad=16;
  const xs=vs.map((v,i)=>pad+(vs.length>1?i/(vs.length-1):0.5)*(W-2*pad));
  const ys=vs.map(v=>H-pad-(v[3]/10)*(H-2*pad));
  const path=xs.map((x,i)=>(i?'L':'M')+x.toFixed(1)+' '+ys[i].toFixed(1)).join(' ');
  const grid=[0,5,10].map(p=>{const y=(H-pad-(p/10)*(H-2*pad)).toFixed(1);
    return `<line x1="${pad}" y1="${y}" x2="${W-pad}" y2="${y}" stroke="#e2e8f0" stroke-width="1" vector-effect="non-scaling-stroke"/>`+
           `<text x="2" y="${(+y+3).toFixed(1)}" font-size="9" fill="#cbd5e1">${p}</text>`;}).join('');
  const dots=xs.map((x,i)=>`<circle cx="${x.toFixed(1)}" cy="${ys[i].toFixed(1)}" r="3.5" fill="#4f46e5" vector-effect="non-scaling-stroke"/>`).join('');
  const glabels=xs.map((x,i)=>`<text x="${x.toFixed(1)}" y="${Math.max(11,ys[i]-8).toFixed(1)}" font-size="10" fill="#64748b" text-anchor="middle">${vs[i][3].toFixed(1)}</text>`).join('');
  document.getElementById('tl-detail').innerHTML=
    `<div class="tl-top"><div class="tl-wrap"><div class="tl-line"><div class="tl-axis"></div>${marks}</div></div></div>
     <div class="tl-bot">
       <div class="tl-bot-lbl">pain score over visits (0&ndash;10)</div>
       <svg class="spark" viewBox="0 0 ${W} ${H}">
         <rect x="0" y="0" width="${W}" height="${H}" fill="#f8fafc" rx="6"/>${grid}
         <path d="${path}" stroke="#4f46e5" stroke-width="2" fill="none" vector-effect="non-scaling-stroke"/>${dots}${glabels}</svg>
     </div>`;
}

// ---- sequence explorer ----
function pickMaxSeq(n){
  state.maxSeqLen=n; state.seqTouched=true;
  document.querySelectorAll('#msl-seg button').forEach(b=>b.classList.toggle('on',+b.dataset.v===n));
  setArenaMslNote();
  drawSeqExplorer();
  if(state.seed) scheduleEval();
  updateCompletion();
}
function drawSeqExplorer(){
  if(!DATA) return;
  if(!state.seqTouched){
    document.getElementById('pad-frac').innerHTML='&ndash;';
    document.getElementById('pad-frac-lbl').textContent='pick a value above';
    document.getElementById('trunc-note').innerHTML='Choose a value above &mdash; the padding fraction, the truncation note and the per-patient preview below all update once you do.';
    document.getElementById('seq-rows').innerHTML='<div class="seq-hint">Choose a value above to preview padding &amp; truncation for the first 20 patients.</div>';
    return;
  }
  const L=state.maxSeqLen;
  document.getElementById('arena-msl').textContent=L;
  document.getElementById('pad-frac-lbl').textContent='of all visit-slots are padding';
  const ids=DATA.patients.map(p=>String(p.id));
  const CAP=DATA.max_visits||6;               // slot cells keep their width at the max
  let realSlots=0, truncCount=0;
  const rowsHtml=ids.slice(0,20).map(pid=>{
    const T=(VISITS_BY_PID[pid]||[]).length;
    let slots='';
    for(let t=0;t<L;t++) slots+= t<Math.min(T,L)
      ? '<div class="slot real">v'+(t+1 + Math.max(0,T-L))+'</div>'
      : '<div class="slot pad">pad</div>';
    for(let e=L;e<CAP;e++) slots+='<div class="slot slot-empty"></div>';
    const truncTxt = T>L ? '&#9888; '+(T-L)+' oldest dropped' : '';
    return `<div class="seq-row"><span class="sid">${pid}</span><span class="slots">${slots}</span><span class="trunc-col">${truncTxt}</span></div>`;
  }).join('');
  ids.forEach(pid=>{
    const T=(VISITS_BY_PID[pid]||[]).length;
    realSlots+=Math.min(T,L);
    if(T>L) truncCount++;
  });
  document.getElementById('seq-rows').innerHTML=rowsHtml;
  const frac=1-realSlots/(ids.length*L);
  document.getElementById('pad-frac').textContent=(frac*100).toFixed(0)+'%';
  const vw=L===1?'visit':'visits';
  document.getElementById('trunc-note').innerHTML= truncCount
    ? `<b>${truncCount}</b> patient${truncCount===1?'':'s'} have more than ${L} ${vw} and lose their oldest visits at this setting; the other ${ids.length-truncCount} get zero-padded up to ${L}.`
    : `No patient exceeds ${L} ${vw} &mdash; nothing is truncated, but ${(frac*100).toFixed(0)}% of all ${ids.length}&times;${L} slots are padding a mask has to ignore.`;
}

// ---- sidebar / assign ----
async function setStudentId(){
  const id=document.getElementById('sid-input').value.trim();
  if(!id){alert('Enter your student ID.');return;}
  const d=await (await fetch('/api/assign',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({student_id:id})})).json();
  if(d.error){alert('Error: '+d.error);return;}
  state.studentId=id; state.seed=d.seed; state.oracle=d.oracle;
  const o=d.oracle;
  document.getElementById('seed-disp').style.display='block';
  document.getElementById('seed-disp').textContent=`Seed ${d.seed} · train ${o.train_n} / val ${o.val_n}`;
  document.getElementById('oracle-s').style.display='block';
  document.getElementById('o-f1').textContent='>= '+o.f1.toFixed(3);
  document.getElementById('o-acc').textContent='>= '+(o.accuracy*100).toFixed(1)+'%';
  document.getElementById('o-auc').textContent='>= '+o.auc.toFixed(3);
  document.getElementById('not-set-msg').style.display='none';
  document.getElementById('metrics-panel').style.display='flex';
  runEval();
}

// ---- recurrent arena ----
function pickCell(v){
  state.cell=v;
  document.getElementById('ct-val').textContent=v;
  document.querySelectorAll('#ct-seg button').forEach(b=>b.classList.toggle('on',b.dataset.v===v));
  scheduleEval();
}
function toggleBidir(){
  state.bidir=!state.bidir;
  const b=document.getElementById('bd-tgl');
  b.classList.toggle('on',state.bidir); b.textContent=state.bidir?'on':'off';
  document.getElementById('bd-val').textContent=state.bidir?'on':'off';
  scheduleEval();
}
function scheduleEval(){clearTimeout(state.evalTimer);state.evalTimer=setTimeout(runEval,280);}
async function runEval(){
  if(!state.seed) return;
  if(!state.maxSeqLen){
    // max_seq_len has no default -- nothing runs until it is chosen.
    state.current=null; state.isOptimal=false;
    document.getElementById('metric-bars').innerHTML=
      '<p style="color:#b45309;font-size:.8rem;padding:10px 4px;text-align:center;line-height:1.5">'+
      'Choose a <b>max_seq_len</b> on the <b>Sequence Explorer</b> tab &mdash; training starts and '+
      'the Current vs Target bars appear here once you do.</p>';
    ['baseline-card','vis-panels','log-card'].forEach(id=>document.getElementById(id).style.display='none');
    document.getElementById('opt-banner').style.display='none';
    document.getElementById('hint-box').style.display='none';
    document.getElementById('arena-caption').style.display='none';
    ['o-f1','o-acc','o-auc'].forEach(id=>document.getElementById(id).classList.remove('hit'));
    updateCompletion();
    return;
  }
  const body={seed:state.seed,
    max_seq_len:state.maxSeqLen,
    hidden_units:+document.getElementById('hu-sl').value,
    cell_type:state.cell, bidirectional:state.bidir};
  const d=await (await fetch('/api/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)})).json();
  if(d.error){console.error(d.error);return;}
  state.current=d; state.isOptimal=d.is_optimal;
  document.getElementById('baseline-card').style.display='block';
  document.getElementById('vis-panels').style.display='grid';
  updateMetricBars(d.recurrent,state.oracle,d);
  updateBaseline(d.conv1d);
  drawLoss(d);
  drawROC(d.recurrent.roc_curve,d.recurrent.auc,d.conv1d.roc_curve,d.conv1d.auc);
  const r=d.recurrent;
  updateCM(r.tp,r.fp,r.tn,r.fn,r.accuracy);
  const cap=document.getElementById('arena-caption');
  cap.style.display='block'; cap.textContent=d.caption;
  logRow(body,d);
  updateCompletion();
}
function updateMetricBars(m,o,d){
  const bars=[{n:'F1',c:m.f1,t:o.f1},{n:'Accuracy',c:m.accuracy,t:o.accuracy,pct:true},{n:'AUC',c:m.auc,t:o.auc}];
  document.getElementById('metric-bars').innerHTML=bars.map(b=>{
    const ratio=Math.min(b.c/b.t,1.05), pct=Math.round(ratio*100);
    const cls=ratio>=0.995?'good':(ratio>=0.9?'warn':'');
    const cs=b.pct?(b.c*100).toFixed(1)+'%':b.c.toFixed(3);
    const ts=b.pct?(b.t*100).toFixed(1)+'%':b.t.toFixed(3);
    return `<div class="m-row"><div class="ml"><span class="mn">${b.n}</span>
      <span class="mv"><span class="cur">${cs}</span><span class="tgt">target &ge; ${ts}</span></span></div>
      <div class="prog-track"><div class="prog-fill ${cls}" style="width:${Math.min(pct,100)}%"></div></div></div>`;
  }).join('');
  const hb=document.getElementById('hint-box');
  if(d.hint&&!d.is_optimal){hb.style.display='block';hb.textContent='💡 '+d.hint;}else hb.style.display='none';
  const ob=document.getElementById('opt-banner');
  ob.style.display=d.is_optimal?'block':'none';
  if(d.is_optimal){
    document.getElementById('reveal-params').textContent=
      `max_seq_len=${d.max_seq_len}, hidden_units=${d.hidden_units}, cell_type="${d.cell_type}"`;
  }
  ['o-f1','o-acc','o-auc'].forEach(id=>document.getElementById(id).classList.toggle('hit',d.is_optimal));
}
function updateBaseline(c){
  document.getElementById('bl-metrics').innerHTML=
    `<span>AUC <b>${c.auc.toFixed(3)}</b></span><span>F1 <b>${c.f1.toFixed(3)}</b></span>`+
    `<span>Acc <b>${(c.accuracy*100).toFixed(1)}%</b></span>`;
}
function drawLoss(d){
  const cv=document.getElementById('loss-canvas'),x=cv.getContext('2d'),W=cv.width,H=cv.height,P=30;
  x.clearRect(0,0,W,H);
  const all=[...d.recurrent.train_loss,...d.recurrent.val_loss,...d.conv1d.train_loss,...d.conv1d.val_loss];
  const mn=Math.min(...all),mx=Math.max(...all),rng=mx-mn||1,E=d.epochs;
  const tx=i=>P+(W-2*P)*i/(E-1), ty=v=>H-P-(H-2*P)*(v-mn)/rng;
  x.strokeStyle='#e2e8f0';x.beginPath();x.moveTo(P,P);x.lineTo(P,H-P);x.lineTo(W-P,H-P);x.stroke();
  x.fillStyle='#94a3b8';x.font='9px sans-serif';
  x.fillText(mx.toFixed(2),3,P+4);x.fillText(mn.toFixed(2),3,H-P);
  x.textAlign='center';
  [1,10,20,E].forEach(e=>{const xx=tx(e-1);x.fillText(e,xx,H-P+13);});
  x.textAlign='left';x.fillText('epoch →',W-P-38,H-6);
  function line(arr,color,dash){x.strokeStyle=color;x.lineWidth=2;x.setLineDash(dash?[5,4]:[]);
    x.beginPath();arr.forEach((v,i)=>i?x.lineTo(tx(i),ty(v)):x.moveTo(tx(i),ty(v)));x.stroke();x.setLineDash([]);}
  line(d.recurrent.train_loss,'#4f46e5',false); line(d.recurrent.val_loss,'#4f46e5',true);
  line(d.conv1d.train_loss,'#f59e0b',false); line(d.conv1d.val_loss,'#f59e0b',true);
}
function drawROC(pts,auc,pts2,auc2){
  const cv=document.getElementById('roc-canvas'),x=cv.getContext('2d'),W=cv.width,H=cv.height,P=28;
  x.clearRect(0,0,W,H);
  const tx=f=>P+(W-2*P)*f, ty=t=>H-P-(H-2*P)*t;
  x.strokeStyle='#e2e8f0';x.beginPath();x.moveTo(P,P);x.lineTo(P,H-P);x.lineTo(W-P,H-P);x.stroke();
  x.strokeStyle='#cbd5e1';x.setLineDash([4,3]);x.beginPath();x.moveTo(P,H-P);x.lineTo(W-P,P);x.stroke();x.setLineDash([]);
  function curve(p,color,dash){if(!p||!p.length)return;x.strokeStyle=color;x.lineWidth=2;x.setLineDash(dash?[4,3]:[]);
    x.beginPath();p.forEach(([f,t],i)=>i?x.lineTo(tx(f),ty(t)):x.moveTo(tx(f),ty(t)));x.stroke();x.setLineDash([]);}
  curve(pts2,'#f59e0b',true); curve(pts,'#4f46e5',false);
  x.fillStyle='#4f46e5';x.font='bold 10px sans-serif';x.fillText('recurrent AUC '+auc.toFixed(3),P+4,P+12);
  x.fillStyle='#f59e0b';x.fillText('conv1d AUC '+auc2.toFixed(3),P+4,P+24);
  const o=state.oracle;
  document.getElementById('roc-caption').textContent=
    o?(auc>=o.auc-0.005?'✓ recurrent model at or above oracle target ('+o.auc.toFixed(3)+')'
       :'gap '+(o.auc-auc).toFixed(3)+' to oracle ('+o.auc.toFixed(3)+')'):'';
}
function updateCM(tp,fp,tn,fn,acc){
  document.getElementById('cm-tp').innerHTML=`<div class="cnt">${tp}</div><div class="lbl">TP pain caught</div>`;
  document.getElementById('cm-fn').innerHTML=`<div class="cnt">${fn}</div><div class="lbl">FN pain missed</div>`;
  document.getElementById('cm-fp').innerHTML=`<div class="cnt">${fp}</div><div class="lbl">FP false alarm</div>`;
  document.getElementById('cm-tn').innerHTML=`<div class="cnt">${tn}</div><div class="lbl">TN healthy cleared</div>`;
  const miss=Math.round(fn/Math.max(tp+fn,1)*100);
  document.getElementById('cm-caption').textContent=`Accuracy ${(acc*100).toFixed(1)}% · miss rate ${miss}% of real pain patients`;
}
function logRow(b,d){
  state.log.push({...b,auc:d.recurrent.auc,f1:d.recurrent.f1,opt:d.is_optimal});
  document.getElementById('log-card').style.display='block';
  document.getElementById('log-cnt').textContent=state.log.length;
  document.getElementById('log-body').innerHTML=[...state.log].reverse().map((e,i)=>
    `<tr><td>${state.log.length-i}</td><td>${e.max_seq_len}</td><td>${e.hidden_units}</td>
     <td>${e.cell_type}</td><td>${e.bidirectional?'yes':'no'}</td>
     <td>${e.auc.toFixed(3)}</td><td>${e.f1.toFixed(3)}</td>
     <td>${e.opt?'<span class="opt-y">OPTIMAL</span>':'<span class="opt-n">--</span>'}</td></tr>`).join('');
}
function copyParams(){
  const sid=state.studentId||'your_id';
  const s=`return {\n  "student_id":   "${sid}",\n  "max_seq_len":  ${state.maxSeqLen},\n  "hidden_units": ${document.getElementById('hu-sl').value},\n  "cell_type":    "${state.cell}",\n  "val_fraction": 0.20,\n}`;
  navigator.clipboard?.writeText(s).then(()=>alert('Copied:\n\n'+s)).catch(()=>alert(s));
}
function updateCompletion(){
  const done=state.isOptimal && state.explored.size>=CONCEPTS.length && state.seqTouched;
  document.getElementById('top-complete-bar').style.display=done?'flex':'none';
  document.body.classList.toggle('has-completion-bar',done);
}

// ---- init ----
renderConcepts(); loadData(); setArenaMslNote();
document.getElementById('sid-input').addEventListener('keydown',e=>{if(e.key==='Enter')setStudentId();});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    log.info("=" * 45)
    log.info("Assignment 3 -- AI-Sandbox Explorer")
    log.info("URL : http://localhost:3003")
    log.info("Log : %s", _LOG_FILE)
    log.info("=" * 45)
    threading.Timer(1.3, lambda: webbrowser.open("http://localhost:3003")).start()
    app.run(port=3003, debug=False, use_reloader=False)
