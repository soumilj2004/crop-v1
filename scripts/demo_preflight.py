#!/usr/bin/env python3
"""
CropGuard AI - demo pre-flight and offline backup builder.

Two jobs, both driven by REAL output from the running /analyze endpoint:

  1. PRE-FLIGHT  Post every image in a folder to the live API and report what the
                 models actually predict, so you know before the review which
                 images are safe to show. Nothing is faked or overridden.

  2. BACKUP      Emit a single self-contained HTML file replaying those real
                 captured runs, for use if the laptop or the API dies mid-review.
                 The page is permanently and visibly labelled as a recording.

Folder layout expected (the parent folder name is taken as the expected class):

    demo v2/
      rice/Bacterial Blight/*.jpg
      wheat/Leaf Rust/*.jpg
    ...or simply:
      Bacterial Blight/*.jpg

Usage (with the API already running):

    python scripts/demo_preflight.py --images "C:\\CropGuardAI\\demo v2"
    python scripts/demo_preflight.py --images "C:\\CropGuardAI\\demo v2" --report-only
    python scripts/demo_preflight.py --images "..." --only-matches   # backup page keeps only correct predictions
"""

import argparse
import base64
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests is not installed.  pip install requests")

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is not installed.  pip install pillow")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------- collection

def collect(root: Path):
    """Return [(path, expected_class_or_None)] sorted by class then name."""
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            expected = p.parent.name if p.parent != root else None
            out.append((p, expected))
    return out


def norm(s):
    """Loose class-name comparison: 'Crown_and_Root_Rot' == 'Crown & Root Rot'."""
    if not s:
        return ""
    s = s.lower().replace("&", "and").replace("_", " ").replace("-", " ")
    return " ".join(s.split())


# ---------------------------------------------------------------- API calls

def analyze(api: str, path: Path, timeout: int):
    with open(path, "rb") as fh:
        files = {"file": (path.name, fh, "image/jpeg")}
        r = requests.post(api.rstrip("/") + "/analyze", files=files, timeout=timeout)
    r.raise_for_status()
    return r.json()


def thumb_data_uri(path: Path, box: int) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((box, box))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=78, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------- reporting

def report(rows):
    print()
    print("  " + "-" * 96)
    print(f"  {'FILE':<34}{'EXPECTED':<20}{'PREDICTED':<20}{'CONF':>7}  {'CROP':<7} {'VERDICT'}")
    print("  " + "-" * 96)
    per_class = {}
    for r in rows:
        if r.get("error"):
            print(f"  {r['name'][:33]:<34}{(r['expected'] or '-')[:19]:<20}{'ERROR':<20}{'':>7}  {'':<7} {r['error'][:28]}")
            continue
        d = r["resp"]
        conf = d.get("disease_confidence") or 0
        pred = d.get("disease", "?")
        exp = r["expected"]
        if exp is None:
            verdict = "no expected label"
        elif norm(exp) == norm(pred):
            verdict = "MATCH"
        else:
            verdict = "*** MISMATCH ***"
        if exp is not None:
            k = exp
            per_class.setdefault(k, [0, 0])
            per_class[k][1] += 1
            if verdict == "MATCH":
                per_class[k][0] += 1
        print(f"  {r['name'][:33]:<34}{(exp or '-')[:19]:<20}{pred[:19]:<20}{conf*100:>6.1f}%  "
              f"{str(d.get('detected_crop', '?'))[:6]:<7} {verdict}")
    print("  " + "-" * 96)
    if per_class:
        print("\n  Per-class summary (use only classes that pass; a class at 0/5 needs different images):\n")
        for k in sorted(per_class):
            ok, tot = per_class[k]
            flag = "" if ok == tot else ("   <-- WEAK" if ok else "   <-- ALL FAIL")
            print(f"    {k:<26}{ok}/{tot}{flag}")
    print()


# ---------------------------------------------------------------- backup page

BANNER_NOTE = ("This page is a RECORDING of real runs captured from the live system. "
               "It is not performing inference. Use it only if the live demo cannot run, "
               "and say so when you show it.")

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#fbf9f3;color:#1a1a1a;
 font:14px/1.55 "Public Sans",-apple-system,Segoe UI,Roboto,Arial,sans-serif}
.rec{position:sticky;top:0;z-index:9;background:#7a4a06;color:#fff;padding:10px 18px;
 font-size:12.5px;letter-spacing:.3px;display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
.rec b{font-size:13px;letter-spacing:1.2px;text-transform:uppercase;background:#fff;color:#7a4a06;
 padding:2px 8px;border-radius:4px}
.rec span{opacity:.92}
.wrap{max-width:1180px;margin:0 auto;padding:22px 18px 60px}
h1{font-family:Newsreader,Georgia,serif;font-weight:500;font-size:27px;margin:6px 0 2px}
.sub{color:#6b6a63;font-size:13px;margin-bottom:20px}
.cols{display:grid;grid-template-columns:212px 1fr;gap:22px;align-items:start}
.strip{display:flex;flex-direction:column;gap:9px;max-height:76vh;overflow:auto;padding-right:4px}
.th{display:flex;gap:9px;align-items:center;padding:6px;border:1px solid #e6e3d8;border-radius:9px;
 background:#fff;cursor:pointer}
.th:hover{border-color:#c9c5b4}
.th.on{border-color:#2f4a34;box-shadow:0 0 0 2px rgba(47,74,52,.14)}
.th img{width:52px;height:52px;object-fit:cover;border-radius:6px;flex:none}
.th .m{min-width:0}
.th .c{font-weight:600;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.th .f{font-size:10.5px;color:#8b8980;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.panel{background:#fff;border:1px solid #e6e3d8;border-radius:14px;padding:20px 22px}
.hero{display:flex;gap:20px;align-items:flex-start;margin-bottom:18px}
.hero img{width:190px;height:190px;object-fit:cover;border-radius:11px;flex:none}
.dz{font-family:Newsreader,Georgia,serif;font-size:27px;font-weight:500;line-height:1.15}
.pth{color:#6b6a63;font-size:12.5px;font-style:italic;margin-top:2px}
.pill{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;
 padding:3px 9px;border-radius:20px;margin-top:9px}
.crop{background:#eef3ee;color:#2f4a34;margin-right:6px}
.LOW{background:#e8f5e9;color:#1b6b21}.MODERATE{background:#fff6e0;color:#8a5d0b}
.HIGH{background:#fdeee8;color:#9c3a1c}.CRITICAL{background:#fbe3e0;color:#8f2419}
h3{font-size:11px;letter-spacing:1.1px;text-transform:uppercase;color:#8b8980;
 margin:20px 0 8px;font-weight:700}
.bars{display:flex;flex-direction:column;gap:5px}
.bar{display:grid;grid-template-columns:132px 1fr 46px;gap:9px;align-items:center;font-size:12.5px}
.bar .t{height:7px;background:#eeece2;border-radius:4px;overflow:hidden}
.bar .t i{display:block;height:100%;background:#2f4a34;border-radius:4px}
.bar .v{text-align:right;color:#6b6a63;font-variant-numeric:tabular-nums}
.bar.top .v,.bar.top>span:first-child{color:#1a1a1a;font-weight:600}
.wx{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.wxc{background:#f7f6f0;border-radius:9px;padding:10px;text-align:center}
.wxc .v{font-size:19px;font-weight:600}.wxc .l{font-size:10.5px;color:#8b8980;margin-top:2px}
.wxs{margin-top:8px;font-size:10.5px;color:#8b8980;text-align:right}
.wxs.cached{color:#8a5d0b;background:#fdf6e6;border:1px solid #efdfba;border-radius:7px;
 padding:6px 9px;text-align:left;font-weight:600}
table.tx{width:100%;border-collapse:collapse;font-size:13px}
table.tx td{padding:7px 0;border-bottom:1px solid #f0eee5;vertical-align:top}
table.tx td:first-child{width:104px;color:#8b8980;font-size:11.5px;text-transform:uppercase;
 letter-spacing:.6px;padding-right:12px}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:#2f4a34}
.none{color:#a9a79d;font-style:italic}
ul.cv{margin:0;padding-left:17px;color:#6b6a63;font-size:12px}
ul.cv li{margin-bottom:3px}
.cap{margin-top:22px;padding-top:12px;border-top:1px solid #eeece2;font-size:11px;color:#8b8980}
@media(max-width:820px){.cols{grid-template-columns:1fr}.strip{flex-direction:row;max-height:none}}
"""

JS = """
var D=__DATA__;
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function pct(x){return (x==null||isNaN(x))?'--':(Number(x)*100).toFixed(1)+'%';}
function bars(o){
  if(!o) return '<div class="none">Not returned.</div>';
  var ks=Object.keys(o).sort(function(a,b){return o[b]-o[a];});
  return '<div class="bars">'+ks.map(function(k,i){
    return '<div class="bar'+(i===0?' top':'')+'"><span>'+esc(k)+'</span>'+
      '<span class="t"><i style="width:'+(Math.max(0,Math.min(1,o[k]))*100).toFixed(1)+'%"></i></span>'+
      '<span class="v">'+pct(o[k])+'</span></div>';}).join('')+'</div>';
}
function wx(w){
  if(!w) return '<div class="none">No weather captured for this run.</div>';
  function n(x,dp){return (x==null||isNaN(x))?'--':Number(x).toFixed(dp||0);}
  function c(v,u,l){return '<div class="wxc"><div class="v">'+v+u+'</div><div class="l">'+l+'</div></div>';}
  return '<div class="wx">'+c(n(w.temperature_c),'&deg;','Temp')+c(n(w.humidity_pct),'%','Humidity')+
    c(n(w.rainfall_mm,1),'mm','Rain')+c(n(w.wind_kph),'','Wind kph')+'</div>'+
    (w.source?'<div class="wxs'+(/cached/i.test(w.source)?' cached':'')+'">'+esc(w.source)+'</div>':'');
}
function tx(t){
  if(!t) return '<div class="none">No treatment plan returned.</div>';
  var rows=[['Action',t.action,0],['Chemical',t.chemical,1],['Dosage',t.dosage,1],
            ['Timing',t.timing,0],['Cultural',t.cultural,0]];
  return '<table class="tx">'+rows.map(function(r){
    var v=r[1]?('<span class="'+(r[2]?'mono':'')+'">'+esc(r[1])+'</span>')
              :'<span class="none">not applicable for this disease</span>';
    return '<tr><td>'+r[0]+'</td><td>'+v+'</td></tr>';}).join('')+'</table>';
}
function show(i){
  var r=D.runs[i], d=r.resp;
  document.querySelectorAll('.th').forEach(function(e,j){e.classList.toggle('on',j===i);});
  var h='<div class="hero"><img src="'+r.thumb+'" alt=""><div>'+
    '<div class="dz">'+esc(d.disease)+'</div>'+
    (d.pathogen?'<div class="pth">'+esc(d.pathogen)+'</div>':'')+
    '<div><span class="pill crop">'+esc(d.detected_crop)+' &middot; '+pct(d.crop_confidence)+'</span>'+
    '<span class="pill '+esc(d.risk_level)+'">'+esc(d.risk_level)+' risk</span></div>'+
    '<div style="margin-top:9px;font-size:12.5px;color:#6b6a63">Confidence '+pct(d.disease_confidence)+
    (d.stage?' &middot; stage <b>'+esc(d.stage)+'</b>'+(d.stage_confidence?' ('+pct(d.stage_confidence)+')':''):'')+
    '</div>'+
    '<div style="margin-top:6px;font-size:11.5px;color:#8b8980">'+esc(r.name)+
    (r.expected?' &middot; folder label: '+esc(r.expected):'')+'</div>'+
    '</div></div>';
  h+='<h3>Disease probabilities</h3>'+bars(d.disease_probs);
  h+='<h3>Field conditions at capture time</h3>'+wx(d.weather);
  if(d.risk_reason) h+='<h3>Why this risk level</h3><div>'+esc(d.risk_reason)+'</div>';
  h+='<h3>Treatment plan</h3>'+tx(d.treatment);
  if(d.caveats&&d.caveats.length)
    h+='<h3>Caveats shown to the user</h3><ul class="cv">'+
       d.caveats.map(function(c){return '<li>'+esc(c)+'</li>';}).join('')+'</ul>';
  h+='<div class="cap">Captured '+esc(D.captured_at)+' from '+esc(D.api)+
     ' &middot; replayed verbatim, no inference performed by this page.</div>';
  document.getElementById('panel').innerHTML=h;
}
document.getElementById('strip').innerHTML=D.runs.map(function(r,i){
  return '<div class="th" onclick="show('+i+')"><img src="'+r.thumb+'" alt="">'+
    '<div class="m"><div class="c">'+esc(r.resp.disease)+'</div>'+
    '<div class="f">'+esc(r.name)+'</div></div></div>';}).join('');
show(0);
"""


def build_page(runs, api, out: Path):
    payload = {
        "captured_at": datetime.now().strftime("%d %b %Y, %H:%M"),
        "api": api,
        "runs": [{"name": r["name"], "expected": r["expected"],
                  "thumb": r["thumb"], "resp": r["resp"]} for r in runs],
    }
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>CropGuard AI - recorded demo walkthrough</title>"
        "<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link href='https://fonts.googleapis.com/css2?family=Newsreader:wght@400;500&"
        "family=Public+Sans:wght@400;600;700&display=swap' rel='stylesheet'>"
        "<style>" + CSS + "</style></head><body>"
        "<div class='rec'><b>Recorded</b><span>" + BANNER_NOTE + "</span></div>"
        "<div class='wrap'><h1>CropGuard AI &mdash; recorded walkthrough</h1>"
        "<div class='sub'>Captured " + payload["captured_at"] +
        " from the live <code>/analyze</code> endpoint. Offline fallback only.</div>"
        "<div class='cols'><div class='strip' id='strip'></div>"
        "<div class='panel' id='panel'></div></div></div>"
        "<script>" + JS.replace("__DATA__", json.dumps(payload)) + "</script>"
        "</body></html>"
    )
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="CropGuard AI demo pre-flight + offline backup builder")
    ap.add_argument("--images", required=True, help="folder of demo images (searched recursively)")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="running API base URL")
    ap.add_argument("--out", default=None, help="output HTML (default: <images>/backup_demo.html)")
    ap.add_argument("--report-only", action="store_true", help="print the table, write no backup page")
    ap.add_argument("--only-matches", action="store_true",
                    help="backup page includes only images whose prediction matches the folder label")
    ap.add_argument("--thumb", type=int, default=560, help="max thumbnail edge in the backup page")
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()

    root = Path(args.images)
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")

    try:
        h = requests.get(args.api.rstrip("/") + "/health", timeout=10).json()
        print(f"API OK at {args.api}   models: {h.get('models', h.get('status'))}")
    except Exception as e:
        sys.exit(f"Cannot reach the API at {args.api} - start it first.\n  {e}")

    items = collect(root)
    if not items:
        sys.exit(f"No images found under {root}")
    print(f"Found {len(items)} images under {root}\n")

    runs = []
    for i, (path, expected) in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {path.name} ... ", end="", flush=True)
        rec = {"name": path.name, "expected": expected, "path": str(path)}
        try:
            rec["resp"] = analyze(args.api, path, args.timeout)
            print(f"{rec['resp'].get('disease')} ({rec['resp'].get('disease_confidence', 0)*100:.1f}%)")
        except Exception as e:
            rec["error"] = str(e)
            print(f"ERROR {e}")
        runs.append(rec)
        time.sleep(0.05)

    report(runs)

    if args.report_only:
        print("  --report-only: no backup page written.\n")
        return

    good = [r for r in runs if not r.get("error")]
    if args.only_matches:
        good = [r for r in good if r["expected"] and norm(r["expected"]) == norm(r["resp"]["disease"])]
    if not good:
        sys.exit("Nothing to put in the backup page.")

    print(f"  Embedding {len(good)} runs into the backup page ...")
    for r in good:
        r["thumb"] = thumb_data_uri(Path(r["path"]), args.thumb)

    out = Path(args.out) if args.out else root / "backup_demo.html"
    build_page(good, args.api, out)
    mb = out.stat().st_size / 1e6
    print(f"\n  Backup page written: {out}  ({mb:.1f} MB, self-contained)")
    print("  Double-click to open. Works with no server, no network, no models.\n")


if __name__ == "__main__":
    main()
