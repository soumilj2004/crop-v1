"""Generate a labeled HTML gallery of the full dataset.

Scans data/split/<crop>/{train,val,test}/<class> images, joins them with the
HSV stage labels (early/mid/late), writes thumbnails and a clean self-contained
index.html with class + stage badges, filters, and a per-class stats table.
"""
import json, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE = Path(os.environ.get("CROPGUARD_BASE", str(Path(__file__).resolve().parents[1])))
DATA = BASE / "data"
GALLERY = DATA / "gallery"
THUMBS = GALLERY / "thumbs"
THUMB_SIZE = (224, 224)
THUMB_Q = 78

CROP_NAMES = {"wheat": "Wheat", "rice": "Rice"}
STAGE_COLORS = {"early": "#2e9e44", "mid": "#e0912a", "late": "#d83c3c"}

IMAGES = {'jpg', 'jpeg', 'png', 'bmp'}


def thumb_src(rel: str) -> str:
    """Map a split-relative path like train/Healthy/x.jpg to thumbs/....jpg"""
    p = Path(rel)
    return "thumbs/" + str(p.with_suffix(".jpg")).replace("\\", "/")


def make_thumb(rel: Path, src: Path) -> bool:
    out = THUMBS / rel.with_suffix(".jpg")
    if out.exists():
        return True
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(src).convert("RGB")
        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
        img.save(out, "JPEG", quality=THUMB_Q, optimize=True)
        return True
    except Exception:
        return False


def main():
    GALLERY.mkdir(parents=True, exist_ok=True)
    THUMBS.mkdir(parents=True, exist_ok=True)

    crops = ["wheat", "rice"]
    stages = {}
    for crop in crops:
        sp = DATA / "stage_labels" / f"{crop}_hsv_stage_labels.json"
        if sp.exists():
            stages[crop] = json.load(open(sp))

    items = []          # (crop, split, cls, rel, abs_src)
    per_class = {}      # (crop, cls) -> total
    stage_tab = {}      # (crop, cls, stage) -> count

    for crop in crops:
        split_dir = DATA / "split" / crop
        for split in ["train", "val", "test"]:
            sp = split_dir / split
            if not sp.exists():
                continue
            for cls_dir in sorted(sp.iterdir()):
                if not cls_dir.is_dir():
                    continue
                cls = cls_dir.name
                for f in sorted(cls_dir.iterdir()):
                    if not f.is_file() or f.suffix.lower().lstrip('.') not in IMAGES:
                        continue
                    rel = str(f.relative_to(split_dir)).replace("\\", "/")
                    per_class[(crop, cls)] = per_class.get((crop, cls), 0) + 1
                    stag = (stages.get(crop) or {}).get(rel)
                    stage_tab[(crop, cls, stag or "none")] = stage_tab.get((crop, cls, stag or "none"), 0) + 1
                    items.append((crop, split, cls, rel, f))

    print(f"collected {len(items)} images", flush=True)

    t0 = time.time()
    done = [0]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(make_thumb, Path(rel), src) for (_, _, _, rel, src) in items]
        for i, fut in enumerate(futs):
            fut.result()
            done[0] = i + 1
            if done[0] % 1000 == 0:
                print(f"  thumbs {done[0]}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"thumbs done in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for (crop, split, cls, rel, src) in items:
        stag = (stages.get(crop) or {}).get(rel)
        badge = f'<span class="b-stage s-{stag}">{stag}</span>' if stag else ''
        rows.append(
            f'<figure data-crop="{crop}" data-cls="{cls}" data-stage="{stag or "none"}" data-split="{split}">'
            f'<a href="../split/{rel}" target="_blank"><img src="{thumb_src(rel)}" loading="lazy" alt="{cls}"></a>'
            f'<figcaption><span class="b-cls">{cls}</span>{badge}<span class="b-split">{split}</span></figcaption>'
            f'</figure>'
        )

    # stats table
    stat_rows = []
    for (crop, cls), total in sorted(per_class.items()):
        s = stage_tab
        e = s.get((crop, cls, "early"), 0)
        m = s.get((crop, cls, "mid"), 0)
        l = s.get((crop, cls, "late"), 0)
        n = s.get((crop, cls, "none"), 0)
        stat_rows.append(
            f'<tr><td>{CROP_NAMES[crop]}</td><td class="c">{cls}</td><td>{total}</td>'
            f'<td><span class="b-stage s-early">{e}</span></td>'
            f'<td><span class="b-stage s-mid">{m}</span></td>'
            f'<td><span class="b-stage s-late">{l}</span></td>'
            f'<td>{n}</td></tr>'
        )

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CropGuard Dataset Gallery</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--txt:#e8ecf3;--dim:#8b93a3}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;padding:0 18px 60px}
h1{font-size:22px;margin:22px 0 4px}
.sub{color:var(--dim);margin:0 0 18px}
.head{position:sticky;top:0;background:rgba(15,17,21,.96);padding:12px 0 10px;z-index:10;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input{background:var(--panel);border:1px solid var(--line);color:var(--txt);border-radius:8px;padding:8px 12px;font-size:14px;width:260px}
.node{background:var(--panel);border:1px solid var(--line);color:var(--txt);border-radius:8px;padding:7px 14px;font-size:13px;cursor:pointer}
.node.active{background:#3b82f6;border-color:#3b82f6}
table{border-collapse:collapse;margin:18px 0 26px;font-size:13px}
th,td{border:1px solid var(--line);padding:7px 14px;text-align:left}
th{background:var(--panel);color:var(--dim);font-weight:600}
.grp{margin:10px 0 5px;font-size:16px;border-left:4px solid #3b82f6;padding-left:10px}
.grp small{color:var(--dim);font-weight:400}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px;margin:12px 0 34px}
figure{margin:0;position:relative}
figure img{width:100%;height:128px;object-fit:cover;border-radius:6px;display:block;background:#000}
figcaption{display:flex;gap:4px;flex-wrap:wrap;margin-top:5px;font-size:11px;align-items:center}
.b-cls{background:#3b82f6;color:#fff;padding:2px 7px;border-radius:4px}
.b-split{background:#3a4150;color:#cdd5e2;padding:2px 6px;border-radius:4px}
.b-stage{padding:2px 6px;border-radius:4px;color:#fff}
.s-early{background:#2e9e44}.s-mid{background:#e0912a}.s-late{background:#d83c3c}.s-none{background:#555d6b}
.hide{display:none!important}
#count{color:var(--dim);font-size:13px}
</style>
</head>
<body>
<div class="head">
<input id="q" type="search" placeholder="Search disease / stage / split..." autocomplete="off">
<button class="node" data-f="crop,wheat" onclick="setFilter('crop','wheat',this)">Wheat</button>
<button class="node" data-f="crop,rice" onclick="setFilter('crop','rice',this)">Rice</button>
<button class="node" onclick="clearFilter()">All</button>
<span id="count"></span>
</div>
<h1>CropGuard — Full Dataset Gallery</h1>
<p class="sub">All images from <code>data/split/&lt;crop&gt;/&lt;train|val|test&gt;/&lt;class&gt;/</code> — labeled by disease class and HSV-derived severity stage (early / mid / late). Click any image to open the original. Stage label source: <code>data/stage_labels/&lt;crop&gt;_hsv_stage_labels.json</code>.</p>
<table>
<tr><th>Crop</th><th>Disease / Class</th><th>Total</th><th style="color:#2e9e44">early</th><th style="color:#e0912a">mid</th><th style="color:#d83c3c">late</th><th>unlabeled</th></tr>
__STAT_ROWS__
</table>
__GROUPS__
<script>
const q=document.getElementById('q');
const figs=()=>document.querySelectorAll('figure');
let fKey=null;
function setFilter(k,v,btn){fKey=[k,v];document.querySelectorAll('.node.active').forEach(b=>b.classList.remove('active'));if(btn)btn.classList.add('active');apply();}
function clearFilter(){fKey=null;document.querySelectorAll('.node.active').forEach(b=>b.classList.remove('active'));apply();}
function apply(){
  let shown=0;
  figs().forEach(f=>{
    let ok=true;
    if(fKey&&f.dataset[fKey[0]]!==fKey[1])ok=false;
    const s=q.value.toLowerCase();
    if(ok&&s&&!f.dataset.cls.toLowerCase().includes(s)&&!f.dataset.stage.toLowerCase().includes(s)&&!f.dataset.split.toLowerCase().includes(s))ok=false;
    f.classList.toggle('hide',!ok);
    if(ok)shown++;
  });
  document.getElementById('count').textContent=shown+' shown';
}
q.addEventListener('input',apply);
apply();
</script>
</body>
</html>"""

    groups = []
    for crop in crops:
        classes = sorted({c for (crop2, c) in per_class if crop2 == crop})
        for cls in classes:
            total = per_class[(crop, cls)]
            group_n = "g-" + crop + "-" + cls.replace(" & ", "_").replace(" ", "_").lower()
            figures = [r for r in rows if (r.split('data-crop="')[1].split('"')[0] == crop and r.split('data-cls="')[1].split('"')[0] == cls)]
            groups.append(
                f'<div class="grp" id="{group_n}">{CROP_NAMES[crop]} — {cls} <small>({total} images)</small></div>\n'
                f'<div class="grid">{"".join(figures)}</div>'
            )
    html = html.replace("__STAT_ROWS__", "\n".join(stat_rows)).replace("__GROUPS__", "\n".join(groups))

    out = GALLERY / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"WROTE {out}  ({out.stat().st_size/1e6:.1f} MB, {len(items)} images)", flush=True)


if __name__ == "__main__":
    main()