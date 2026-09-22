"""
CropGuardAI Training Dashboard - Streamlit GUI
Run: streamlit run scripts/training_dashboard.py
"""
import streamlit as st
import subprocess, sys, json, time, os
from pathlib import Path
from datetime import datetime
import threading, queue

# Page config
st.set_page_config(
    page_title="CropGuardAI Training Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
REPO = Path(os.environ.get("CROPGUARD_BASE", str(Path(__file__).resolve().parents[1])))
MODELS_DIR = REPO / "models"
LOGS_DIR = REPO / "logs"
OUTPUT_DIR = REPO / "training_outputs"
ORCHESTRATOR = REPO / "scripts" / "training_orchestrator.py"

LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Phase definitions (mirror orchestrator)
PHASES = [
    {"id": "0_leakproof", "name": "Leakproof Bootstrap", "icon": "shield", "required": True,
     "desc": "Zero-leakage pipeline: dedup + label clean + verified splits"},
    {"id": "1_swin_t", "name": "Phase 1: Swin-T Baseline", "icon": "🚀", "required": True,
     "desc": "Swin-T 60 epochs, EMA, MixUp+CutMix. Best single-model ceiling."},
    {"id": "1b_effnetv2s", "name": "Phase 1b: EfficientNetV2-S", "icon": "🔧", "required": False,
     "desc": "EffNetV2-S for ensemble diversity. Optional."},
    {"id": "2_leafvision_b0", "name": "Phase 2: LeafVision EffNet-B0", "icon": "🧠", "required": True,
     "desc": "Domain SSL init (EffNet-B0 DINO on 540k leaves). Expect +2-4%."},
    {"id": "2b_leafvision_resnet50", "name": "Phase 2b: LeafVision ResNet-50", "icon": "🧠", "required": True,
     "desc": "Domain SSL init (ResNet-50 DINO). Strongest backbone."},
    {"id": "3_wheat_stage", "name": "Phase 3: Wheat Stage", "icon": "🌱", "required": True,
     "desc": "Retrain stage model on GPU with Swin-T."},
    {"id": "3b_rice_stage", "name": "Phase 3b: Rice Stage", "icon": "🌾", "required": False,
     "desc": "Retrain rice stage model."},
    {"id": "4_rice", "name": "Phase 4: Rice Disease", "icon": "🍚", "required": False,
     "desc": "Retrain rice disease if needed."},
]

PHASE_MAP = {p["id"]: p for p in PHASES}

# Session state
if "running" not in st.session_state:
    st.session_state.running = False
if "current_phase" not in st.session_state:
    st.session_state.current_phase = None
if "log_queue" not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if "process" not in st.session_state:
    st.session_state.process = None
if "phase_results" not in st.session_state:
    st.session_state.phase_results = {}
if "start_time" not in st.session_state:
    st.session_state.start_time = None


def run_phase_bg(phase_id: str, resume: bool):
    """Background thread to run a single phase."""
    phase = PHASE_MAP[phase_id]
    script = REPO / "scripts" / "training_orchestrator.py"
    args = [sys.executable, str(script), "--phase", phase_id]
    if resume:
        args.append("--resume")

    log_file = LOGS_DIR / f"{phase_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    st.session_state.log_queue.put(("info", f"Starting {phase['name']}..."))
    st.session_state.log_queue.put(("info", f"Log: {log_file.name}"))

    try:
        with open(log_file, "w", encoding="utf-8", buffering=1) as lf:
            lf.write(f"Command: {' '.join(args)}\n")
            lf.write(f"Started: {datetime.now().isoformat()}\n\n")
            proc = subprocess.Popen(
                args,
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            st.session_state.process = proc
            for line in proc.stdout:
                st.session_state.log_queue.put(("output", line.rstrip()))
                lf.write(line)
            proc.wait()
            st.session_state.process = None
            if proc.returncode == 0:
                st.session_state.log_queue.put(("success", f"✅ {phase['name']} completed"))
                st.session_state.phase_results[phase_id] = {"success": True, "log": str(log_file)}
            else:
                st.session_state.log_queue.put(("error", f"❌ {phase['name']} failed (exit {proc.returncode})"))
                st.session_state.phase_results[phase_id] = {"success": False, "log": str(log_file), "error": f"Exit {proc.returncode}"}
    except Exception as e:
        st.session_state.log_queue.put(("error", f"❌ {phase['name']} error: {e}"))
        st.session_state.phase_results[phase_id] = {"success": False, "error": str(e)}
    finally:
        st.session_state.running = False
        st.session_state.current_phase = None


def run_all_bg(resume: bool, skip_optional: bool):
    """Background thread to run all phases sequentially."""
    for phase in PHASES:
        if skip_optional and not phase["required"]:
            st.session_state.log_queue.put(("info", f"⏭️ Skipping optional: {phase['name']}"))
            continue
        if st.session_state.phase_results.get(phase["id"], {}).get("success"):
            st.session_state.log_queue.put(("info", f"⏭️ Already done: {phase['name']}"))
            continue
        st.session_state.current_phase = phase["id"]
        st.session_state.running = True
        run_phase_bg(phase["id"], resume)
        # Wait for completion
        while st.session_state.running:
            time.sleep(1)
        if not st.session_state.phase_results.get(phase["id"], {}).get("success"):
            if phase["required"]:
                st.session_state.log_queue.put(("error", f"Required phase {phase['name']} failed. Stopping."))
                break
            else:
                st.session_state.log_queue.put(("warning", f"Optional phase {phase['name']} failed. Continuing..."))
    st.session_state.log_queue.put(("success", "🏁 All phases complete"))
    # Auto-package
    try:
        subprocess.run([sys.executable, str(ORCHESTRATOR), "--skip-optional" if skip_optional else ""],
                       cwd=REPO, capture_output=True, text=True, timeout=300)
        st.session_state.log_queue.put(("success", "📦 Output package created in training_outputs/"))
    except Exception as e:
        st.session_state.log_queue.put(("warning", f"Packaging failed: {e}"))


def stop_current():
    if st.session_state.process:
        st.session_state.process.terminate()
        st.session_state.process = None
    st.session_state.running = False
    st.session_state.current_phase = None
    st.session_state.log_queue.put(("warning", "🛑 Stopped by user"))


# --- UI ---
st.title("🌾 CropGuardAI Training Dashboard")
st.caption("One-click training for all phases · Real-time logs · Auto-packaging")

# Sidebar: Controls
with st.sidebar:
    st.header("Controls")
    resume = st.checkbox("Resume from checkpoints", value=True, help="Continue each phase from its last checkpoint")
    skip_optional = st.checkbox("Skip optional phases", value=False, help="Run only required phases")
    st.divider()

    if st.button("🛑 STOP CURRENT", type="secondary", use_container_width=True, disabled=not st.session_state.running):
        stop_current()
        st.rerun()

    if st.button("🧹 Clear Logs", use_container_width=True):
        while not st.session_state.log_queue.empty():
            st.session_state.log_queue.get()
        st.session_state.phase_results = {}
        st.rerun()

    st.divider()
    st.header("Quick Actions")
    if st.button("Leakproof Bootstrap", use_container_width=True, disabled=st.session_state.running):
        st.session_state.current_phase = "0_leakproof"
        st.session_state.running = True
        threading.Thread(target=run_phase_bg, args=("0_leakproof", False), daemon=True).start()
        st.rerun()

    if st.button("▶️ RUN ALL REQUIRED", type="primary", use_container_width=True, disabled=st.session_state.running):
        st.session_state.running = True
        st.session_state.phase_results = {}
        threading.Thread(target=run_all_bg, args=(resume, skip_optional), daemon=True).start()
        st.rerun()

    if st.button("▶️ RUN ALL (incl. optional)", use_container_width=True, disabled=st.session_state.running):
        st.session_state.running = True
        st.session_state.phase_results = {}
        threading.Thread(target=run_all_bg, args=(resume, False), daemon=True).start()
        st.rerun()

# Main: Phase grid + Live log
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Training Phases")
    for phase in PHASES:
        pid = phase["id"]
        result = st.session_state.phase_results.get(pid, {})
        status = "✅" if result.get("success") else ("🔄" if pid == st.session_state.current_phase else ("❌" if result.get("success") is False else "⏳"))
        req_badge = " 🔴" if phase["required"] else " ⚪"
        with st.container():
            cols = st.columns([4, 1])
            cols[0].markdown(f"**{phase['icon']} {phase['name']}**{req_badge}")
            cols[0].caption(phase["desc"])
            if cols[1].button("Run", key=f"run_{pid}", disabled=st.session_state.running or result.get("success"), use_container_width=True):
                st.session_state.current_phase = pid
                st.session_state.running = True
                threading.Thread(target=run_phase_bg, args=(pid, resume), daemon=True).start()
                st.rerun()
            if pid == st.session_state.current_phase:
                st.progress(0.5, text="Running...")
            elif result.get("success"):
                st.success("Done")
            elif result.get("success") is False:
                st.error(f"Failed: {result.get('error', 'unknown')}")

with col2:
    st.subheader("Live Log")
    log_container = st.container(height=600)
    # Drain queue
    while not st.session_state.log_queue.empty():
        level, msg = st.session_state.log_queue.get()
        if "logs" not in st.session_state:
            st.session_state.logs = []
        st.session_state.logs.append((level, msg))
    # Keep last 500 lines
    if "logs" in st.session_state and len(st.session_state.logs) > 500:
        st.session_state.logs = st.session_state.logs[-500:]
    # Display
    with log_container:
        if "logs" in st.session_state and st.session_state.logs:
            for level, msg in st.session_state.logs:
                if level == "error":
                    st.error(msg)
                elif level == "success":
                    st.success(msg)
                elif level == "warning":
                    st.warning(msg)
                elif level == "info":
                    st.info(msg)
                else:
                    st.code(msg, language=None)
        else:
            st.caption("Logs appear here when training runs...")

# Auto-refresh while running
if st.session_state.running:
    time.sleep(1)
    st.rerun()

# Footer: Model status
st.divider()
st.subheader("Model Artifacts")
models = list(MODELS_DIR.glob("*.pt")) if MODELS_DIR.exists() else []
if models:
    for m in sorted(models):
        size = m.stat().st_size / 1e6
        mtime = datetime.fromtimestamp(m.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        st.text(f"  {m.name}  ({size:.1f} MB)  {mtime}")
else:
    st.caption("No models yet")

# Output packages
st.subheader("Output Packages")
packages = list(OUTPUT_DIR.glob("package_*")) if OUTPUT_DIR.exists() else []
if packages:
    for p in sorted(packages, key=lambda x: x.stat().st_mtime, reverse=True):
        manifest = p / "manifest.json"
        if manifest.exists():
            with open(manifest) as f:
                mf = json.load(f)
            st.text(f"  {p.name}  —  {len(mf.get('collected', {}).get('models', []))} models  {mf.get('created', '')[:19]}")
        else:
            st.text(f"  {p.name}")
else:
    st.caption("No packages yet. Run training to generate.")