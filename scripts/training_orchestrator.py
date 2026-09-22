"""
CropGuardAI Training Orchestrator
Runs all training phases sequentially, handles resume, produces packaged outputs.
"""
import json, os, sys, subprocess, time, shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

REPO = Path(os.environ.get("CROPGUARD_BASE", str(Path(__file__).resolve().parents[1])))
MODELS_DIR = REPO / "models"
DATA_DIR = REPO / "data"
LOGS_DIR = REPO / "logs"
OUTPUT_DIR = REPO / "training_outputs"
LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_okphase=True)

PHASES = [
    {
        "id": "0_leakproof",
        "name": "Leakproof Bootstrap",
        "description": "Download base wheat data, create splits, merge LWDCD2020",
        "script": "scripts/leakproof_bootstrap.py",
        "args": [],
        "required": True,
    },
    {
        "id": "1_swin_t",
        "name": "Phase 1: Swin-T Baseline",
        "description": "Swin Transformer Tiny, 60 epochs, EMA, MixUp+CutMix",
        "script": "scripts/train_efficientnet.py",
        "args": [
            "--model", "wheat",
            "--arch", "swin_t",
            "--epochs", "60",
            "--phase1_epochs", "10",
            "--phase2_lr", "2e-5",
            "--img-size", "224",
            "--aug", "both",
            "--ema", "0.999",
        ],
        "required": True,
        "output_model": "wheat_efficientnet_best.pt",
        "output_history": "wheat_efficientnet_history.json",
    },
    {
        "id": "1b_effnetv2s",
        "name": "Phase 1b: EfficientNetV2-S (optional)",
        "description": "EfficientNetV2-S baseline for ensemble diversity",
        "script": "scripts/train_efficientnet.py",
        "args": [
            "--model", "wheat",
            "--arch", "efficientnet_v2_s",
            "--epochs", "60",
            "--phase1_epochs", "10",
            "--phase2_lr", "2e-5",
            "--img-size", "224",
            "--aug", "both",
            "--ema", "0.999",
        ],
        "required": False,
        "output_model": "wheat_efficientnet_v2_s_best.pt",
        "output_history": "wheat_efficientnet_v2_s_history.json",
    },
    {
        "id": "2_leafvision_b0",
        "name": "Phase 2: LeafVision DINO EffNet-B0",
        "description": "Domain-specific SSL init (EffNet-B0)",
        "script": "scripts/train_efficientnet.py",
        "args": [
            "--model", "wheat",
            "--arch", "efficientnet_b0",
            "--epochs", "60",
            "--phase1_epochs", "10",
            "--phase2_lr", "2e-5",
            "--img-size", "224",
            "--aug", "both",
            "--ema", "0.999",
            "--init", "leafvision_dino_efficientnet_b0",
        ],
        "required": True,
        "output_model": "wheat_efficientnet_b0_leafvision_best.pt",
        "output_history": "wheat_efficientnet_b0_leafvision_history.json",
    },
    {
        "id": "2b_leafvision_resnet50",
        "name": "Phase 2b: LeafVision DINO ResNet-50",
        "description": "Domain-specific SSL init (ResNet-50, strongest backbone)",
        "script": "scripts/train_efficientnet.py",
        "args": [
            "--model", "wheat",
            "--arch", "resnet50",
            "--epochs", "60",
            "--phase1_epochs", "10",
            "--phase2_lr", "2e-5",
            "--img-size", "224",
            "--aug", "both",
            "--ema", "0.999",
            "--init", "leafvision_dino_resnet50",
        ],
        "required": True,
        "output_model": "wheat_resnet50_leafvision_best.pt",
        "output_history": "wheat_resnet50_leafvision_history.json",
    },
    {
        "id": "3_wheat_stage",
        "name": "Phase 3: Wheat Stage Model",
        "description": "Retrain stage model on GPU with Swin-T",
        "script": "scripts/train_efficientnet.py",
        "args": [
            "--model", "wheat_stage",
            "--arch", "swin_t",
            "--epochs", "60",
            "--phase1_epochs", "10",
            "--phase2_lr", "2e-5",
            "--img-size", "224",
            "--aug", "both",
            "--ema", "0.999",
        ],
        "required": True,
        "output_model": "wheat_stage_efficientnet_best.pt",
        "output_history": "wheat_stage_efficientnet_history.json",
    },
    {
        "id": "3b_rice_stage",
        "name": "Phase 3b: Rice Stage Model",
        "description": "Retrain rice stage model",
        "script": "scripts/train_efficientnet.py",
        "args": [
            "--model", "rice_stage",
            "--arch", "swin_t",
            "--epochs", "60",
            "--phase1_epochs", "10",
            "--phase2_lr", "2e-5",
            "--img-size", "224",
            "--aug", "both",
            "--ema", "0.999",
        ],
        "required": False,
        "output_model": "rice_stage_efficientnet_best.pt",
        "output_history": "rice_stage_efficientnet_history.json",
    },
    {
        "id": "4_rice",
        "name": "Phase 4: Rice Disease Model",
        "description": "Retrain rice disease if needed",
        "script": "scripts/train_efficientnet.py",
        "args": [
            "--model", "rice",
            "--arch", "swin_t",
            "--epochs", "60",
            "--phase1_epochs", "10",
            "--phase2_lr", "2e-5",
            "--img-size", "224",
            "--aug", "both",
            "--ema", "0.999",
        ],
        "required": False,
        "output_model": "rice_efficientnet_best.pt",
        "output_history": "rice_efficientnet_history.json",
    },
]


def run_phase(phase: Dict, resume: bool = False, dry_run: bool = False) -> Dict:
    """Execute a single training phase."""
    result = {
        "phase_id": phase["id"],
        "name": phase["name"],
        "started": datetime.now().isoformat(),
        "success": False,
        "duration_sec": 0,
        "error": None,
    }

    script = REPO / phase["script"]
    if not script.exists():
        result["error"] = f"Script not found: {script}"
        result["finished"] = datetime.now().isoformat()
        return result

    args = ["python", str(script)] + phase["args"]
    if resume:
        args.append("--resume")

    log_file = LOGS_DIR / f"{phase['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    print(f"\n{'='*60}")
    print(f"Running: {phase['name']}")
    print(f"Command: {' '.join(args)}")
    print(f"Log: {log_file}")
    print(f"{'='*60}")

    if dry_run:
        result["success"] = True
        result["finished"] = datetime.now().isoformat()
        return result

    t0 = time.time()
    try:
        with open(log_file, "w", encoding="utf-8", buffering=1) as lf:
            lf.write(f"Command: {' '.join(args)}\n")
            lf.write(f"Started: {result['started']}\n\n")
            proc = subprocess.Popen(
                args,
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                print(line, end="")
                lf.write(line)
            proc.wait()
        result["success"] = proc.returncode == 0
        if not result["success"]:
            result["error"] = f"Exit code {proc.returncode}. See log: {log_file}"
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["duration_sec"] = round(time.time() - t0, 1)
        result["finished"] = datetime.now().isoformat()

    return result


def evaluate_model(model_path, model_name="wheat", split="test"):
    """Run per-class evaluation on specified split"""
    from inference import load_model
    import torch
    from torchvision import transforms
    from PIL import Image
    
    classes = ["Crown & Root Rot", "Healthy", "Leaf Rust", "Loose Smut"]
    ckpt = Path(model_path)
    if not ckpt.exists():
        return {"error": f"Checkpoint not found: {ckpt}"}
    
    model, _ = load_model(ckpt, len(classes), classes)
    model.eval()
    size = getattr(model, "input_size", 224)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    split_dir = REPO / "data" / "split" / model_name / split
    if not split_dir.is_dir():
        return {"error": f"Split not found: {split_dir}"}
    
    images, labels = [], []
    for idx, cls in enumerate(classes):
        cdir = split_dir / cls
        if not cdir.is_dir():
            continue
        for f in sorted(cdir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                images.append(f)
                labels.append(idx)
    
    if not images:
        return {"error": "No images found"}
    
    preds = []
    with torch.no_grad():
        for f, lab in zip(images, labels):
            img = Image.open(f).convert("RGB")
            x = tf(img).unsqueeze(0)
            out = model(x)
            preds.append(int(out.argmax(1).item()))
    
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    for lab, p in zip(labels, preds):
        cm[lab][p] += 1
    
    rows = []
    for c in range(n):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n)) - tp
        fn = sum(cm[c][r] for r in range(n)) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": classes[c], "count": tp + fn, "tp": tp, 
                     "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})
    
    acc = sum(cm[c][c] for c in range(n)) / len(images)
    macro_f1 = sum(r["f1"] for r in rows) / n
    gate_passed = all(min(r["precision"], r["recall"], r["f1"]) >= 0.90 for r in rows)
    
    return {
        "split": split,
        "images": len(images),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": rows,
        "confusion_matrix": cm,
        "gate_passed": gate_passed,
        "confusion_matrix": cm
    }


def run_all_phases(resume: bool = False, skip_optional: bool = False, dry_run: bool = False) -> List[Dict]:
    """Run all phases in order, with automatic test evaluation after each."""
    results = []
    for phase in PHASES:
        if skip_optional and not phase["required"]:
            print(f"Skipping optional phase: {phase['name']}")
            continue
        result = run_phase(phase, resume=resume, dry_run=dry_run)
        results.append(result)
        
        # Auto-evaluate on test set if training succeeded
        if result["success"] and phase.get("output_model"):
            model_path = MODELS_DIR / phase["output_model"]
            if model_path.exists():
                print(f"\n📊 Auto-evaluating {phase['name']} on TEST set...")
                eval_result = evaluate_model(model_path)
                result["test_eval"] = eval_result
                if eval_result.get("gate_passed"):
                    print(f"✅ SHIP GATE PASSED: All classes ≥90% P/R/F1")
                else:
                    print(f"❌ SHIP GATE FAILED: {eval_result}")
        
        if not result["success"] and phase["required"]:
            print(f"\n❌ Required phase failed: {phase['name']}. Stopping.")
            break
        elif not result["success"]:
            print(f"\n⚠️ Optional phase failed: {phase['name']}. Continuing...")
    return results


def collect_outputs(results: List[Dict]) -> Dict:
    """Collect trained models, histories, and eval reports into output package."""
    pkg_dir = OUTPUT_DIR / f"package_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    models_dir = pkg_dir / "models"
    models_dir.mkdir(exist_ok=True)
    reports_dir = pkg_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    logs_dir = pkg_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    collected = {"models": [], "histories": [], "reports": [], "logs": []}

    # Copy models
    for phase in PHASES:
        out_model = phase.get("output_model")
        out_hist = phase.get("output_history")
        if out_model:
            src = MODELS_DIR / out_model
            if src.exists():
                dst = models_dir / out_model
                shutil.copy2(src, dst)
                collected["models"].append(str(dst.relative_to(pkg_dir)))
        if out_hist:
            src = MODELS_DIR / out_hist
            if src.exists():
                dst = reports_dir / out_hist
                shutil.copy2(src, dst)
                collected["histories"].append(str(dst.relative_to(pkg_dir)))

    # Run eval on each wheat model
    for model_file in models_dir.glob("wheat*.pt"):
        if "stage" in model_file.name:
            continue
        eval_log = reports_dir / f"{model_file.stem}_eval.json"
        cmd = [
            "python", str(REPO / "scripts/eval_per_class.py"),
            "--model", "wheat",
            "--ckpt", str(model_file),
            "--split", "val",
        ]
        try:
            subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
            if eval_log.exists():
                collected["reports"].append(str(eval_log.relative_to(pkg_dir)))
        except Exception:
            pass

    # Copy logs
    for log in LOGS_DIR.glob("*.log"):
        shutil.copy2(log, logs_dir / log.name)
        collected["logs"].append(str((logs_dir / log.name).relative_to(pkg_dir)))

    # Manifest
    manifest = {
        "created": datetime.now().isoformat(),
        "phases_run": [r["phase_id"] for r in results if r["success"]],
        "phases_failed": [r["phase_id"] for r in results if not r["success"]],
        "collected": collected,
        "ship_gate": "Run eval_per_class.py on each model to verify P/R/F1 >= 0.90 per class",
    }
    with open(pkg_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n📦 Output package created: {pkg_dir}")
    print(f"   Models: {len(collected['models'])}")
    print(f"   Histories: {len(collected['histories'])}")
    print(f"   Eval reports: {len(collected['reports'])}")
    return manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CropGuardAI Training Orchestrator")
    ap.add_argument("--resume", action="store_true", help="Resume each phase from checkpoint")
    ap.add_argument("--skip-optional", action="store_true", help="Skip optional phases")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    ap.add_argument("--phase", type=str, help="Run only specific phase ID")
    args = ap.parse_args()

    if args.phase:
        phase = next((p for p in PHASES if p["id"] == args.phase), None)
        if not phase:
            print(f"Unknown phase: {args.phase}")
            sys.exit(1)
        results = [run_phase(phase, resume=args.resume, dry_run=args.dry_run)]
    else:
        results = run_all_phases(resume=args.resume, skip_optional=args.skip_optional, dry_run=args.dry_run)

    if not args.dry_run:
        collect_outputs(results)

    print("\n=== SUMMARY ===")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {r['name']} ({r['duration_sec']:.0f}s)" + (f" - {r['error']}" if r['error'] else ""))