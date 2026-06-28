"""
Load all .pkl classifiers.

Local dev : reads from  models/model (42 dim)/  (already on disk).
Production: set env var HF_MODEL_REPO=username/acne-cv-models  →  downloads
            from Hugging Face Hub on first startup, then caches in /tmp.
"""
import os
import sys
import glob
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LOCAL_MODELS_DIR   = os.path.join(_ROOT, "models", "model (42 dim)")
_LOCAL_SUMMARY_PATH = os.path.join(_ROOT, "models", "Evaluation_Summary(42dim).txt")

_HF_REPO = os.environ.get("HF_MODEL_REPO")  # e.g. "yourname/acne-cv-models"


def _resolve_paths() -> tuple[str, str]:
    """Return (models_dir, summary_path) — local if present, else HF download."""
    if os.path.isdir(_LOCAL_MODELS_DIR):
        return _LOCAL_MODELS_DIR, _LOCAL_SUMMARY_PATH

    if not _HF_REPO:
        raise RuntimeError(
            "No local models found and HF_MODEL_REPO env var is not set.\n"
            "Either copy models/model (42 dim)/ into place, or set:\n"
            "  HF_MODEL_REPO=yourname/acne-cv-models"
        )

    print(f"[models] Local models not found. Downloading from HF Hub: {_HF_REPO} …")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")

    # Downloads & caches under $HF_HOME or /tmp/hf_cache
    cache = os.environ.get("HF_HOME", "/tmp/hf_cache")
    local = snapshot_download(repo_id=_HF_REPO, repo_type="model", cache_dir=cache)

    models_dir   = os.path.join(local, "model_42dim")
    summary_path = os.path.join(local, "Evaluation_Summary(42dim).txt")
    return models_dir, summary_path

CLASSES = ["acne1", "acne2", "acne3"]
_CLASS_MAP = {f"acne{i}_1024": f"acne{i}" for i in range(4)}
CHAMPION = "sgd_classifier"

_models: dict[str, object] = {}
_model_meta: dict[str, dict] = {}


def _pretty_name(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.replace("acne_", "").replace("_model", "")


def _parse_summary(summary_path: str) -> dict[str, dict]:
    meta = {}
    if not os.path.exists(summary_path):
        return meta
    with open(summary_path) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 5:
                name = parts[1].strip().lower().replace(" ", "_")
                try:
                    meta[name] = {"accuracy": float(parts[2].strip()), "f1": float(parts[3].strip())}
                except ValueError:
                    pass
    return meta


def load_models():
    global _models, _model_meta
    models_dir, summary_path = _resolve_paths()
    _model_meta = _parse_summary(summary_path)
    pkls = glob.glob(os.path.join(models_dir, "*.pkl"))
    if not pkls:
        print(f"[models] No .pkl found in {models_dir}")
        return
    import joblib
    for path in sorted(pkls):
        name = _pretty_name(path)
        try:
            _models[name] = joblib.load(path)
        except Exception as e:
            print(f"[models] Failed to load {path}: {e}")
    print(f"[models] Loaded {len(_models)} classifiers.")


def predict_all(vec: np.ndarray) -> list[dict]:
    """Run every model on the 42-dim vector. Returns sorted by F1."""
    X = vec.reshape(1, -1)
    results = []
    for name, model in _models.items():
        try:
            label_idx = int(model.predict(X)[0])
            # Map numeric label back to class name
            if hasattr(model, "classes_"):
                raw = model.classes_[label_idx] if label_idx < len(model.classes_) else label_idx
            else:
                raw = label_idx

            # Normalize to display string
            if isinstance(raw, str):
                label = _CLASS_MAP.get(raw, raw)
            else:
                label = CLASSES[min(int(raw), len(CLASSES) - 1)]

            # Confidence
            conf = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)[0]
                    conf = float(proba.max())
                except Exception:
                    pass
            if conf is None and hasattr(model, "decision_function"):
                try:
                    df = model.decision_function(X)[0]
                    if df.ndim == 0:
                        df = np.array([df])
                    exp = np.exp(df - df.max())
                    conf = float(exp.max() / exp.sum())
                except Exception:
                    pass

            meta = _model_meta.get(name, {})
            results.append({
                "name": name,
                "label": label,
                "confidence": round(conf, 4) if conf is not None else None,
                "accuracy": meta.get("accuracy"),
                "f1": meta.get("f1"),
                "is_champion": name == CHAMPION,
            })
        except Exception as e:
            results.append({"name": name, "label": "error", "confidence": None,
                            "accuracy": None, "f1": None, "is_champion": name == CHAMPION,
                            "error": str(e)})

    # Sort: champion first, then by F1 desc
    results.sort(key=lambda r: (0 if r["is_champion"] else 1, -(r["f1"] or 0)))
    return results
