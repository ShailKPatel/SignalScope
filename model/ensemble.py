"""
SignalScope 3-Member Stacked Ensemble
  1. dima806/deepfake_vs_real_image_detection   (ViT-Base)
  2. Organika/sdxl-detector                      (Swin)
  3. SignalScope Dual-Stream                     (ResNet34 spatial + 2D FFT frequency)

Each member's raw output is converted to P(AI-generated) by model.labels. The
members are frozen level-0 models; their outputs are fused by a level-1 stacking
meta-learner: an L2-regularized (ridge) logistic regression over the members'
logit(P(AI)), fitted on out-of-fold predictions by retrain/train_stacker.py and
stored as plain JSON at STACKER_PATH.

When no meta-learner has been trained yet, or a member fails to load (the
meta-learner needs every feature it was fitted on), the ensemble falls back to a
hard majority vote. An even split resolves to real, since wrongly flagging a
genuine photo is the costly error.
"""

import json
import os
import time
import numpy as np
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from .labels import ai_class_index
from .pretrained_detector import generate_spatial_gradcam

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DUAL_STREAM_CHECKPOINTS = [
    os.path.join(_ROOT, "trained-v1", "best_model.zip"),
    os.path.join(_ROOT, "retrain", "checkpoints", "best_model.pt"),
]
STACKER_PATH = os.environ.get(
    "SIGNALSCOPE_STACKER", os.path.join(_ROOT, "retrain", "checkpoints", "stacking_metalearner.json"))

_ENSEMBLE_MODELS = {}
_DUAL_STREAM = {}
_STACKER = {}

ENSEMBLE_MEMBERS = [
    {
        "id": "vit_dima806",
        "family": "Vision Transformer (ViT)",
        "name": "dima806/deepfake_vs_real_image_detection",
        "architecture": "ViT-Base/16 (in21k), fine-tuned",
        "specialization": "Global Spatial Attention & Semantic Incoherence",
    },
    {
        "id": "swin_sdxl",
        "family": "Swin Transformer (SDXL fine-tune)",
        "name": "Organika/sdxl-detector",
        "architecture": "Swin, fine-tuned on Wikimedia vs SDXL pairs",
        "specialization": "Latent Diffusion Noise & SDXL Render Anomaly Detection",
    },
    {
        "id": "dual_stream_freq",
        "family": "Dual-Stream Spatial + 2D FFT Frequency",
        "name": "SignalScope PyTorch Dual-Stream Backbone",
        "architecture": "ResNet34 + 2D Fast Fourier Transform (FFT)",
        "specialization": "High-Frequency Spectral Grid Spikes & Fourier Artifacts",
    },
]


def load_ensemble_model(model_name):
    """
    Loads and caches a HuggingFace vision classification model with its AI class index.
    Returns (model, processor, ai_idx) or (None, None, None).
    """
    if not (HAS_TORCH and HAS_TRANSFORMERS):
        return None, None, None

    if model_name in _ENSEMBLE_MODELS:
        return _ENSEMBLE_MODELS[model_name]

    for local_only in (True, False):
        try:
            proc = AutoImageProcessor.from_pretrained(model_name, local_files_only=local_only)
            mdl = AutoModelForImageClassification.from_pretrained(
                model_name, attn_implementation="eager", local_files_only=local_only)
            mdl.eval()
            entry = (mdl, proc, ai_class_index(mdl.config.id2label))
            _ENSEMBLE_MODELS[model_name] = entry
            print(f"Ensemble: loaded '{model_name}' id2label={mdl.config.id2label} -> AI index {entry[2]}")
            return entry
        except Exception as e:
            if not local_only:
                print(f"Ensemble load note for '{model_name}': {e}")
    return None, None, None


def evaluate_hf_member(model_name, pil_img, with_cam=True):
    """Returns (p_ai, cam_map) for a HuggingFace member, or (None, None)."""
    mdl, proc, ai_idx = load_ensemble_model(model_name)
    if mdl is None:
        return None, None
    try:
        inputs = proc(images=pil_img.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            p_ai = float(torch.softmax(mdl(**inputs).logits, dim=-1)[0, ai_idx].item())
        cam = generate_spatial_gradcam(mdl, inputs, target_class_idx=ai_idx) if with_cam else None
        return p_ai, cam
    except Exception as e:
        print(f"Ensemble member '{model_name}' note: {e}")
        return None, None


def load_dual_stream():
    """Loads the dual-stream checkpoint once. Returns (model, transform, temperature) or None."""
    if "entry" in _DUAL_STREAM:
        return _DUAL_STREAM["entry"]
    _DUAL_STREAM["entry"] = None
    if not HAS_TORCH:
        return None

    import torchvision.transforms as transforms
    from retrain.backbone import SignalScopeDualStreamModel

    for path in _DUAL_STREAM_CHECKPOINTS:
        if not os.path.exists(path):
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model = SignalScopeDualStreamModel(
                spatial_backbone=ckpt.get("spatial_backbone", ckpt.get("backbone", "resnet34")), pretrained=False)
            model.load_state_dict(state)
            model.eval()
            # Mirror the training transform. Newer checkpoints record native_size (resize + center
            # crop, which keeps frequency artifacts); older ones such as trained-v1 were trained
            # on a plain square resize to image_size.
            if "native_size" in ckpt:
                tf = transforms.Compose([
                    transforms.Resize(ckpt["native_size"]),
                    transforms.CenterCrop(ckpt["image_size"]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                size = ckpt.get("image_size", 224)
                tf = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            # square_size marks the plain-resize pipeline, which the batched path can reproduce exactly.
            square_size = None if "native_size" in ckpt else ckpt.get("image_size", 224)
            entry = (model, tf, float(ckpt.get("temperature", 1.0)), square_size)
            _DUAL_STREAM["entry"] = entry
            print(f"Ensemble: loaded dual-stream checkpoint '{os.path.normpath(path)}' (T={entry[2]:.3f})")
            return entry
        except Exception as e:
            print(f"Dual-stream checkpoint '{path}' note: {e}")
    return None


def evaluate_dual_stream_member(pil_img):
    """Returns (p_ai, None). Trained with label 1 = AI, so sigmoid output is already P(AI)."""
    entry = load_dual_stream()
    if entry is None:
        return None, None
    model, tf, temperature, _ = entry
    try:
        with torch.no_grad():
            logit = model(tf(pil_img.convert("RGB")).unsqueeze(0))
            return float(torch.sigmoid(logit / temperature).item()), None
    except Exception as e:
        print(f"Dual-stream member note: {e}")
        return None, None


def score_members(pil_img, with_cam=False):
    """Runs every member. Returns a list of (member_cfg, p_ai or None, cam or None)."""
    results = []
    for cfg in ENSEMBLE_MEMBERS:
        if cfg["id"] == "dual_stream_freq":
            p_ai, cam = evaluate_dual_stream_member(pil_img)
        else:
            p_ai, cam = evaluate_hf_member(cfg["name"], pil_img, with_cam=with_cam)
        results.append((cfg, p_ai, cam))
    return results


# ---------------------------------------------------------------------------
# Batched scoring (evaluation). Every HF member here square-resizes to 224 with
# no crop (ViTs bilinear, Swins bicubic), and the trained-v1 dual-stream uses a
# plain bilinear 224 resize, so resizing once per resample mode and normalizing
# in torch feeds each model exactly what its own preprocessing would.
# ---------------------------------------------------------------------------
_BATCH_SIZE = 224
_PIL_RESAMPLE = {2: Image.BILINEAR, 3: Image.BICUBIC}


def prepare_image(pil_img):
    """CPU-side prep, safe to run in threads: {resample_mode: uint8 HxWx3 array}."""
    rgb = pil_img.convert("RGB")
    return {mode: np.asarray(rgb.resize((_BATCH_SIZE, _BATCH_SIZE), pil_mode)) for mode, pil_mode in _PIL_RESAMPLE.items()}


def _to_normalized(prepared, mode, mean, std):
    x = torch.from_numpy(np.stack([p[mode] for p in prepared])).permute(0, 3, 1, 2).float() / 255.0
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


@torch.no_grad() if HAS_TORCH else (lambda f: f)
def score_members_batch(prepared):
    """
    Scores a batch of prepare_image() outputs with every member.
    Returns an [N, len(ENSEMBLE_MEMBERS)] array of P(AI), NaN where a member is unavailable.
    """
    out = np.full((len(prepared), len(ENSEMBLE_MEMBERS)), np.nan)
    for j, cfg in enumerate(ENSEMBLE_MEMBERS):
        if cfg["id"] == "dual_stream_freq":
            entry = load_dual_stream()
            if entry is None:
                continue
            model, tf, temperature, square_size = entry
            if square_size != _BATCH_SIZE:
                x = torch.stack([tf(Image.fromarray(p[2])) for p in prepared])
            else:
                x = _to_normalized(prepared, 2, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            out[:, j] = torch.sigmoid(model(x) / temperature).squeeze(1).numpy()
        else:
            mdl, proc, ai_idx = load_ensemble_model(cfg["name"])
            if mdl is None:
                continue
            size = proc.size
            hw = (size.get("height"), size.get("width")) if isinstance(size, dict) else (getattr(size, "height", None), getattr(size, "width", None))
            if hw != (_BATCH_SIZE, _BATCH_SIZE) or proc.resample not in _PIL_RESAMPLE \
                    or getattr(proc, "do_center_crop", False):
                raise ValueError(f"{cfg['name']} preprocessing {size}/{proc.resample} not supported by the batched path")
            x = _to_normalized(prepared, proc.resample, proc.image_mean, proc.image_std)
            out[:, j] = torch.softmax(mdl(pixel_values=x).logits, dim=-1)[:, ai_idx].numpy()
    return out


def majority_vote(p_ai_values):
    """
    Hard majority vote over P(AI) values. Returns (is_ai, ai_votes, n_voters).
    Ties resolve to real.
    """
    votes = [p >= 0.5 for p in p_ai_values]
    ai_votes = sum(votes)
    return ai_votes > len(votes) / 2, ai_votes, len(votes)


# ---------------------------------------------------------------------------
# Stacking meta-learner (level 1). Trained by retrain/train_stacker.py; stored
# as JSON so inference needs only numpy.
# ---------------------------------------------------------------------------
_LOGIT_EPS = 1e-6


def logit_features(P, eps=_LOGIT_EPS):
    """[N, M] member P(AI) -> [N, M] logits. Logit space makes a linear meta-learner a log-odds pool."""
    P = np.clip(np.asarray(P, dtype=np.float64), eps, 1.0 - eps)
    return np.log(P) - np.log1p(-P)


def load_stacker(path=STACKER_PATH):
    """Loads and caches the meta-learner JSON. Returns the spec dict, or None when untrained or invalid."""
    if path in _STACKER:
        return _STACKER[path]
    _STACKER[path] = None
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            spec = json.load(f)
        member_ids = [m["id"] for m in ENSEMBLE_MEMBERS]
        if spec.get("members") != member_ids:
            print(f"Stacker '{path}' was trained on {spec.get('members')}, ensemble is {member_ids} - ignoring it")
            return None
        for key in ("scaler_mean", "scaler_scale", "coefficients"):
            spec[key] = np.asarray(spec[key], dtype=np.float64)
            if spec[key].shape != (len(member_ids),):
                raise ValueError(f"{key} has shape {spec[key].shape}")
        _STACKER[path] = spec
        print(f"Ensemble: loaded stacking meta-learner '{os.path.normpath(path)}' "
              f"(C={spec.get('C')}, threshold={spec.get('threshold', 0.5):.3f})")
    except Exception as e:
        print(f"Stacker '{path}' note: {e}")
    return _STACKER[path]


def stacked_decision(stacker, p_ai_values):
    """
    Returns (p_stacked, is_ai, score). p_stacked is the meta-learner's P(AI);
    is_ai compares it to the trained threshold t. score maps [0, t) -> [0, 0.5)
    and [t, 1] -> [0.5, 1] piecewise-linearly, so callers using a fixed 0.5 cut
    (model/predict.py) agree with the tuned operating point.
    """
    x = logit_features([p_ai_values], stacker.get("eps", _LOGIT_EPS))[0]
    z = float(np.dot((x - stacker["scaler_mean"]) / stacker["scaler_scale"], stacker["coefficients"])
              + stacker["intercept"])
    p = float(1.0 / (1.0 + np.exp(-z)))
    t = float(stacker.get("threshold", 0.5))
    is_ai = p >= t
    score = 0.5 + 0.5 * (p - t) / (1.0 - t) if is_ai else 0.5 * p / t
    return p, is_ai, min(max(score, 0.0), 1.0)


def _fuse(scored, n_members):
    """
    Fuses available member outputs. Returns (score, ai_votes, n, stacking_info or None, strategy).

    With a trained meta-learner and every member available, score is the stacked
    P(AI) remapped so >= 0.5 means AI at the trained threshold. Otherwise score is
    the fraction of AI votes, nudged just below 0.5 on a tie so a >= 0.5 threshold
    agrees with the vote.
    """
    probs = [p for _, p, _ in scored]
    is_ai, ai_votes, n = majority_vote(probs)
    stacker = load_stacker()
    if stacker is not None and len(scored) == n_members:
        p_stacked, is_ai, score = stacked_decision(stacker, probs)
        # Each member's additive share of the meta-learner's log-odds.
        x = logit_features([probs], stacker.get("eps", _LOGIT_EPS))[0]
        contrib = (x - stacker["scaler_mean"]) / stacker["scaler_scale"] * stacker["coefficients"]
        stacking = {
            "meta_learner": "L2-regularized (ridge) logistic regression",
            "C": stacker.get("C"),
            "stacked_ai_probability": round(p_stacked, 4),
            "decision_threshold": round(float(stacker.get("threshold", 0.5)), 4),
            "member_weights": {cfg["id"]: round(float(w), 4) for (cfg, _, _), w in zip(scored, stacker["coefficients"])},
            "member_logit_contributions": {cfg["id"]: round(float(c), 4) for (cfg, _, _), c in zip(scored, contrib)},
            "intercept": round(float(stacker["intercept"]), 4),
            "cifake_test_metrics": stacker.get("test_metrics"),
        }
        return score, ai_votes, n, stacking, "Stacked Generalization (ridge logistic meta-learner)"

    score = ai_votes / n
    if not is_ai and score >= 0.5:
        score = 0.499
    reason = "meta-learner not trained" if stacker is None else "a member failed to load"
    return score, ai_votes, n, None, f"Hard Majority Vote fallback ({reason}; ties resolve to real)"


def score_image(pil_img):
    """Ensemble score only (no saliency), same fusion as run_ensemble_inference. None if no member loads."""
    all_scored = score_members(pil_img, with_cam=False)
    scored = [(cfg, p, cam) for cfg, p, cam in all_scored if p is not None]
    if not scored:
        return None
    return float(_fuse(scored, len(all_scored))[0])


def score_images(pil_imgs):
    """
    Ensemble scores for several images, HF members batched. The dual-stream member is scored
    per image through its own transform, because the batched path would resample through 224px
    and disturb the frequency content it keys on. Entries are None where no member loads.
    """
    try:
        P = score_members_batch([prepare_image(im) for im in pil_imgs])
    except Exception as e:
        print(f"Batched scoring note: {e}")
        return [score_image(im) for im in pil_imgs]

    out = []
    for im, row in zip(pil_imgs, P):
        scored = []
        for cfg, p in zip(ENSEMBLE_MEMBERS, row):
            if cfg["id"] == "dual_stream_freq":
                p = evaluate_dual_stream_member(im)[0]
            if p is not None and not np.isnan(p):
                scored.append((cfg, float(p), None))
        out.append(float(_fuse(scored, len(ENSEMBLE_MEMBERS))[0]) if scored else None)
    return out


def run_ensemble_inference(pil_img):
    """Returns (score, composite_cam, breakdown). See _fuse for how score is formed."""
    t0 = time.time()
    all_scored = score_members(pil_img, with_cam=True)
    scored = [(cfg, p, cam) for cfg, p, cam in all_scored if p is not None]
    if not scored:
        return None, None, {}

    score, ai_votes, n, stacking, strategy = _fuse(scored, len(all_scored))

    model_results = [{
        "model_id": cfg["id"],
        "family": cfg["family"],
        "model_name": cfg["name"],
        "architecture": cfg["architecture"],
        "specialization": cfg["specialization"],
        "ai_probability": round(p, 4),
        "is_ai_pred": p >= 0.5,
    } for cfg, p, _ in scored]

    composite_cam = None
    cams = [cam for _, _, cam in scored if cam is not None]
    if cams:
        valid = [c for c in cams if c.shape == cams[0].shape]
        composite_cam = np.max(np.stack(valid, axis=0), axis=0)
        if composite_cam.max() > 0:
            composite_cam = (composite_cam - composite_cam.min()) / (composite_cam.max() - composite_cam.min() + 1e-8)

    breakdown = {
        "ensemble_strategy": strategy,
        "stacking": stacking,
        "num_families_evaluated": n,
        "ai_votes": ai_votes,
        "mean_ai_probability": round(float(np.mean([p for _, p, _ in scored])), 4),
        "total_inference_ms": round((time.time() - t0) * 1000, 2),
        "family_models": model_results,
    }
    return round(float(score), 4), composite_cam, breakdown
