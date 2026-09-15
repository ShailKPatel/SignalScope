"""
SignalScope Master Prediction Engine
Executes 2-Level Cascading Forensic Architecture:
  Level 1: Smart Metadata & Provenance Engine (explicit generator signatures, C2PA)
  Level 2: 3-member stacked ensemble (ViT + Swin + ResNet34/FFT dual-stream), see model/ensemble.py
Module A renders the ViT attention map as a saliency overlay on both levels.
"""

import os
import time
from PIL import Image

from .explainability import format_probability, generate_heatmap_and_explanations
from .attribution import predict_generator_attribution
from .robustness import evaluate_degradation_robustness
from .metadata import analyze_smart_metadata
from .multimodal import evaluate_multimodal_consistency
from .active_defense import analyze_active_defence

# The dual-stream member dominates the stacked decision on CIFAKE but yields no map, so the
# overlay comes from the ViT member; the caveat travels with it.
SALIENCY_SOURCE = ("ViT-Base member's last-layer CLS attention (14x14). The stacked decision is driven mainly "
                   "by the dual-stream member, which produces no map, so this overlay is context, not the "
                   "decision's evidence")


class DetectorUnavailableError(RuntimeError):
    """No detector model could be loaded, so no honest score exists."""


def _score_many_fn():
    """Returns the batched ensemble scorer used to re-score degraded variants, or None if unavailable."""
    try:
        from .ensemble import score_images
        return score_images
    except Exception as e:
        print(f"Ensemble scorer note: {e}")
        return None


def predict_image(image_input, caption_text=None, filename="image.jpg"):
    """
    Standardized 2-Level Cascading Inference Interface.

    Args:
        image_input: File path (str), Bytes, or PIL Image.
        caption_text: Optional caption/claim string (Module E, not built).
        filename: Original filename, echoed in the result only.

    Returns:
        Structured dictionary matching SignalScope 2-level schema.

    Raises:
        DetectorUnavailableError: when Level 1 finds no signature and no Level 2 model loads.
    """
    start_time = time.time()

    if isinstance(image_input, str):
        filename = os.path.basename(image_input)
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        img = Image.open(image_input).convert("RGB")

    width, height = img.size

    # =========================================================================
    # LEVEL 1: SMART METADATA & PROVENANCE ENGINE
    # =========================================================================
    metadata_res = analyze_smart_metadata(image_input if isinstance(image_input, str) else img)
    l1 = metadata_res.get("level_1_result", {})
    is_level_1_ai = l1.get("is_conclusive_ai", False)

    saliency_map = None
    ensemble_info = {}

    if is_level_1_ai:
        matched_gen = l1.get("matched_generator", "AI Generator Metadata Signature")
        evidence_text = l1.get("evidence_detail", "Explicit AI generation metadata signature present.")
        confidence = l1.get("confidence_score", 0.98)
        is_ai_generated = True
        detection_level = "Level 1: Smart Metadata & Provenance Engine"
        verdict_label = f"likely AI-generated (Confirmed via Level 1 Metadata: {matched_gen})"

        # Still run the ViT member for a saliency overlay.
        try:
            from .pretrained_detector import run_pretrained_inference
            _, _, saliency_map = run_pretrained_inference(img, model_name="dima806/deepfake_vs_real_image_detection")
        except Exception as e:
            print(f"Level 1 saliency extraction note: {e}")

    else:
        # =====================================================================
        # LEVEL 2: 3-MEMBER STACKED ENSEMBLE
        # =====================================================================
        detection_level = "Level 2: Deep Learning Vision Model (3-Model Stacked Ensemble)"
        raw_prob = None
        try:
            from .ensemble import run_ensemble_inference
            ens_prob, ens_cam, ens_info = run_ensemble_inference(img)
            if ens_prob is not None:
                raw_prob, saliency_map, ensemble_info = ens_prob, ens_cam, ens_info
        except Exception as e:
            print(f"Multi-Model Ensemble note: {e}")

        if raw_prob is None:
            raise DetectorUnavailableError(
                "No detector model could be loaded (ensemble members unavailable). "
                "Install requirements and the checkpoint described in README, then retry.")

        confidence = round(float(raw_prob), 3)
        is_ai_generated = confidence >= 0.50
        verdict_label = "likely AI-generated" if is_ai_generated else "likely real"
        evidence_text = f"Evaluated via {detection_level} (Confidence Score: {format_probability(confidence)})."

    # Operating point comes from the stacker's CIFAKE test metrics; absent in majority-vote fallback.
    test_metrics = (ensemble_info.get("stacking") or {}).get("cifake_test_metrics") or {}

    # =========================================================================
    # MODULES
    # =========================================================================
    explainability_res = generate_heatmap_and_explanations(
        img, confidence, is_ai_generated,
        saliency_map=saliency_map,
        ensemble_info=ensemble_info,
        metadata_evidence=evidence_text if is_level_1_ai else None,
        saliency_source=SALIENCY_SOURCE,
    )
    attribution_res = predict_generator_attribution(l1.get("matched_generator") if is_level_1_ai else None)
    robustness_res = evaluate_degradation_robustness(img, _score_many_fn())
    multimodal_res = evaluate_multimodal_consistency(img, caption_text)
    defense_res = analyze_active_defence()

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "filename": filename,
        "resolution": f"{width}x{height}",
        "processing_time_ms": elapsed_ms,

        "verdict": {
            "label": verdict_label,
            "is_ai_generated": is_ai_generated,
            "confidence_score": confidence,
            "detection_level": detection_level,
            "metadata_evidence": evidence_text if is_level_1_ai else None,
            "matched_generator": l1.get("matched_generator") if is_level_1_ai else None,
            "ensemble_breakdown": ensemble_info or None,
            "decision_threshold": 0.50,
            "operating_point_fpr": test_metrics.get("fpr"),
            "operating_point_accuracy": test_metrics.get("accuracy"),
            "operating_point_source": "CIFAKE test split, stacked ensemble @ 0.5" if test_metrics else None,
            "framing_note": "Responsible likelihood assessment (never an absolute accusation)"
        },

        "modules": {
            "module_a_explainability": explainability_res,
            "module_b_attribution": attribution_res,
            "module_c_robustness": robustness_res,
            "module_d_metadata": metadata_res,
            "module_e_multimodal": multimodal_res,
            "module_g_active_defense": defense_res
        }
    }


def predict_batch(image_inputs):
    """
    Processes a list of image inputs and returns structured batch predictions.
    """
    batch_results = []
    total_start = time.time()

    for idx, item in enumerate(image_inputs):
        if isinstance(item, tuple):
            img_inp, cap, fname = item
        else:
            img_inp, cap, fname = item, None, f"image_{idx+1}.jpg"

        res = predict_image(img_inp, caption_text=cap, filename=fname)
        batch_results.append(res)

    total_elapsed = round(time.time() - total_start, 2)
    ai_count = sum(1 for r in batch_results if r["verdict"]["is_ai_generated"])
    real_count = len(batch_results) - ai_count

    return {
        "total_images": len(batch_results),
        "total_time_seconds": total_elapsed,
        "summary": {
            "likely_ai_generated_count": ai_count,
            "likely_real_count": real_count,
            "ai_ratio": round(ai_count / max(1, len(batch_results)), 3)
        },
        "results": batch_results
    }
