"""
Module C: Robustness to Degradation
Actually degrades the submitted image (JPEG re-compression, downscaling) and
re-scores the clean image and every variant with the same pixel detector in
one batch, so each curve point is a measured model output. File metadata is not
part of this test: a JPEG re-encode strips it.
"""

import io
from PIL import Image

JPEG_QUALITIES = [90, 70, 50, 30]
RESIZE_SCALES = [0.75, 0.5, 0.25]
_MIN_SIDE = 8


def _jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _downscale(img, scale):
    w, h = img.size
    return img.resize((max(_MIN_SIDE, round(w * scale)), max(_MIN_SIDE, round(h * scale))), Image.BICUBIC)


def _unavailable(reason):
    return {
        "status": "unavailable",
        "overall_stability_rating": f"Unavailable ({reason})",
        "verdict_preserved_under_jpeg70": None,
        "verdict_preserved_under_resize50": None,
        "jpeg_degradation_curve": [],
        "resize_degradation_curve": [],
        "robustness_score": None,
    }


def evaluate_degradation_robustness(img, score_many):
    """
    img: RGB PIL image.
    score_many: list of PIL images -> list of detector scores (>= 0.5 means AI; None if unscorable),
    or None when no detector is available.
    """
    if score_many is None:
        return _unavailable("no detector loaded")

    w, h = img.size
    jpeg_variants = [(q, _jpeg(img, q)) for q in JPEG_QUALITIES]
    resize_variants = []
    for s in RESIZE_SCALES:
        small = _downscale(img, s)
        resize_variants.append((f"{small.width}x{small.height} ({int(s * 100)}%)", small))

    try:
        scores = score_many([img] + [v for _, v in jpeg_variants] + [v for _, v in resize_variants])
    except Exception as e:
        print(f"Robustness scoring note: {e}")
        return _unavailable("detector error")
    if any(s is None for s in scores):
        return _unavailable("detector returned no score")

    base_score = scores[0]
    base_is_ai = base_score >= 0.5

    def row(score):
        return {
            "ai_score": round(score, 3),
            # Confidence in the clean-image verdict, so the curve reads the same for AI and real verdicts.
            "confidence_retained": round(score if base_is_ai else 1.0 - score, 3),
            "verdict_stable": (score >= 0.5) == base_is_ai,
        }

    jpeg_scores = scores[1:1 + len(jpeg_variants)]
    resize_scores = scores[1 + len(jpeg_variants):]
    jpeg_curve = [{"quality": "Original", **row(base_score)}]
    jpeg_curve += [{"quality": q, **row(s)} for (q, _), s in zip(jpeg_variants, jpeg_scores)]
    resize_curve = [{"resolution": f"Original {w}x{h}", **row(base_score)}]
    resize_curve += [{"resolution": label, **row(s)} for (label, _), s in zip(resize_variants, resize_scores)]

    degraded = jpeg_curve[1:] + resize_curve[1:]
    stable = sum(r["verdict_stable"] for r in degraded)
    fraction = stable / len(degraded)
    level = "High" if fraction == 1.0 else "Moderate" if fraction >= 0.5 else "Low"

    return {
        "status": "measured",
        "pixel_detector_clean_score": round(base_score, 3),
        "pixel_detector_verdict": "likely AI-generated" if base_is_ai else "likely real",
        "overall_stability_rating": f"{level} ({stable}/{len(degraded)} degraded variants keep the pixel detector's verdict)",
        "verdict_preserved_under_jpeg70": next(r["verdict_stable"] for r in jpeg_curve if r["quality"] == 70),
        "verdict_preserved_under_resize50": next(r["verdict_stable"] for r in resize_curve if "(50%)" in r["resolution"]),
        "jpeg_degradation_curve": jpeg_curve,
        "resize_degradation_curve": resize_curve,
        "robustness_score": round(fraction, 3),
        "method": "Clean image and its JPEG / downscaled variants re-scored together by the pixel ensemble (metadata ignored).",
    }
