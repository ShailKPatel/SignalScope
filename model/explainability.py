"""
Module A: Explanation & Saliency Overlay
Renders the detector's saliency map (when one exists) and builds cues only from
measured signals: member model outputs, the stacker's per-member contributions,
the image's own frequency spectrum, and metadata evidence. Nothing here is
invented when a signal is missing; the cue is simply omitted.
"""

import base64
import io
import numpy as np
from PIL import Image

_FFT_SIDE = 256


def format_probability(p):
    """Percent text that never claims certainty: >=0.999 -> ">99.9%", <=0.001 -> "<0.1%" (matches the UI)."""
    if p >= 0.999:
        return ">99.9%"
    if p <= 0.001:
        return "<0.1%"
    return f"{p * 100:.1f}%"


def high_frequency_energy_ratio(image_pil):
    """
    Share of spectral power above half-Nyquist, measured on a native-resolution
    center crop (no resampling, which would itself change the spectrum).
    """
    g = np.asarray(image_pil.convert("L"), dtype=np.float32)
    side = min(_FFT_SIDE, *g.shape)
    top, left = (g.shape[0] - side) // 2, (g.shape[1] - side) // 2
    g = g[top:top + side, left:left + side]
    g = g - g.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    yy, xx = np.indices(power.shape)
    radius = np.hypot(yy - side / 2, xx - side / 2)
    return float(power[radius > side / 4].sum() / (power.sum() + 1e-8))


def _overlay_base64(width, height, saliency, is_ai_generated):
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if saliency is not None:
        h = Image.fromarray((np.clip(saliency, 0, 1) * 255).astype(np.uint8), mode="L")
        h_arr = np.asarray(h.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        # JET colormap (red = high, blue = low)
        r = np.clip(1.5 - np.abs(h_arr * 4.0 - 3.0), 0, 1)
        g = np.clip(1.5 - np.abs(h_arr * 4.0 - 2.0), 0, 1)
        b = np.clip(1.5 - np.abs(h_arr * 4.0 - 1.0), 0, 1)
        a = h_arr * (0.7 if is_ai_generated else 0.25)
        overlay = Image.fromarray(np.stack([r, g, b, a], axis=-1).__mul__(255).astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_heatmap_and_explanations(image_pil, confidence, is_ai_generated, saliency_map=None,
                                      ensemble_info=None, metadata_evidence=None, saliency_source=None,
                                      saliency_check=None):
    width, height = image_pil.size
    cues = []

    if metadata_evidence:
        cues.append({"type": "Metadata signature", "detail": metadata_evidence, "value": "Declared in file"})

    members = (ensemble_info or {}).get("family_models") or []
    contributions = ((ensemble_info or {}).get("stacking") or {}).get("member_logit_contributions") or {}
    for m in members:
        detail = f"{m['model_name']} ({m['architecture']}) scores P(AI) {format_probability(m['ai_probability'])}."
        if m["model_id"] in contributions:
            c = contributions[m["model_id"]]
            detail += f" Its share of the stacked log-odds is {c:+.2f} ({'toward AI' if c > 0 else 'toward real'})."
        cues.append({"type": f"Member: {m['family']}", "detail": detail, "value": f"P(AI) {format_probability(m['ai_probability'])}"})

    if saliency_check:
        top, rand = saliency_check["logit_shift_top_salient"], saliency_check["logit_shift_random_mean"]
        outcome = ("salient pixels move the score more than random ones, so the map tracks pixels the model uses"
                   if saliency_check["salient_exceeds_random"] else
                   "salient pixels do not move the score more than random ones, so treat the map with caution")
        cues.append({
            "type": "Saliency deletion check",
            "detail": (f"Replacing the {int(saliency_check['fraction_masked'] * 100)}% most salient pixels changes the "
                       f"dual-stream log-odds by {top:+.2f}; the same number of random pixels changes it by {rand:+.2f} "
                       f"(mean of 5 draws). Here {outcome}."),
            "value": f"salient {top:+.2f} vs random {rand:+.2f}",
        })

    hf = high_frequency_energy_ratio(image_pil)
    cues.append({
        "type": "Spectral measurement",
        "detail": ("Share of spectral power above half-Nyquist on a native-resolution crop. Reported for inspection "
                   "only; it is not calibrated as a detector and does not drive the verdict."),
        "value": f"{hf * 100:.2f}%",
    })

    if members:
        ai_votes = sum(m["is_ai_pred"] for m in members)
        agreement = f" {ai_votes} of {len(members)} ensemble members lean AI."
    else:
        agreement = ""
    summary_text = (f"The detector rates this image {'likely AI-generated' if is_ai_generated else 'likely real'} "
                    f"(score {format_probability(confidence)}).{agreement} This is a likelihood assessment, not proof; the cues below list "
                    f"the measured signals behind it.")

    if saliency_map is not None:
        localization = (f"{saliency_source or 'Model saliency map'}. Shows which pixels the score is most "
                        "sensitive to; the deletion check cue tests whether they matter. Not a verified "
                        "localization of generation artifacts.")
    else:
        localization = "Unavailable: no saliency map was produced for this input, so no overlay is drawn."

    return {
        "heatmap_base64": _overlay_base64(width, height, saliency_map, is_ai_generated),
        "has_saliency": saliency_map is not None,
        "cues": cues,
        "summary_text": summary_text,
        "localization_quality": localization,
        "high_frequency_energy_ratio": round(hf, 5),
        "uncertainty_hedged": True,
    }
