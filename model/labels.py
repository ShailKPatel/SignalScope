"""
Label normalization layer.

Every ensemble member reports its answer in its own vocabulary and class order:

  dima806/deepfake_vs_real_image_detection    {0: Real,       1: Fake}
  umm-maybe/AI-image-detector                 {0: artificial, 1: human}   <- AI is index 0
  Organika/sdxl-detector                      {0: artificial, 1: human}   <- AI is index 0
  prithivMLmods/Deep-Fake-Detector-v2-Model   {0: Realism,    1: Deepfake}
  SignalScope dual-stream (sigmoid output)    trained with label 1 = AI

Everything downstream works with a single convention: p_ai = probability the
image is AI-generated, and a vote of True means "AI".
"""

import re

# Normalized label text -> True if the class means AI-generated, False if real.
_EXACT = {
    "fake": True, "deepfake": True, "artificial": True, "ai": True, "aiartdata": True,
    "synthetic": True, "generated": True, "aigenerated": True,
    "real": False, "realism": False, "human": False, "realart": False,
    "authentic": False, "natural": False,
}
_AI_TOKENS = ("fake", "artificial", "synthetic", "generated")
_REAL_TOKENS = ("real", "human", "authentic")


def _label_means_ai(label):
    key = re.sub(r"[^a-z]", "", str(label).lower())
    if key in _EXACT:
        return _EXACT[key]
    # "fake" is checked before "real" so labels like "not_real"/"real_or_fake" lean on the AI token.
    if any(tok in key for tok in _AI_TOKENS):
        return True
    if any(tok in key for tok in _REAL_TOKENS):
        return False
    return None


def ai_class_index(id2label):
    """
    Returns the logit index that means "AI-generated" for a binary classifier.
    Raises instead of guessing, since a wrong guess silently inverts every vote.
    """
    ai_idx = [int(i) for i, lbl in id2label.items() if _label_means_ai(lbl) is True]
    real_idx = [int(i) for i, lbl in id2label.items() if _label_means_ai(lbl) is False]
    if len(ai_idx) == 1 and len(real_idx) == len(id2label) - 1:
        return ai_idx[0]
    raise ValueError(f"Cannot tell which class means AI-generated from id2label={id2label}")


def ai_probability_from_logits(logits, id2label):
    """Softmax over a single image's logits, returning P(AI-generated)."""
    import torch
    probs = torch.softmax(logits, dim=-1)
    return probs[..., ai_class_index(id2label)]
