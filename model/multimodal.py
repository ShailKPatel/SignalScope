"""
Module E: Multimodal Image-Text Consistency - NOT BUILT.

No image-text model is loaded, so no consistency score is produced. The caption
is echoed back so the UI can show what was submitted.
"""


def evaluate_multimodal_consistency(image_pil, caption_text=None):
    caption = caption_text.strip() if caption_text else ""
    return {
        "status": "not_built",
        "has_caption": bool(caption),
        "caption_text": caption or None,
        "semantic_similarity": None,
        "is_semantically_consistent": None,
        "assessment": "Module E (image-text consistency) is not part of this submission; no score computed.",
        "embedding_model": None,
    }
