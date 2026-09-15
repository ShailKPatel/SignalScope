"""
Module B: Generator Attribution - NOT BUILT.

No attribution classifier has been trained (CIFAKE has a single generator,
Stable Diffusion v1.4, so it cannot supervise one). The only attribution this
system reports is an explicit generator signature found by the metadata engine
(Module D), which predict.py fills in when present.
"""

NOT_BUILT = "Not built"


def predict_generator_attribution(metadata_generator=None):
    """Returns the metadata-declared generator if one was found, otherwise a not-built marker."""
    if metadata_generator:
        return {
            "status": "metadata_only",
            "family": metadata_generator,
            "specific_model": metadata_generator,
            "source": "Generator signature declared in file metadata (Module D), not a visual classifier.",
            "family_probabilities": {},
        }
    return {
        "status": "not_built",
        "family": NOT_BUILT,
        "specific_model": NOT_BUILT,
        "source": "Module B (visual generator attribution) is not part of this submission.",
        "family_probabilities": {},
    }
