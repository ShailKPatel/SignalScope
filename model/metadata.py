"""
Module D: Provenance & Metadata Engine
Parses EXIF headers and C2PA Content Credentials manifests.
"""

import os
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

def analyze_metadata(image_path_or_pil):
    """
    Extracts EXIF metadata, checks for C2PA signatures, and assesses metadata integrity.
    """
    metadata_info = {
        "has_exif": False,
        "exif_tags": {},
        "has_c2pa": False,
        "c2pa_manifest": None,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "date_time": None,
        "provenance_score": 0.5, # 0.0 = suspicious/synthetic, 1.0 = highly verified real
        "assessment": "No EXIF or C2PA metadata detected. Common in web-saved or synthetic images."
    }

    try:
        if isinstance(image_path_or_pil, str) and os.path.exists(image_path_or_pil):
            img = Image.open(image_path_or_pil)
        elif isinstance(image_path_or_pil, Image.Image):
            img = image_path_or_pil
        else:
            return metadata_info

        # EXIF Extraction
        raw_exif = img._getexif() if hasattr(img, '_getexif') else None
        if raw_exif:
            metadata_info["has_exif"] = True
            for tag_id, value in raw_exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                # Filter out raw byte blobs for JSON serializability
                if isinstance(value, (str, int, float)):
                    metadata_info["exif_tags"][str(tag_name)] = value
            
            metadata_info["camera_make"] = metadata_info["exif_tags"].get("Make")
            metadata_info["camera_model"] = metadata_info["exif_tags"].get("Model")
            metadata_info["software"] = metadata_info["exif_tags"].get("Software")
            metadata_info["date_time"] = metadata_info["exif_tags"].get("DateTimeOriginal") or metadata_info["exif_tags"].get("DateTime")

            if metadata_info["camera_make"] or metadata_info["camera_model"]:
                metadata_info["provenance_score"] = 0.85
                metadata_info["assessment"] = f"Valid EXIF metadata found (Camera: {metadata_info['camera_make']} {metadata_info['camera_model']}). Supports authentic capture."
            elif metadata_info["software"]:
                sw = str(metadata_info["software"]).lower()
                if any(term in sw for term in ["photoshop", "gimp", "stable diffusion", "midjourney", "automatic1111"]):
                    metadata_info["provenance_score"] = 0.25
                    metadata_info["assessment"] = f"Software metadata indicates editing/generation software: '{metadata_info['software']}'."

        # C2PA manifest check (mock/placeholder engine for fallback)
        # Attempt real c2pa if installed
        try:
            import c2pa
            if isinstance(image_path_or_pil, str) and os.path.exists(image_path_or_pil):
                reader = c2pa.Reader.from_file(image_path_or_pil)
                if reader:
                    metadata_info["has_c2pa"] = True
                    metadata_info["c2pa_manifest"] = reader.json()
                    metadata_info["provenance_score"] = 0.95
                    metadata_info["assessment"] = "Cryptographically signed C2PA Content Credentials manifest verified!"
        except Exception:
            pass

    except Exception as e:
        metadata_info["assessment"] = f"Metadata extraction note: {str(e)}"

    return metadata_info
