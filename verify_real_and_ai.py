import os
import sys
import json
import urllib.request

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model.predict import predict_image

def main():
    os.makedirs("test_images", exist_ok=True)

    # 1. Real photograph: CIFAKE test split, REAL class (CIFAR-10), no people
    real_path = "test_images/cifake_real_0.png"

    # 2. Path to AI Image sample (e.g., test_images/gemini_imagen3_sample.png)
    ai_path = "test_images/gemini_imagen3_sample.png"

    print("\n" + "="*80)
    print("      SIGNAL SCOPE - REAL VS AI IMAGE DETECTOR VERIFICATION")
    print("="*80)

    # Analyze Real Image
    print("\n--- ANALYZING REAL IMAGE ---")
    res_real = predict_image(real_path)
    
    print("\n[REAL IMAGE RESULTS]")
    print(f"Filename:               {res_real['filename']}")
    print(f"Detection Level:        {res_real['verdict']['detection_level']}")
    print(f"Verdict Label:          {res_real['verdict']['label']}")
    print(f"Is AI Generated:        {res_real['verdict']['is_ai_generated']}")
    print(f"Confidence Score:       {res_real['verdict']['confidence_score'] * 100:.1f}%")
    print(f"Metadata Assessment:    {res_real['modules']['module_d_metadata']['assessment']}")
    print(f"Grad-CAM Heatmap Len:   {len(res_real['modules']['module_a_explainability']['heatmap_base64'])} chars")

    # Analyze AI Image
    print("\n--- ANALYZING AI IMAGE ---")
    res_ai = predict_image(ai_path)

    print("\n[AI IMAGE RESULTS]")
    print(f"Filename:               {res_ai['filename']}")
    print(f"Detection Level:        {res_ai['verdict']['detection_level']}")
    print(f"Verdict Label:          {res_ai['verdict']['label']}")
    print(f"Is AI Generated:        {res_ai['verdict']['is_ai_generated']}")
    print(f"Confidence Score:       {res_ai['verdict']['confidence_score'] * 100:.1f}%")
    print(f"Matched AI Generator:   {res_ai['verdict']['matched_generator']}")
    print(f"Metadata Evidence:      {res_ai['verdict']['metadata_evidence']}")
    print(f"Metadata Assessment:    {res_ai['modules']['module_d_metadata']['assessment']}")
    print(f"Grad-CAM Heatmap Len:   {len(res_ai['modules']['module_a_explainability']['heatmap_base64'])} chars")

    # Save detailed JSON log
    output_summary = {
        "real_image_analysis": res_real,
        "ai_image_analysis": res_ai
    }
    with open("verification_results.json", "w") as f:
        json.dump(output_summary, f, indent=2)
        
    print("\nVerification complete! Saved results to verification_results.json.")

if __name__ == "__main__":
    main()
