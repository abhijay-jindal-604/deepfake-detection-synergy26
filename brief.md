# Deepfake ML Hackathon — Synergy '26 (IIIT Bangalore)

- **Unstop link:** https://unstop.com/hackathons/deepfake-ml-hackathon-synergy-26-international-institute-of-information-technology-iiit-bangalore-1762774
- **Mode:** Online, single-round ML challenge
- **Team size:** 1–4 members
- **Your status:** Registered (team "abhijayjindal604", solo). Registration is closed
- **Prize pool:** ₹20,000 (1st ₹10,000 · 2nd ₹6,000 · 3rd ₹4,000, plus participation certificate)
- **Eligibility:** UG, PG, engineering, management, arts/commerce/sciences, law, medical, freshers, experienced professionals
- **Organiser contact:** Shreyas — synergy@iiitb.ac.in
- **Dataset (Google Drive folder, with related documents):** https://drive.google.com/drive/folders/1hs7q-RG-gu0TZYUUcA15qTa2kb1jMaqP?usp=drive_link
  The organisers say "read all the documents" in that folder. I did not open or download it. It is the first thing to grab.

## Key dates (IST)
| Date | Event |
|---|---|
| 30 Oct 2026, 11:59 PM | Registration deadline (passed) |
| 12 Nov 2026, 11:59 PM → 15 Nov 2026, 12:26 AM | Deliverables-submission round window on Unstop (as listed) |
| **15 Nov 2026, 11:59 PM** | **Final submission deadline** (time TBA). The round window above ends 12:26 AM on 15 Nov, so aim to submit well before the 14th ends to be safe |

## The task
Build accurate, efficient and robust models that detect **deepfake / synthetic images**. Output a probability of Fake for each of **3,000 test images**.

## Deliverables (the Unstop round says one zip)
Submit a single zip named `"TeamName"_submission.zip` containing:
1. `predictions.csv` — columns `image_id,label,confidence` for all 3,000 test images; confidence = P(Fake) in [0.0, 1.0]
2. `inference_script.py` — working script matching `inference_template.py` (from the dataset folder)
3. Model weights — `.pth`, `.pt`, `.h5` or `.keras`, loaded by the script
4. `requirements.txt` — dependencies to run the code
5. `report.pdf` — brief 1–2 pages: architecture, augmentations, validation strategy

Constraints: **no external datasets**, and the script must **reproduce predictions.csv**. If the zip is too large for the upload limit, submit a txt file with a **public** Google Drive folder link. Non-public folders are disqualified.

(The long description also lists, for Drive-link submissions: source code as a zip or public GitHub repo with a README, trained weights for PyTorch or TensorFlow, prediction CSV, detailed documentation covering approach, architecture, hyperparameters, training pipeline and challenges, and an optional 3–5 minute explainer video.)

## Rules
- Train only on the **officially provided dataset**. External training data is strictly prohibited
- **Pre-trained architectures are allowed**, but fine-tuning must use only the provided dataset
- Fully **reproducible**. Incomplete or unverified code is not evaluated
- Original work; plagiarism means immediate disqualification
- One final submission per team is evaluated. You can update files until the deadline and the latest upload counts

## Evaluation criteria
| Weight | Criterion |
|---|---|
| 50% | **Accuracy on the unlabelled test set** |
| 20% | **Model efficiency:** inference speed and memory footprint |
| 20% | **Generalization and robustness:** stability and resistance to edge-case noise/artifacts |
| 10% | **Documentation clarity:** quality and depth of the technical report |

## Notes for the build
- Start from the dataset folder and its docs, since the exact image_id format and the inference template define the script's interface.
- A pre-trained backbone (e.g. an EfficientNet, ConvNeXt or ViT variant) fine-tuned only on the provided data is the usual strong baseline.
- Plan augmentations for the robustness score (compression, blur, noise, resize), and keep a small, fast model for the efficiency score.
- Keep the whole pipeline reproducible: fixed seeds, one command to regenerate predictions.csv, pinned requirements.
