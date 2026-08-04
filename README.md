# Health-Triage-Model-GPT-2
A GPT-2 model fine-tuned on a small symptom-to-diagnosis dataset (gretelai/symptom_to_diagnosis) to support triage prioritization. Rule-based safety layers handle emergencies and self-harm language, a symptom-plausibility check filters unclear input, and a confidence threshold withholds low-certainty suggestions

## ⚠️ Model Files Not Included

This repository does **not** include the fine-tuned model weights (`gpt2-triage-final/`) because the files are too large for GitHub. You have two options to get them:

### Regenerate the model yourself 
1. Open `Notebook/Health_Triage_Group_1.ipynb` in Google Colab
2. Run all cells from top to bottom (Steps 1–11). This will:
   - Download the dataset
   - Fine-tune GPT-2 on it (~5–10 minutes on a free Colab GPU)
   - Save the model into a folder called `gpt2-triage-final/`
3. Download that folder from Colab (zip it, then use `files.download(...)`)
4. Unzip it into your local project folder, so the structure looks like:

├── gpt2-triage-final/ ← place it here

├── app.py

├── symptom_vocab.json

└── README.md
