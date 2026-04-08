from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

MODEL_ID  = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
MODEL_DIR = "./phishing_model"          # local save path

# Load from disk if already saved, otherwise download once and save
if os.path.exists(MODEL_DIR):
    print("Loading model from local cache...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
else:
    print("Downloading model for the first time...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    print(f"Model saved to '{MODEL_DIR}' — won't download again.")

model.eval()

# Corrected labels based on actual training dataset schema
LABEL_MAP = {
    0: "legitimate_email",  # safe email content
    1: "phishing_email",    # phishing via email body language
    2: "legitimate_url",    # safe URL
    3: "phishing_url",      # phishing via embedded URL
}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)[0].tolist()

    labeled        = {LABEL_MAP[i]: round(p * 100, 2) for i, p in enumerate(probs)}
    best           = max(labeled, key=labeled.get)
    phishing_score = round(probs[1] * 100 + probs[3] * 100, 2)
    legit_score    = round(probs[0] * 100 + probs[2] * 100, 2)

    if phishing_score >= 60:
        verdict = "PHISHING"
    elif phishing_score >= 30:
        verdict = "SUSPICIOUS"
    else:
        verdict = "LEGITIMATE"


    return {
        "verdict":        verdict,
        "prediction":     best,
        "confidence":     labeled[best],
        "phishing_score": phishing_score,
        "legit_score":    legit_score,
        "all_probs":      labeled,
    }


# --- Test it ---

phishing_email = """http://secure-paypa1-verify.com/login"""

legit_email = """google.com"""

# this model works for text as well as link
for label, text in [("Phishing email", phishing_email), ("Legit email", legit_email)]:
    result = predict(text)
    print(f"\n=== {label} ===")
    print(f"Verdict       : {result['verdict']}")
    print(f"Prediction    : {result['prediction']}")
    print(f"Confidence    : {result['confidence']}%")
    print(f"Phishing score: {result['phishing_score']}%")
    print(f"Legit score   : {result['legit_score']}%")
    print("All probs:")
    for k, v in result['all_probs'].items():
        print(f"  {k:<20} {v}%")
