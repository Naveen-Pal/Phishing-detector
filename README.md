# Phishing Detector

A Python-based, multi-layered phishing detection system combining powerful fine-tuned linguistic analysis (DistilBERT Multi-task) and hybrid URL evaluation (Machine Learning + Deep Static Analysis).

## Features

- **Text Feature Analysis (DistilBERT Multi-Task)** — A memory-optimized, multi-head classifier predicting four dimensions of text communication:
  - **Intent**: Normal, Suspicious, or Phishing
  - **Request Type**: Credentials, Payment, Personal Info, or None
  - **Manipulation**: Yes or No
  - **Impersonation**: Yes or No
- **URL Analysis (Machine Learning)** — Uses a fine-tuned sequence classification model to accurately score URLs for phishing probability.
- **Deep Static URL Analysis** — A robust rule-based engine that extracts 14+ static heuristics (e.g., suspicious TLDs, IP as domain, hexadecimal obfuscation, redirection parameters, high entropy, double slash tricks).
- **Hybrid Ensemble Scoring** — Intelligently integrates text insights and URL findings yielding a holistic verdict (`LEGITIMATE`, `SUSPICIOUS`, or `PHISHING`) and combined risk scores.
- **Detailed Output** — Outputs component score breakdowns, predictions, and human-readable reasoning based on both ML classification and static indicators.

## Project Structure

```text
├── Merged/
│   └── phishing_detector.py      # Main 
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| NLP Framework | Hugging Face Transformers ≥ 4.35.0 |
| Text ML Model | `DistilBERT` (Fine-tuned, Multi-task) |
| Deep Learning | PyTorch ≥ 2.0.0, safetensors |
| Data Processing | `datasets`, `pandas`, `scikit-learn` |
| Utilities | `urllib`, `re`, `math` |

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Naveen-Pal/Phishing-detector.git
   cd Phishing-detector
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   .venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

You can test the system using the `PhishingDetector` class inside `Merged/phishing_detector.py`. 

By default, executing the script runs the integrated demo block:

```bash
python Merged/phishing_detector.py
```

### Example Scanning Output

```json
{
    "text_analysis": {
        "verdict": "PHISHING",
        "risk_score": 6.8,
        "intent": "phishing",
        "request_type": "credentials",
        "manipulation": "yes",
        "impersonation": "yes"
    },
    "url_analysis": {
        "ml_prediction": {
            "verdict": "PHISHING",
            "phish_probability": 92.5,
            "prediction_type": "phish_url"
        },
        "static_indicators": {
            "suspicious_tld": "ru",
            "brand_in_subdomain": true,
            "suspicious_file_extension": "/verify"
        }
    }
}
```
