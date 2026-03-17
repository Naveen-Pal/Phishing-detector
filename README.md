# Phishing Detector

A Python-based phishing detection system that identifies phishing emails and messages using a multi-layered approach combining rule-based detection, text feature analysis, URL analysis, and machine learning.

## Features

- **Rule-based detection** — instant, high-precision pattern matching for known phishing patterns
- **Text feature analysis** — detects urgency language, credential requests, excessive punctuation, and generic greetings
- **URL analysis** — extracts and classifies URLs found in messages using ML
- **Machine learning classification** — zero-shot classification via `facebook/bart-large-mnli` (no training required)
- **Hybrid ensemble scoring** — combines all signals for a final phishing score and confidence level
- **Detailed output** — per-component score breakdown with human-readable reasons

## Project Structure

```
Phishing-detector/
├── phishing_detector.py          # Main application
├── requirements.txt              # Python dependencies
├── message.txt                   # Sample message for testing
└── PRETRAINED_MODEL_APPROACH.md  # Technical deep-dive & model guide
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.6+ |
| NLP Framework | Hugging Face Transformers ≥ 4.35.0 |
| ML Model | `facebook/bart-large-mnli` (zero-shot) |
| Deep Learning | PyTorch ≥ 2.0.0 |
| HTTP Requests | requests ≥ 2.28.0 |

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

> **Note:** On first run the `facebook/bart-large-mnli` model (~1.6 GB) will be downloaded automatically and cached in `~/.cache/huggingface/`.

## 🚀 Usage

1. Put the message you want to analyze in `message.txt` (or any file of your choice):
   ```bash
   echo "URGENT: Your account has been suspended. Verify your password immediately at http://example.com/login" > message.txt
   ```

2. Run the detector:
   ```bash
   python phishing_detector.py               # uses message.txt by default
   python phishing_detector.py path/to/file  # custom file path
   ```

### Example Outputs

**Phishing detected**
```
⚠️  PHISHING DETECTED ⚠️
Label:      PHISHING
Score:      1.000
Confidence: HIGH
Reason:     Phishing pattern detected: urgent_credential
```

**Suspicious message**
```
⚡ SUSPICIOUS MESSAGE
Label:      SUSPICIOUS
Score:      0.650
Confidence: MEDIUM
Reason:     text: urgency language; ML model flagged as phishing
```

**Benign message**
```
✓ Message appears safe
Label:      BENIGN
Score:      0.150
Confidence: HIGH
Reason:     no major concerns
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Benign or suspicious message |
| `1` | Error occurred |
| `2` | Phishing detected |
