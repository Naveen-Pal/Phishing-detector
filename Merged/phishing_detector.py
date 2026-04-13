import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from urllib.parse import urlparse, parse_qs
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForSequenceClassification,
    DistilBertPreTrainedModel, 
    DistilBertModel
)
from safetensors.torch import load_file

# ============================================================
# 1. TEXT DETECTION COMPONENT (BERT Multi-Task)
# ============================================================

class DistilBertMultiTaskClassifier(DistilBertPreTrainedModel):
    """Memory-optimized multi-head classifier from distilBERT.ipynb"""
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        
        self.intent_classifier = nn.Linear(config.hidden_size, 3)
        self.manipulation_classifier = nn.Linear(config.hidden_size, 2)
        self.request_type_classifier = nn.Linear(config.hidden_size, 4)
        self.impersonation_classifier = nn.Linear(config.hidden_size, 2)
        
        self.config.use_cache = False

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        pooled_output = outputs[0][:, 0]
        pooled_output = self.dropout(pooled_output)

        return {
            "intent": self.intent_classifier(pooled_output),
            "manipulation": self.manipulation_classifier(pooled_output),
            "request_type": self.request_type_classifier(pooled_output),
            "impersonation": self.impersonation_classifier(pooled_output),
        }

# ============================================================
# 2. STATIC URL ANALYSIS COMPONENT (from static_url.py)
# ============================================================

def get_static_phishing_indicators(url):
    if not url.startswith("http"):
        url = "http://" + url

    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full = url.lower()

    domain_parts = host.split(".")
    tld = domain_parts[-1] if len(domain_parts) > 1 else ""
    subdomains = domain_parts[:-2] if len(domain_parts) > 2 else []
    params = parse_qs(query)
    
    suspicious_tlds = ['tk','ml','ga','cf','gq','ru','cn','xyz','top','click','link','online','site','win','loan','review','download']

    def entropy(s):
        if not s: return 0
        prob = [s.count(c)/len(s) for c in set(s)]
        return -sum(p * math.log2(p) for p in prob)

    flags = {}
    if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host): flags["ip_as_domain"] = host
    if parsed.scheme == "http": flags["not_https"] = True
    if tld in suspicious_tlds: flags["suspicious_tld"] = tld
    if len(subdomains) > 2: flags["too_many_subdomains"] = len(subdomains)
    if any(b in ".".join(subdomains) for b in ["paypal","google","amazon","apple","bank"]): flags["brand_in_subdomain"] = True
    if "@" in full: flags["has_at_symbol"] = True
    if "//" in path: flags["double_slash_trick"] = True
    if re.search(r'0x[0-9a-f]+', full): flags["hex_obfuscation"] = True
    if re.search(r'%[0-9a-f]{2}', full): flags["url_encoding"] = True
    if any(ord(c) > 127 for c in host): flags["non_ascii_domain"] = True
    if len(full) > 100: flags["long_url"] = len(full)
    if any(k in params for k in ["url","redirect","next","goto","return"]): flags["redirect_param"] = True
    if re.search(r'\.(php|exe|zip|scr)$', path): flags["suspicious_file_extension"] = path
    if entropy(full) > 4.5: flags["high_entropy"] = round(entropy(full), 2)
    if full.count("http") > 1: flags["multiple_http"] = True
    if "https" in path or "https" in query: flags["fake_https_in_path"] = True

    return flags

# ============================================================
# 3. INTEGRATED PHISHING DETECTOR CLASS
# ============================================================

class PhishingDetector:
    def __init__(self, text_model_path=None, url_model_path=None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if text_model_path is None:
            text_model_path = os.path.join(project_root, "distilbert-phishing-model")
        if url_model_path is None:
            url_model_path = os.path.join(project_root, "phishing_model")
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing models on {self.device}...")

        # Load Text Model (Multi-Task)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        text_config = AutoConfig.from_pretrained(text_model_path)
        self.text_model = DistilBertMultiTaskClassifier(text_config)
        text_state_dict = load_file(os.path.join(text_model_path, "model.safetensors"))
        self.text_model.load_state_dict(text_state_dict)
        self.text_model.to(self.device).eval()

        # Load URL ML Model
        self.url_tokenizer = AutoTokenizer.from_pretrained(url_model_path)
        self.url_model = AutoModelForSequenceClassification.from_pretrained(url_model_path)
        self.url_model.to(self.device).eval()

        self.INTENT_LABELS = {0: "normal", 1: "phishing", 2: "suspicious"}
        self.REQ_LABELS = {0: "none", 1: "credentials", 2: "payment", 3: "personal_info"}
        self.URL_LABEL_MAP = {0: "legit_email", 1: "phish_email", 2: "legit_url", 3: "phish_url"}

    def _predict_text(self, text, channel="email"):
        combined_text = f"{channel}: {text}"
        inputs = self.text_tokenizer(combined_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)

        probs = {k: F.softmax(v, dim=1)[0] for k, v in outputs.items()}
        
        # Scoring logic from notebook
        score = (3 * probs["manipulation"][1] + 4 * probs["request_type"][1] + 
                 2 * probs["request_type"][2] + 1 * probs["request_type"][3] + 
                 2 * probs["impersonation"][1]).item()
        
        if "click" in text.lower() or "verify" in text.lower(): score += 2
        
        intent_idx = probs["intent"].argmax().item()
        intent_conf = probs["intent"][intent_idx].item()
        
        verdict = "LEGITIMATE"
        if intent_conf > 0.85 or score >= 6: verdict = "PHISHING"
        elif score >= 3: verdict = "SUSPICIOUS"

        return {
            "verdict": verdict,
            "risk_score": round(score, 2),
            "intent": self.INTENT_LABELS[intent_idx],
            "request_type": self.REQ_LABELS[probs["request_type"].argmax().item()],
            "manipulation": "yes" if probs["manipulation"].argmax().item() == 1 else "no",
            "impersonation": "yes" if probs["impersonation"].argmax().item() == 1 else "no"
        }

    def _predict_url_ml(self, url):
        inputs = self.url_tokenizer(url, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.url_model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]

        phish_score = (probs[1] + probs[3]).item() * 100
        best_idx = probs.argmax().item()
        
        verdict = "LEGITIMATE"
        if phish_score >= 60: verdict = "PHISHING"
        elif phish_score >= 30: verdict = "SUSPICIOUS"

        return {
            "verdict": verdict,
            "phish_probability": round(phish_score, 2),
            "prediction_type": self.URL_LABEL_MAP[best_idx]
        }

    def scan(self, text=None, url=None):
        results = {}
        if text:
            results["text_analysis"] = self._predict_text(text)
        
        if url:
            ml_res = self._predict_url_ml(url)
            static_idx = get_static_phishing_indicators(url)
            
            results["url_analysis"] = {
                "ml_prediction": ml_res,
                "static_indicators": static_idx if (ml_res["verdict"] != "LEGITIMATE" or static_idx) else {}
            }
            
            # Highlight reasons why it was found phishing (per user request)
            if ml_res["verdict"] == "PHISHING":
                results["url_analysis"]["why_phishing"] = list(static_idx.keys()) if static_idx else ["ML model detected high risk pattern"]

        return results

# ============================================================
# 4. DEMO BLOCK
# ============================================================

if __name__ == "__main__":
    detector = PhishingDetector()

    SAMPLE_TEXT = "Urgent: Your account has been suspended. Please click here to verify your identity: http://secure-login-bank.ru/verify"
    SAMPLE_URL = "http://secure-login-bank.ru/verify"

    print("\n--- Scanning Potential Phishing ---")
    report = detector.scan(text=SAMPLE_TEXT, url=SAMPLE_URL)

    import json
    print(json.dumps(report, indent=4))
