# 🛡️ Phishing Detection System (Hybrid AI Pipeline)

## 📌 Overview

This project builds a **hybrid phishing detection system** combining:

* **LLM-based reasoning (Qwen)**
* **ML-based pattern learning (DistilBERT – upcoming)**

The goal is to create a **robust, real-world phishing detector** that understands both:

* *why* an email is phishing (semantic reasoning)
* *how* phishing appears in data (patterns)

---

## 🧠 Step 1: Design Philosophy

Instead of relying on a single model, we use a **multi-signal architecture**:

```
Email
 ↓
Qwen (semantic reasoning)
 ↓
Structured features
 ↓
+ DistilBERT (text embeddings)
 ↓
+ URL features
 ↓
Final classifier
```

---

## 🤖 Step 2: Qwen Integration

We use **Qwen 1.5B** as a **feature extractor (NOT classifier)**.

### 🎯 Role of Qwen

* Extract **semantic signals** from text
* Capture phishing psychology:

  * urgency
  * fear/threat
  * impersonation
  * action requests

---

## 🧾 Step 3: Feature Schema (Final)

Qwen outputs structured JSON:

```json
{
  "intent": "urgent_action | reward | warning | normal",
  "manipulation": 0,
  "request_type": "login | payment | download | action | none",
  "impersonation": 0
}
```

---

## 🔢 Step 4: Feature Encoding

Converted into ML-ready vector:

* intent → one-hot (4)
* request_type → one-hot (5)
* manipulation → binary (1)
* impersonation → binary (1)

👉 Final vector = **11 dimensions**

---

## 🧠 Step 5: Prompt Engineering

### Key techniques used:

#### ✅ Strict JSON Output

* No extra text allowed
* Fully machine-readable

#### ✅ Hidden Reasoning

* Model "thinks" internally
* Outputs only structured data

#### ✅ Few-shot Learning

Added representative examples:

* urgency + login
* reward scam
* impersonation
* normal email

#### ✅ Strong Definitions

Explicit rules for:

* manipulation (urgency, fear, threat)
* impersonation (Amazon, bank, etc.)
* intent classification

---

## 🧹 Step 6: Preprocessing

Basic normalization applied:

* lowercase text
* remove extra whitespace

---

## 🔧 Step 7: Robust Engineering

### Implemented safeguards:

* ✅ JSON extraction using regex (non-greedy)
* ✅ fallback mechanism (never crashes)
* ✅ category normalization
* ✅ fixed schema contract

---

## ⚠️ Challenges Faced

### 1. JSON Parsing Issues

* Greedy regex → fixed using non-greedy matching

### 2. Model Weakness (0.5B)

* Poor generalization
* Fixed by upgrading to **Qwen 1.5B**

### 3. Numeric Scoring Failure

* Removed `suspicious_score`
* Delegated scoring to final ML model

### 4. f-string Errors

* Fixed `{}` → `{{}}` escaping in prompts

---

## 🚀 Current Status

✅ Qwen module fully working
✅ Structured feature extraction stable
✅ Ready for ML pipeline

---

## 🔜 Next Steps

### 1. DistilBERT Integration

* Extract embeddings (mean pooling)
* Capture text patterns

### 2. Feature Fusion

Combine:

* Qwen features
* DistilBERT embeddings
* URL features

### 3. Final Classifier

* Logistic Regression / XGBoost
* Outputs phishing probability

---

## 🧠 Key Insight

> Qwen explains *why* it's phishing
> DistilBERT learns *how* phishing looks
> Final model decides *if* it's phishing

---

## 🏁 Conclusion

This project implements a **hybrid AI system** combining:

* symbolic reasoning (LLM)
* statistical learning (ML)

👉 Result: more robust and real-world phishing detection

---
