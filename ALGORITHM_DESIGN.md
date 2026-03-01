# Phishing Detection Algorithm - In-Depth Design

## Problem Statement
**Input:** Text message from `Message.txt`  
**Output:** Classification (phishing/benign) + confidence score (0-1) + explanation

---

## 1. Feature Engineering (Text → Numerical Features)

### 1.1 Lexical & Statistical Features
Extract surface-level patterns that distinguish phishing messages:

| Feature | Formula/Method | Rationale |
|---------|----------------|-----------|
| **Message length** | `len(text)` characters | Phishing often verbose or very short |
| **Word count** | `len(text.split())` | Density indicator |
| **Avg word length** | `sum(len(w) for w in words) / word_count` | Complex words = legitimate? |
| **Uppercase ratio** | `sum(1 for c in text if c.isupper()) / len(text)` | ALL CAPS = urgency tactic |
| **Digit ratio** | `sum(1 for c in text if c.isdigit()) / len(text)` | Account numbers, fake tracking IDs |
| **Punctuation density** | Count of `!?.,;:` per 100 chars | Excessive punctuation = alarm |
| **Special char ratio** | Non-alphanumeric except whitespace | Obfuscation attempts |
| **Consecutive caps** | Max run of uppercase letters | URGENT!!! patterns |
| **Exclamation count** | Count of `!` | Emotional manipulation |

### 1.2 Linguistic Features
Semantic and structural analysis:

| Feature | Method | Why It Matters |
|---------|--------|----------------|
| **Urgency keywords** | Match regex: `(urgent\|immediate\|act now\|expire\|suspend\|verify\|confirm\|limited time)` (case-insensitive) | Core phishing tactic |
| **Financial keywords** | Match: `(account\|bank\|payment\|credit card\|refund\|prize\|inheritance\|money transfer)` | Target user's finances |
| **Action verbs** | Count: `(click\|verify\|update\|confirm\|download\|open\|respond\|call)` | Directive language |
| **Credential requests** | Match: `(password\|username\|PIN\|SSN\|security question)` | Legitimate orgs don't ask via message |
| **Greeting type** | Check if starts with `Dear [Name]` vs. generic `Dear Customer` | Personalization indicator |
| **Typo density** | Use SymSpell or edit-distance on common words | Intentional obfuscation or foreign origin |
| **Brand mentions** | NER or keyword search for `(PayPal\|Amazon\|Netflix\|IRS\|DHL)` | Impersonation attempts |

### 1.3 Embedded URL Features (if URLs present)
Even in text-only messages, URLs can appear:

| Feature | Extraction | Phishing Indicator |
|---------|------------|-------------------|
| **Has URL?** | Regex: `http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+` | Boolean flag |
| **URL count** | Number of links | Multiple redirects suspicious |
| **IP-based host** | Check if domain is `http://192.168...` or `http://10.0...` | Direct IPs = phishing |
| **Domain mismatch** | Extract display text vs. actual href (if HTML) | `Click PayPal.com` → `http://evil.com` |
| **Shortened URL** | Match: `(bit.ly\|tinyurl\|goo.gl)` | Obfuscates true destination |
| **Suspicious TLD** | Domain ends in `.tk`, `.ml`, `.ga`, `.cf`, `.gq` (free domains) | High phishing rate |
| **Domain age** | WHOIS lookup (if implementing) | Newly registered = suspicious |

### 1.4 Advanced NLP Features (Optional Deep Learning)
For transformer models:

- **Contextualized embeddings:** Feed entire text to BERT/RoBERTa → 768-dim vector → classifier head
- **Semantic similarity:** Compare message embedding to known phishing templates using cosine similarity
- **Named Entity Recognition (NER):** Extract ORG/PERSON/GPE entities; check if legitimate brand mentioned with suspicious action

---

## 2. ML Model Selection & Justification

### Approach A: Traditional ML (Baseline - Recommended Start)

#### Model: **Gradient Boosted Trees (XGBoost or LightGBM)**

**Why XGBoost?**
- **Handles mixed features:** Combines binary (has_url), categorical (urgency_score), continuous (length) seamlessly
- **Non-linear interactions:** Automatically learns "if urgency_keyword AND financial_term AND short_message → phishing"
- **Interpretability:** Feature importance scores via SHAP (Shapley values)
- **Robust to imbalance:** Use `scale_pos_weight` for skewed phishing/benign ratios
- **Fast inference:** <10ms per message on CPU

**Architecture:**
```
Input: Feature vector [length, caps_ratio, urgency_score, ..., has_url] (30-50 dims)
↓
XGBoost with 100-500 trees, max_depth=6, learning_rate=0.05
↓
Output: Probability P(phishing | features) ∈ [0,1]
```

**Training Procedure:**
1. **Data prep:** 
   - Extract features from all labeled messages → CSV/DataFrame
   - Train/val/test split: 70%/15%/15% (stratified by label)
   
2. **Hyperparameter tuning (GridSearch or Bayesian):**
   - `n_estimators`: [100, 300, 500]
   - `max_depth`: [4, 6, 8]
   - `learning_rate`: [0.01, 0.05, 0.1]
   - `subsample`: [0.8, 1.0]
   - `colsample_bytree`: [0.8, 1.0]
   - `scale_pos_weight`: `count(benign) / count(phishing)` if imbalanced

3. **Training loop:**
   ```python
   import xgboost as xgb
   from sklearn.metrics import roc_auc_score, precision_recall_curve
   
   dtrain = xgb.DMatrix(X_train, label=y_train)
   dval = xgb.DMatrix(X_val, label=y_val)
   
   params = {
       'objective': 'binary:logistic',
       'eval_metric': ['auc', 'logloss'],
       'max_depth': 6,
       'eta': 0.05,
       'subsample': 0.8,
       'colsample_bytree': 0.8,
       'scale_pos_weight': pos_weight
   }
   
   model = xgb.train(
       params, 
       dtrain, 
       num_boost_round=300,
       evals=[(dval, 'val')],
       early_stopping_rounds=20,
       verbose_eval=10
   )
   ```

4. **Threshold tuning:**
   - Plot precision-recall curve on validation set
   - Choose threshold where `precision ≥ 0.95` (minimize false positives)
   - Typical: threshold = 0.6-0.8 for phishing classification

**Pros:**
- Interpretable via SHAP
- Works well with small datasets (500-5000 labeled samples)
- Easy to update incrementally

**Cons:**
- Requires manual feature engineering
- Misses deep semantic understanding (sarcasm, context)

---

### Approach B: Deep Learning (Advanced)

#### Model: **Fine-tuned Transformer (DistilBERT or RoBERTa)**

**Why Transformers?**
- **Semantic understanding:** Captures "Your account will be suspended unless you verify" intent without keyword matching
- **Context-aware:** Handles paraphrases, misspellings, language variations
- **Transfer learning:** Pretrained on billions of tokens, fine-tune on 1000-10000 phishing samples

**Architecture:**
```
Input: Raw text string "Dear user, your account has been compromised..."
↓
Tokenizer → [CLS] dear user , your account ... [SEP] → input IDs
↓
DistilBERT (66M params) or RoBERTa-base (125M params)
↓
[CLS] token embedding (768-dim)
↓
Dropout(0.1) → Linear(768 → 2) → Softmax
↓
Output: [P(benign), P(phishing)]
```

**Training Procedure:**
1. **Setup:**
   ```python
   from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
   
   model_name = "distilbert-base-uncased"  # or "roberta-base"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
   ```

2. **Data preprocessing:**
   ```python
   def tokenize(batch):
       return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)
   
   train_dataset = train_df.map(tokenize, batched=True)
   ```

3. **Fine-tuning config:**
   - Learning rate: 2e-5 (critical - transformers sensitive)
   - Batch size: 16 (depends on GPU memory)
   - Epochs: 3-5 (too many = overfitting)
   - Warmup steps: 500
   - Weight decay: 0.01

4. **Training:**
   ```python
   from transformers import TrainingArguments, Trainer
   
   training_args = TrainingArguments(
       output_dir='./results',
       evaluation_strategy='epoch',
       learning_rate=2e-5,
       per_device_train_batch_size=16,
       num_train_epochs=4,
       weight_decay=0.01,
       logging_steps=100,
       save_strategy='epoch',
       load_best_model_at_end=True,
       metric_for_best_model='f1'
   )
   
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=val_dataset,
       compute_metrics=compute_metrics  # precision, recall, F1
   )
   
   trainer.train()
   ```

**Pros:**
- Best accuracy (typically 92-97% F1 on phishing datasets)
- Handles adversarial paraphrases
- Multilingual support (XLM-RoBERTa)

**Cons:**
- Requires GPU for training (4-12 hours on single GPU)
- Inference slower (50-200ms per message)
- Needs 1000+ labeled samples minimum
- Black-box (harder to explain predictions)

---

### Approach C: Hybrid (Recommended for Production)

Combine both approaches:

1. **Rule engine (instant):** Check for:
   - Exact match to known phishing templates (hash-based)
   - Blacklisted phrases: "verify your account immediately"
   - Suspicious patterns: `password` + `expires` + `click here`
   
   → If rule triggers: `score = 1.0, label = "phishing", reason = "Rule: password_urgency_match"`

2. **Traditional ML (fast):** XGBoost on lexical features
   → Get `score_ml` ∈ [0,1]

3. **Deep learning (accurate):** Transformer on full text
   → Get `score_dl` ∈ [0,1]

4. **Ensemble decision:**
   ```python
   if rule_triggered:
       final_score = 1.0
   else:
       final_score = 0.3 * score_ml + 0.7 * score_dl  # Weight DL higher
   
   if final_score > 0.75:
       label = "phishing"
   elif final_score > 0.4:
       label = "suspicious"
   else:
       label = "benign"
   ```

---

## 3. Decision Logic & Thresholds

### 3.1 Scoring Function

**Final score calculation:**
```python
def compute_final_score(rule_hits, ml_proba, dl_proba=None, weights=None):
    """
    Args:
        rule_hits: List of triggered rule IDs
        ml_proba: XGBoost output probability [0,1]
        dl_proba: Transformer output probability [0,1] (optional)
        weights: {'rule': 1.0, 'ml': 0.3, 'dl': 0.7}
    
    Returns:
        final_score: float [0,1]
        explanation: dict with contributions
    """
    if rule_hits:
        return 1.0, {"reason": "High-confidence rule triggered", "rules": rule_hits}
    
    if dl_proba is not None:
        # Hybrid: weighted ensemble
        final = weights['ml'] * ml_proba + weights['dl'] * dl_proba
        explanation = {
            "ml_contribution": ml_proba * weights['ml'],
            "dl_contribution": dl_proba * weights['dl']
        }
    else:
        # ML-only
        final = ml_proba
        explanation = {"ml_score": ml_proba}
    
    return final, explanation
```

### 3.2 Classification Thresholds

| Threshold | Label | Action |
|-----------|-------|--------|
| ≥ 0.80 | **Phishing (High Confidence)** | Block message, warn user |
| 0.50 - 0.79 | **Suspicious** | Flag for review, show warning |
| < 0.50 | **Benign** | Allow message |

**Tuning strategy:**
- **High-security scenario:** Lower threshold to 0.60 (catch more phishing, more false positives)
- **User-friendly:** Raise to 0.85 (fewer false alarms, miss some attacks)
- Use validation set to find optimal point on precision-recall curve

### 3.3 Explainability

For each classification, return:

```json
{
  "score": 0.87,
  "label": "phishing",
  "confidence": "high",
  "top_indicators": [
    {"feature": "urgency_keywords", "contribution": 0.35, "value": 3},
    {"feature": "credential_request", "contribution": 0.28, "value": true},
    {"feature": "suspicious_url", "contribution": 0.15, "value": "http://amaz0n.secure-login.tk"},
    {"feature": "uppercase_ratio", "contribution": 0.09, "value": 0.42}
  ],
  "triggered_rules": ["rule_005: password + urgent + verify"],
  "model_used": "xgboost_v2 + distilbert_finetuned"
}
```

**Implementation:**
- **XGBoost:** Use `model.get_score(importance_type='gain')` → sort by contribution
- **Transformers:** Use attention weights visualization or LIME (Local Interpretable Model-agnostic Explanations)

---

## 4. Training Data Requirements

### 4.1 Dataset Composition

| Category | Examples | Source |
|----------|----------|--------|
| **Phishing emails** | 5,000 | PhishTank API, Kaggle phishing datasets |
| **Spam (not phishing)** | 3,000 | SpamAssassin, Enron spam corpus |
| **Legitimate emails** | 7,000 | Enron legit corpus, personal corpus (anonymized) |
| **Edge cases** | 500 | Legitimate urgent messages (password resets), promotional emails |

**Total:** ~15,000 labeled messages

### 4.2 Data Augmentation
Expand training set by:
- **Synonym replacement:** "urgent" → "immediate", "account" → "profile"
- **Paraphrasing:** Use GPT/T5 to rewrite messages preserving intent
- **Back-translation:** English → French → English for variation
- **Adversarial examples:** Apply known evasion tactics (l33tsp34k, homoglyphs)

### 4.3 Labeling Protocol
- Binary labels: `0 = benign`, `1 = phishing`
- Optional multi-class: `{benign, spam, phishing, suspicious}`
- Require 2 annotators per message, resolve conflicts via majority vote

---

## 5. Evaluation Metrics

### 5.1 Core Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Precision** | TP / (TP + FP) | ≥ 0.95 (minimize false alarms) |
| **Recall** | TP / (TP + FN) | ≥ 0.90 (catch most phishing) |
| **F1 Score** | 2 × (P × R) / (P + R) | ≥ 0.92 |
| **ROC-AUC** | Area under ROC curve | ≥ 0.96 |
| **False Positive Rate** | FP / (FP + TN) | ≤ 0.02 (2%) |

### 5.2 Confusion Matrix Analysis

```
                Predicted
              Benign  Phishing
Actual Benign   TN      FP      ← Minimize FP (user frustration)
       Phish    FN      TP      ← Minimize FN (security risk)
```

**Cost-weighted evaluation:**
- Cost of FN (missed phishing) = 10× cost of FP (false alarm)
- Optimize for: `minimize(10*FN + FP)`

### 5.3 Adversarial Testing
Test against known evasion techniques:
- **Typosquatting:** "PayPaI" (I instead of l)
- **Homoglyphs:** "Αmazon" (Greek Alpha)
- **Obfuscation:** "p@ssw0rd" instead of "password"
- **Image-based text:** (if supporting HTML)

---

## 6. Implementation Pipeline

### 6.1 Training Phase
```
[Labeled messages] 
    ↓
[Feature extraction] → [TF-IDF + lexical features]
    ↓
[Train/val/test split 70/15/15]
    ↓
[Train XGBoost] → [Save model.xgb]
    ↓
[Train DistilBERT] → [Save model/]
    ↓
[Threshold tuning on validation set]
    ↓
[Final evaluation on test set]
    ↓
[Model registry with versioning]
```

### 6.2 Inference Phase
```
[Read Message.txt]
    ↓
[Preprocess text: lowercase, tokenize]
    ↓
[Check rules] → If hit → [Return score=1.0, reason]
    ↓ (else)
[Extract features] → [30-dim vector]
    ↓
[XGBoost predict] → [score_ml]
    ↓
[DistilBERT predict] → [score_dl]
    ↓
[Ensemble: 0.3*ml + 0.7*dl]
    ↓
[Apply threshold] → [Label: phishing/suspicious/benign]
    ↓
[Generate explanation] → [Top 5 features + SHAP values]
    ↓
[Return JSON result]
```

---

## 7. Recommended Starting Point

**Phase 1 (Prototype - 1 week):**
1. Collect 1000 labeled messages (500 phishing, 500 benign)
2. Implement lexical feature extractor (20 features)
3. Train XGBoost baseline
4. Build simple inference script: `python detect.py --input Message.txt`
5. Evaluate on 200-message test set

**Phase 2 (Enhancement - 2 weeks):**
1. Add rule engine (10 high-precision rules)
2. Expand dataset to 5000 messages
3. Fine-tune DistilBERT
4. Implement ensemble logic
5. Add SHAP explainability

**Phase 3 (Production - 1 month):**
1. Build REST API with FastAPI
2. Add model monitoring (data drift detection)
3. Implement feedback loop for corrections
4. Deploy with Docker + Redis cache
5. Create admin dashboard

---

## 8. Technology Stack

**Core ML:**
- Python 3.9+
- scikit-learn 1.3+
- XGBoost 2.0+ or LightGBM 4.0+
- Transformers 4.35+ (Hugging Face)
- PyTorch 2.0+ (for transformers)

**Feature Engineering:**
- NLTK or spaCy for NLP
- pandas for data manipulation
- regex for pattern matching

**Evaluation:**
- scikit-learn metrics
- SHAP for explainability
- matplotlib/seaborn for visualization

**Production:**
- FastAPI or Flask
- Docker
- Redis (feature cache)
- PostgreSQL (label storage)

---

## 9. Expected Performance

Based on literature and similar systems:

| Model | Precision | Recall | F1 | Inference Time |
|-------|-----------|--------|-------|----------------|
| Rule-based only | 0.98 | 0.65 | 0.78 | <1ms |
| XGBoost | 0.94 | 0.89 | 0.91 | 5-10ms |
| DistilBERT | 0.96 | 0.94 | 0.95 | 50-100ms |
| Hybrid ensemble | 0.96 | 0.93 | 0.94 | 50-100ms |

**Target for production:** F1 ≥ 0.93, FPR ≤ 2%, latency < 200ms

---

## Summary

**Recommended algorithm:**
1. **Baseline:** XGBoost on 30+ lexical/linguistic features → F1 ~0.91
2. **Advanced:** Fine-tuned DistilBERT on raw text → F1 ~0.95
3. **Production:** Rule engine + XGBoost + DistilBERT ensemble → F1 ~0.94

**Decision logic:**
- Rules triggered → Immediate phishing classification
- Ensemble score ≥ 0.80 → Phishing
- Score 0.50-0.79 → Suspicious (flag)
- Score < 0.50 → Benign

**Next step:** Collect labeled dataset or use existing corpus (PhishTank, Kaggle) and implement XGBoost baseline.
