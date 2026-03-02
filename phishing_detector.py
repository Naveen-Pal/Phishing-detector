#!/usr/bin/env python3
"""
Phishing Detection System
Uses rule-based checks, text analysis, and optional ML model.
"""

import re
import requests
from typing import Dict, Tuple, Optional, List
from urllib.parse import urlparse

print("Initializing Phishing Detection System...")

# Try to import transformers
try:
    from transformers import pipeline
    MODELS_AVAILABLE = True
    print("Loading ML model...")
    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    print("✓ Model loaded!")
except ImportError:
    print("⚠️  Running in rule-based mode (install transformers for ML)")
    MODELS_AVAILABLE = False
except Exception as e:
    print(f"⚠️  Model loading failed: {e}")
    MODELS_AVAILABLE = False

print("=" * 70 + "\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Core phishing detection rules
PHISHING_RULES = {
    'urgent_credential': r'(?i)(urgent|immediate|suspended?).{0,30}(password|account|login)',
    'credential_request': r'(?i)(verify|confirm|update).{0,30}(password|ssn|credit\s*card)',
    # 'typosquat': r'(?i)(paypa1|paypai|amaz0n|g00gle|faceb00k|micros0ft)',
    # 'urgency_financial': r'(?i)(account|payment).{0,30}(suspend|expir|lock)',
    # 'prize_scam': r'(?i)(won|winner|prize).{0,30}(\$|\d+|money)',
}

# ============================================================================
# URL EXTRACTION & ANALYSIS FUNCTIONS
# ============================================================================

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text using regex"""
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    return urls


def fetch_url_content(url: str, timeout: int = 5) -> Optional[str]:
    """Fetch content from URL using requests"""
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return response.text[:2000]  # Return first 2000 chars
    except requests.RequestException as e:
        return None


def analyze_url_with_zeroshot(url: str, content: Optional[str]) -> Dict:
    """Use zero-shot classification to analyze URL and fetched content"""
    if not MODELS_AVAILABLE:
        return {'ml_score': 0.5, 'error': 'Model not available'}
    
    try:
        # Combine URL and content for analysis
        analysis_text = f"URL: {url}\nContent: {content if content else 'Could not fetch'}"
        
        result = zero_shot_classifier(
            analysis_text[:512],
            candidate_labels=["phishing attempt", "legitimate website"],
            multi_label=False
        )
        phishing_idx = result['labels'].index("phishing attempt")
        phishing_score = result['scores'][phishing_idx]
        
        return {
            'ml_score': phishing_score,
            'labels': result['labels'],
            'scores': result['scores']
        }
    except Exception as e:
        return {'ml_score': 0.5, 'error': str(e)}


def analyze_urls_in_message(text: str, verbose: bool = True) -> Dict:
    """Extract and analyze all URLs in message"""
    urls = extract_urls(text)
    
    if not urls:
        return {
            'found_urls': False,
            'urls': []
        }
    
    url_results = []
    for url in urls:
        if verbose:
            print(f"  Analyzing URL: {url}")
        
        content = fetch_url_content(url, timeout=5)
        if verbose and content:
            print(f"    ✓ Fetched content ({len(content)} chars)")
        elif verbose:
            print(f"    ⚠️  Could not fetch content")
        
        zeroshot_result = analyze_url_with_zeroshot(url, content)
        
        url_results.append({
            'url': url,
            'fetched': content is not None,
            'content_preview': content[:200] if content else None,
            'phishing_score': zeroshot_result.get('ml_score', 0.5),
            'error': zeroshot_result.get('error')
        })
    
    return {
        'found_urls': True,
        'urls': url_results,
        'max_phishing_score': max([u['phishing_score'] for u in url_results])
    }

# ============================================================================
# TEXT ANALYSIS FUNCTIONS
# ============================================================================

def check_rules(text: str) -> Tuple[bool, Optional[str]]:
    """Check message against phishing rules"""
    for rule_name, pattern in PHISHING_RULES.items():
        if re.search(pattern, text):
            return True, rule_name
    return False, None


def analyze_text_features(text: str) -> Dict:
    """Analyze text for phishing indicators"""
    score = 0.0
    indicators = []

    # Check for urgency
    urgency_words = ['urgent', 'immediate', 'now', 'expire', 'suspended']
    urgency_count = sum(1 for word in urgency_words if word in text.lower())
    if urgency_count > 0:
        score += 0.25
        indicators.append('urgency language')

    # Check for credential requests
    if re.search(r'(?i)(password|ssn|credit card)', text):
        score += 0.3
        indicators.append('credential request')

    # Check for excessive punctuation
    if text.count('!') > 2:
        score += 0.15
        indicators.append('excessive punctuation')

    # Check for generic greetings
    if re.search(r'(?i)dear (customer|user|member)', text):
        score += 0.15
        indicators.append('generic greeting')

    return {
        'score': round(min(score, 1.0), 3),
        'indicators': indicators
    }


def get_ml_score(text: str) -> float:
    """Get phishing probability from ML model"""
    if not MODELS_AVAILABLE:
        return 0.5
    try:
        result = zero_shot_classifier(
            text[:512],
            candidate_labels=["phishing attempt", "legitimate message"],
            multi_label=False
        )
        phishing_idx = result['labels'].index("phishing attempt")
        return result['scores'][phishing_idx]
    except Exception as e:
        print(f"  Warning: ML classifier error: {e}")
        return 0.5

# ============================================================================
# MAIN CLASSIFICATION FUNCTION
# ============================================================================

def classify_message(text: str, verbose: bool = True) -> Dict:
    """Phishing detection using rules, text analysis, URL analysis, and ML."""
    if not text or len(text.strip()) < 10:
        return {
            'label': 'invalid',
            'score': 0.0,
            'confidence': 'n/a',
            'reason': 'Message too short'
        }

    if verbose:
        print("  Analyzing message...")

    # Step 1: Check rules (high confidence patterns)
    rule_hit, rule_name = check_rules(text)
    if rule_hit:
        return {
            'label': 'phishing',
            'score': 1.0,
            'confidence': 'high',
            'reason': f'Phishing pattern detected: {rule_name}'
        }

    # Step 2: Analyze URLs in message
    url_analysis = analyze_urls_in_message(text, verbose=verbose)
    url_score = 0.0
    url_indicators = []
    
    if url_analysis['found_urls']:
        url_score = url_analysis['max_phishing_score']
        if url_score >= 0.5:
            url_indicators.append(f"suspicious URLs detected (score: {url_score:.3f})")

    # Step 3: Analyze text features
    text_analysis = analyze_text_features(text)
    text_score = text_analysis['score']

    # Step 4: ML model
    ml_score = get_ml_score(text) if MODELS_AVAILABLE else 0.5

    # Step 5: Combine scores (prioritize URL analysis if URLs found)
    if url_analysis['found_urls'] and url_score >= 0.5:
        final_score = url_score
    elif MODELS_AVAILABLE:
        final_score = max(ml_score, text_score)
    else:
        final_score = text_score

    # Determine label
    if final_score >= 0.5:
        label = 'phishing'
        confidence = 'high' if final_score >= 0.85 else 'medium'
    elif final_score >= 0.2:
        label = 'suspicious'
        confidence = 'medium'
    else:
        label = 'benign'
        confidence = 'high' if final_score <= 0.25 else 'medium'

    # Build reason
    reasons = []
    if url_indicators:
        reasons.extend(url_indicators)
    if text_analysis['indicators']:
        reasons.append(f"text: {', '.join(text_analysis['indicators'])}")
    if MODELS_AVAILABLE and ml_score > 0.6:
        reasons.append('ML model flagged as phishing')

    return {
        'label': label,
        'score': round(final_score, 3),
        'confidence': confidence,
        'reason': '; '.join(reasons) if reasons else 'no major concerns',
        'details': {
            'text_score': round(text_score, 3),
            'ml_score': round(ml_score, 3) if MODELS_AVAILABLE else None,
            'url_score': round(url_score, 3) if url_analysis['found_urls'] else None,
            'urls': url_analysis if url_analysis['found_urls'] else None,
        }
    }

# ============================================================================
# FILE INPUT & OUTPUT
# ============================================================================

def detect_phishing_from_file(filepath: str = 'Message.txt', verbose: bool = True) -> Dict:
    """Read message from file and classify"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            message = f.read().strip()

        if not message:
            return {'error': 'Message file is empty', 'filepath': filepath}

        if verbose:
            print(f"\nAnalyzing message from: {filepath}")
            print(f"Length: {len(message)} characters\n")

        return classify_message(message, verbose=verbose)

    except FileNotFoundError:
        return {
            'error': f'File not found: {filepath}',
            'suggestion': 'Create the file with a message to analyze'
        }
    except Exception as e:
        return {'error': f'Unexpected error: {str(e)}'}


def print_result(result: Dict, show_details: bool = True):
    """Pretty print classification result"""
    print("\n" + "=" * 70)
    print("PHISHING DETECTION RESULT")
    print("=" * 70)

    if 'error' in result:
        print(f"\n❌ ERROR: {result['error']}")
        if 'suggestion' in result:
            print(f"💡 {result['suggestion']}")
    else:
        print(f"\nLabel:      {result['label'].upper()}")
        print(f"Score:      {result['score']:.3f}")
        print(f"Confidence: {result['confidence'].upper()}")
        print(f"Reason:     {result['reason']}")

        if show_details and 'details' in result:
            details = result['details']
            print(f"\n--- Details ---")
            print(f"Text Score:  {details['text_score']:.3f}")
            if details['ml_score'] is not None:
                print(f"ML Score:    {details['ml_score']:.3f}")
            if details['url_score'] is not None:
                print(f"URL Score:   {details['url_score']:.3f}")
            
            # Print URL analysis details
            if details['urls'] and details['urls']['found_urls']:
                print(f"\n--- URLs Found ({len(details['urls']['urls'])}) ---")
                for url_result in details['urls']['urls']:
                    print(f"  URL: {url_result['url']}")
                    print(f"    Phishing Score: {url_result['phishing_score']:.3f}")
                    print(f"    Fetched: {'✓' if url_result['fetched'] else '✗'}")
                    if url_result['content_preview']:
                        print(f"    Preview: {url_result['content_preview'][:100]}...")
                    if url_result['error']:
                        print(f"    Error: {url_result['error']}")

        print("\n" + "=" * 70)

        if result['label'] == 'phishing':
            print("⚠️  WARNING: This message appears to be PHISHING")
            print("   DO NOT click links or provide personal information!")
        elif result['label'] == 'suspicious':
            print("⚡ CAUTION: This message is SUSPICIOUS")
            print("   Verify sender before taking action")
        elif result['label'] == 'benign':
            print("✓ This message appears to be BENIGN")

    print("=" * 70 + "\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else 'message.txt'
    result = detect_phishing_from_file(filepath, verbose=True)
    print_result(result, show_details=True)

    if 'error' in result:
        sys.exit(1)
    elif result.get('label') == 'phishing':
        sys.exit(2)
    else:
        sys.exit(0)
