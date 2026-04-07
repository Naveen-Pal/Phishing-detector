#!/usr/bin/env python3
"""
Phishing Detection System
Uses rule-based checks, text analysis, and optional ML model.
"""

import re
import requests
from email import policy
from email.parser import BytesParser
from typing import Dict, Tuple, Optional, List

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

def parse_email_input(raw_text: str) -> Tuple[str, Dict[str, str]]:
    """Parse simple header-style text input into body and sender metadata."""
    if not raw_text:
        return "", {}

    lines = raw_text.splitlines()
    headers: Dict[str, str] = {}
    body_start = 0
    saw_header = False

    for idx, line in enumerate(lines):
        if line.strip() == "":
            body_start = idx + 1
            break

        header_match = re.match(r'^([A-Za-z][A-Za-z0-9\-]*):\s*(.*)$', line)
        if not header_match:
            return raw_text.strip(), {}

        saw_header = True
        key = header_match.group(1).lower().replace('-', '_')
        headers[key] = header_match.group(2).strip()
    else:
        body_start = len(lines)

    body = "\n".join(lines[body_start:]).strip() if saw_header else raw_text.strip()
    sender_info = {
        'from': headers.get('from', ''),
        'reply_to': headers.get('reply_to', ''),
        'return_path': headers.get('return_path', ''),
        'subject': headers.get('subject', ''),
    }
    return body if body else raw_text.strip(), sender_info


def _strip_html_tags(html: str) -> str:
    """Convert basic HTML body content to readable plain text."""
    if not html:
        return ''
    html = re.sub(r'(?is)<(script|style).*?>.*?</\1>', ' ', html)
    html = re.sub(r'(?is)<br\s*/?>', '\n', html)
    html = re.sub(r'(?is)</p\s*>', '\n', html)
    html = re.sub(r'(?is)<[^>]+>', ' ', html)
    html = re.sub(r'\s+', ' ', html)
    return html.strip()


def parse_eml_input(raw_bytes: bytes) -> Tuple[str, Dict[str, str]]:
    """Parse .eml bytes and return extracted body text + sender metadata."""
    if not raw_bytes:
        return "", {}

    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    except Exception:
        fallback_text = raw_bytes.decode('utf-8', errors='replace')
        return parse_email_input(fallback_text)

    sender_info = {
        'from': str(msg.get('From', '') or '').strip(),
        'reply_to': str(msg.get('Reply-To', '') or '').strip(),
        'return_path': str(msg.get('Return-Path', '') or '').strip(),
        'subject': str(msg.get('Subject', '') or '').strip(),
    }

    plain_parts: List[str] = []
    html_parts: List[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get_content_disposition() or '').lower()
            if disposition == 'attachment':
                continue
            try:
                content = part.get_content()
            except Exception:
                continue

            if not isinstance(content, str) or not content.strip():
                continue

            if content_type == 'text/plain':
                plain_parts.append(content.strip())
            elif content_type == 'text/html':
                cleaned = _strip_html_tags(content)
                if cleaned:
                    html_parts.append(cleaned)
    else:
        try:
            content = msg.get_content()
        except Exception:
            content = ''

        if isinstance(content, str) and content.strip():
            if msg.get_content_type() == 'text/html':
                cleaned = _strip_html_tags(content)
                if cleaned:
                    html_parts.append(cleaned)
            else:
                plain_parts.append(content.strip())

    body = '\n\n'.join(plain_parts).strip() if plain_parts else '\n\n'.join(html_parts).strip()

    if not body:
        fallback_text = raw_bytes.decode('utf-8', errors='replace').strip()
        if fallback_text:
            body, fallback_sender = parse_email_input(fallback_text)
            for key, value in fallback_sender.items():
                if not sender_info.get(key):
                    sender_info[key] = value

    return body, sender_info


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


def get_ml_score(text: str, sender_info: Optional[Dict[str, str]] = None) -> float:
    """Get phishing probability from ML model"""
    if not MODELS_AVAILABLE:
        return 0.5
    try:
        sender_context = ''
        if sender_info:
            sender_context = (
                f"From: {sender_info.get('from', '')}\n"
                f"Reply-To: {sender_info.get('reply_to', '')}\n"
                f"Return-Path: {sender_info.get('return_path', '')}\n"
                f"Subject: {sender_info.get('subject', '')}\n"
            )

        model_input = f"{sender_context}\nMessage: {text}" if sender_context.strip() else text
        result = zero_shot_classifier(
            model_input[:512],
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

def classify_message(text: str, sender_info: Optional[Dict[str, str]] = None, verbose: bool = True) -> Dict:
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

    # Step 4: ML model (includes sender context when available)
    ml_score = get_ml_score(text, sender_info=sender_info) if MODELS_AVAILABLE else 0.5

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
        with open(filepath, 'rb') as f:
            raw_bytes = f.read()

        if not raw_bytes.strip():
            return {'error': 'Message file is empty', 'filepath': filepath}

        looks_like_eml = (
            filepath.lower().endswith('.eml')
            or b'\nMIME-Version:' in raw_bytes
            or (b'\nContent-Type:' in raw_bytes and b'\nFrom:' in raw_bytes)
        )

        if looks_like_eml:
            message, sender_info = parse_eml_input(raw_bytes)
        else:
            raw_content = raw_bytes.decode('utf-8', errors='replace')
            message, sender_info = parse_email_input(raw_content)

        if not message:
            return {'error': 'Message file is empty', 'filepath': filepath}

        if verbose:
            print(f"\nAnalyzing message from: {filepath}")
            print(f"Length: {len(message)} characters\n")
            if sender_info.get('from'):
                print(f"From: {sender_info['from']}")
            if sender_info.get('subject'):
                print(f"Subject: {sender_info['subject']}")
            if sender_info.get('from') or sender_info.get('subject'):
                print()

        return classify_message(message, sender_info=sender_info, verbose=verbose)

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
