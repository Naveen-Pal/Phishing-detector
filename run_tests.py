#!/usr/bin/env python3
"""
Test Suite for Phishing Detection System
Runs all test cases and analyzes performance
"""

import json
import sys
from phishing_detector import classify_message

def load_test_cases(filepath='test_cases.json'):
    """Load test cases from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_single_test(test_case, show_details=False):
    """Run a single test case"""
    print(f"\n{'='*70}")
    print(f"Test #{test_case['id']}: {test_case['name']}")
    print(f"{'='*70}")
    print(f"Expected: {test_case['expected'].upper()}")
    print(f"\nMessage: {test_case['message'][:100]}{'...' if len(test_case['message']) > 100 else ''}")
    print(f"\nAnalyzing...")
    
    result = classify_message(test_case['message'], verbose=False)
    
    print(f"\nPredicted: {result['label'].upper()}")
    print(f"Score: {result['score']:.3f}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reason: {result['reason']}")
    
    # Check if correct
    correct = result['label'] == test_case['expected']
    
    if correct:
        print(f"\n✓ PASS")
    else:
        print(f"\n✗ FAIL - Expected {test_case['expected']}, got {result['label']}")
    
    if show_details and 'analysis' in result:
        print(f"\n--- Details ---")
        if 'text_lexical_score' in result['analysis']:
            print(f"Text Lexical: {result['analysis']['text_lexical_score']:.3f}")
        if 'spam_score' in result['analysis']:
            print(f"Spam Score: {result['analysis']['spam_score']:.3f}")
        if 'phishing_score' in result['analysis']:
            print(f"Phishing Score: {result['analysis']['phishing_score']:.3f}")
        if 'url_score' in result['analysis']:
            print(f"URL Score: {result['analysis']['url_score']:.3f}")
    
    return {
        'test_id': test_case['id'],
        'name': test_case['name'],
        'expected': test_case['expected'],
        'predicted': result['label'],
        'score': result['score'],
        'correct': correct,
        'result': result
    }

def analyze_results(test_results):
    """Analyze overall performance"""
    print(f"\n\n{'='*70}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*70}\n")
    
    total = len(test_results)
    correct = sum(1 for r in test_results if r['correct'])
    accuracy = correct / total
    
    print(f"Total Tests: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy*100:.1f}%\n")
    
    # Per-class analysis
    classes = ['phishing', 'suspicious', 'benign']
    
    for cls in classes:
        expected_cls = [r for r in test_results if r['expected'] == cls]
        if not expected_cls:
            continue
        
        tp = sum(1 for r in expected_cls if r['predicted'] == cls)
        fn = sum(1 for r in expected_cls if r['predicted'] != cls)
        
        all_predicted_cls = [r for r in test_results if r['predicted'] == cls]
        fp = sum(1 for r in all_predicted_cls if r['expected'] != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{cls.upper()}:")
        print(f"  Precision: {precision*100:.1f}%")
        print(f"  Recall: {recall*100:.1f}%")
        print(f"  F1-Score: {f1*100:.1f}%")
        print()
    
    # False positives and false negatives
    print("ERRORS:")
    
    false_positives = [r for r in test_results if r['expected'] == 'benign' and r['predicted'] in ['phishing', 'suspicious']]
    false_negatives = [r for r in test_results if r['expected'] == 'phishing' and r['predicted'] in ['benign', 'suspicious']]
    
    if false_positives:
        print(f"\nFalse Positives ({len(false_positives)}):")
        for r in false_positives:
            print(f"  - Test #{r['test_id']}: {r['name']}")
            print(f"    Predicted: {r['predicted']} (score: {r['score']:.3f})")
    
    if false_negatives:
        print(f"\nFalse Negatives ({len(false_negatives)}):")
        for r in false_negatives:
            print(f"  - Test #{r['test_id']}: {r['name']}")
            print(f"    Predicted: {r['predicted']} (score: {r['score']:.3f})")
    
    if not false_positives and not false_negatives:
        print("  None! Perfect classification.")
    
    print(f"\n{'='*70}\n")
    
    return {
        'accuracy': accuracy,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def main():
    """Run all tests"""
    print("="*70)
    print("PHISHING DETECTION - TEST SUITE")
    print("="*70)
    
    # Load tests
    test_cases = load_test_cases()
    print(f"\nLoaded {len(test_cases)} test cases\n")
    
    # Run tests
    results = []
    for test_case in test_cases:
        result = run_single_test(test_case, show_details=False)
        results.append(result)
    
    # Analyze
    analysis = analyze_results(results)
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump({
            'results': results,
            'analysis': {
                'accuracy': analysis['accuracy'],
                'false_positive_count': len(analysis['false_positives']),
                'false_negative_count': len(analysis['false_negatives'])
            }
        }, f, indent=2)
    
    print("Results saved to test_results.json")
    
    # Return exit code based on performance
    if analysis['accuracy'] >= 0.90:
        print("\n✓ EXCELLENT: >90% accuracy achieved!")
        return 0
    elif analysis['accuracy'] >= 0.75:
        print("\n⚡ GOOD: >75% accuracy, but room for improvement")
        return 0
    else:
        print("\n⚠️  NEEDS IMPROVEMENT: <75% accuracy")
        return 1

if __name__ == "__main__":
    sys.exit(main())
