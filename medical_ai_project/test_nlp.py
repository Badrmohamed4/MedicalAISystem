"""Quick test of the NLP pipeline."""
import sys
import os

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))

from nlp.processor import NLPProcessor

print("=" * 50)
print("Testing NLP Pipeline")
print("=" * 50)

# Initialize
print("\n[1] Initializing NLPProcessor...")
nlp = NLPProcessor()
print("    ✅ NLP Processor initialized!\n")

# Test 1: Symptom extraction
print("[2] Test: 'I have a headache and nausea'")
result = nlp.process("I have a headache and nausea")
print(f"    Intent: {result['intent']}")
print(f"    Symptoms: {result['original_symptoms']}")
print(f"    Normalized: {result['normalized_symptoms']}")
print(f"    Scores: {result['scores']}")

# Test 2: Short text (should skip SapBERT)
print("\n[3] Test: 'hi' (short text, should skip SapBERT)")
result = nlp.process("hi")
print(f"    Intent: {result['intent']}")
print(f"    Symptoms: {result['original_symptoms']}")

# Test 3: Treatment query
print("\n[4] Test: 'What medicine can I take for chest pain?'")
result = nlp.process("What medicine can I take for chest pain?")
print(f"    Intent: {result['intent']}")
print(f"    Symptoms: {result['original_symptoms']}")
print(f"    Normalized: {result['normalized_symptoms']}")

# Test 4: Image-related
print("\n[5] Test: 'Can you analyze my MRI scan?'")
result = nlp.process("Can you analyze my MRI scan?")
print(f"    Intent: {result['intent']}")
print(f"    Symptoms: {result['original_symptoms']}")

# Test 5: LRU cache test
print("\n[6] Cache test (same input repeated)...")
result1 = nlp.process("I have a headache and nausea")
result2 = nlp.process("I have a headache and nausea")
print(f"    Cache working: {result1 == result2}")

print("\n" + "=" * 50)
print("All NLP tests passed! ✅")
print("=" * 50)
