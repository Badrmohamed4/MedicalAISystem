import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'medical_ai_project')))
from nlp.processor import NLPProcessor
import json

processor = NLPProcessor()
print("TEST 1: 'I have pain in my head'")
res = processor.process("I have pain in my head")
print(json.dumps(res, indent=2))

print("TEST 2: 'dizziness'")
res2 = processor.process("dizziness")
print(json.dumps(res2, indent=2))
