import sys, os
sys.path.append(os.path.join(os.getcwd(), 'medical_ai_project'))
from nlp.biobert import BioBERTExtractor
from nlp.processor import NLPProcessor
from medical_chatbot.nlp.pipeline import EntityExtractor, IntentClassifier

extractor = BioBERTExtractor()
print("BioBERT extract:", extractor.extract("dizziness"))

processor = NLPProcessor()
print("NLP Processor:", processor.process("dizziness"))

entity_extractor = EntityExtractor()
print("Entity Extractor:", entity_extractor.extract("dizziness"))

intent_classifier = IntentClassifier()
print("Intent Classifier:", intent_classifier.predict("dizziness"))
