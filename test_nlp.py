import sys
sys.path.append('.')
from medical_chatbot.nlp.pipeline import IntentClassifier, EntityExtractor

print("Initializing classifiers...")
clf = IntentClassifier()
ner = EntityExtractor()

print("Testing Intent: 'I have a severe headache'")
intent = clf.predict("I have a severe headache")
print("Intent:", intent)

print("Testing Entity: 'I have a severe headache'")
entity = ner.extract("I have a severe headache")
print("Entity:", entity)
