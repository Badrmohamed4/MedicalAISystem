"""
NLP Pipeline Evaluation
========================
Measures accuracy of:
1. Intent classification
2. Symptom extraction
3. Medical context detection
4. Urgency detection

Run with:
    cd MedicalAISystem
    python evaluation/nlp_evaluation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medical_ai_project"))

# ── Test Dataset ──────────────────────────────────────────────────────────────
# Format: (input_text, expected_intent, expected_context, expected_symptoms_contain, is_urgent)
TEST_CASES = [
    # Brain-related
    ("I have a severe headache and blurry vision",       "symptom_report", "brain",  ["headache"],          False),
    ("I've been having seizures and memory loss",        "symptom_report", "brain",  ["seizure"],           True),
    ("My head is pounding and I feel dizzy",             "symptom_report", "brain",  ["pounding", "dizziness"], False),
    ("I have a migraine that won't go away for 2 days",  "symptom_report", "brain",  ["migraine"],          False),

    # Lung-related
    ("I have chest pain and shortness of breath",        "symptom_report", "lung",   ["chest"],             True),
    ("I've been coughing for 3 weeks with blood",        "symptom_report", "lung",   ["cough"],             False),
    ("I feel wheezing and tightness in my chest",        "symptom_report", "lung",   ["chest"],             False),

    # Skin-related
    ("I have a red rash on my arm that is very itchy",   "symptom_report", "skin",   ["rash"],              False),
    ("There is a mole on my back that changed color",    "symptom_report", "skin",   ["mole"],              False),
    ("My skin is dry and flaky with white patches",      "symptom_report", "skin",   ["dry", "flaky"],      False),

    # Greetings
    ("Hello, I need some help",                          "greet",          "none",   [],                    False),
    ("Hi there",                                         "greet",          "none",   [],                    False),

    # Urgency cases
    ("I cannot breathe please help me",                  "symptom_report", "lung",   [],                    True),
    ("I think I am having a heart attack",               "symptom_report", "lung",   [],                    True),
    ("I collapsed and I am unconscious",                 "symptom_report", "brain",  [],                    True),

    # General
    ("What should I do about my symptoms?",              "treatment_query","none",   [],                    False),
    ("Can you help me understand my diagnosis?",         "diagnosis_query","none",   [],                    False),
    ("I feel tired and have no appetite",                "symptom_report", "none",   ["tired"],             False),
    ("I am done, thank you",                             "end_conversation","none",  [],                    False),
    ("I have nausea and vomiting since yesterday",       "symptom_report", "none",   ["nausea"],            False),
]


def evaluate():
    print("=" * 60)
    print("  MEDICAL AI — NLP PIPELINE EVALUATION")
    print("=" * 60)

    # Load components
    print("\n[1] Loading NLP Processor...")
    try:
        from nlp.processor import NLPProcessor
        processor = NLPProcessor()
        print("    ✅ NLP Processor loaded")
    except Exception as e:
        print(f"    ❌ Failed to load NLP Processor: {e}")
        return

    print("[2] Loading Urgency Detector...")
    try:
        from medical_chatbot.agents.patient_agent import PatientAgent
        from medical_chatbot.nlp.state_tracker import Session
        dummy_session = Session("eval_session")
        agent = PatientAgent(dummy_session)
        print("    ✅ Urgency Detector loaded")
    except Exception as e:
        print(f"    ❌ Failed to load PatientAgent: {e}")
        agent = None

    print(f"\n[3] Running {len(TEST_CASES)} test cases...\n")
    print("-" * 60)

    # Counters
    intent_correct = 0
    context_correct = 0
    symptom_correct = 0
    urgency_correct = 0
    total = len(TEST_CASES)

    results = []

    for i, (text, exp_intent, exp_context, exp_symptoms, exp_urgent) in enumerate(TEST_CASES):
        print(f"Test {i+1:02d}: \"{text[:55]}...\"" if len(text) > 55 else f"Test {i+1:02d}: \"{text}\"")

        try:
            result = processor.process(text)
            got_intent   = result.get("intent", "others")
            got_context  = result.get("medical_context", "none")
            got_symptoms = result.get("original_symptoms", [])

            # Intent check
            i_ok = got_intent == exp_intent
            if i_ok:
                intent_correct += 1

            # Context check
            c_ok = (exp_context == "none") or (got_context == exp_context)
            if c_ok:
                context_correct += 1

            # Symptom check — at least one expected keyword appears in extracted symptoms
            if not exp_symptoms:
                s_ok = True  # no expectation = pass
            else:
                got_lower = " ".join(got_symptoms).lower()
                s_ok = any(kw.lower() in got_lower for kw in exp_symptoms)
            if s_ok:
                symptom_correct += 1

            # Urgency check
            if agent:
                u_got = agent._is_urgent(text)
            else:
                u_got = False
            u_ok = u_got == exp_urgent
            if u_ok:
                urgency_correct += 1

            status = "✅" if (i_ok and c_ok and s_ok and u_ok) else "⚠️"
            print(f"  {status} Intent: {got_intent:20s} (expected: {exp_intent})")
            if not c_ok:
                print(f"  ❌ Context: {got_context:20s} (expected: {exp_context})")
            if not s_ok:
                print(f"  ❌ Symptoms: {got_symptoms} (expected to contain: {exp_symptoms})")
            if not u_ok:
                print(f"  ❌ Urgency: {u_got} (expected: {exp_urgent})")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")

        print()

    # ── Results Summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Intent Classification Accuracy : {intent_correct}/{total} = {intent_correct/total*100:.1f}%")
    print(f"  Medical Context Accuracy        : {context_correct}/{total} = {context_correct/total*100:.1f}%")
    print(f"  Symptom Extraction Coverage     : {symptom_correct}/{total} = {symptom_correct/total*100:.1f}%")
    print(f"  Urgency Detection Accuracy      : {urgency_correct}/{total} = {urgency_correct/total*100:.1f}%")
    print("-" * 60)
    overall = (intent_correct + context_correct + symptom_correct + urgency_correct) / (total * 4) * 100
    print(f"  Overall Pipeline Score          : {overall:.1f}%")
    print("=" * 60)

    return {
        "intent_accuracy": intent_correct / total,
        "context_accuracy": context_correct / total,
        "symptom_coverage": symptom_correct / total,
        "urgency_accuracy": urgency_correct / total,
        "overall": overall
    }


if __name__ == "__main__":
    evaluate()