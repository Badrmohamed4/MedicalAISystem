# Medical Knowledge Base

# Tumor Descriptions (Patient friendly)
# Medical Knowledge Base

# Tumor Descriptions (Expanded & Patient friendly)
TUMOR_INFO = {
    "Glioma": (
        "Gliomas are tumors that arise from glial cells in the brain or spinal cord. "
        "Symptoms can vary greatly depending on location but often include headaches, seizures, and cognitive changes. "
        "Treatment options may include surgery, radiation, and chemotherapy depending on the grade."
    ),
    "Meningioma": (
        "A meningioma is usually a slow-growing tumor that forms on the membranes (meninges) covering the brain and spinal cord. "
        "Many are benign, but they can cause symptoms by pressing on adjacent brain tissue. "
        "Observation or surgical removal are common management strategies."
    ),
    "Pituitary": (
        "Pituitary adenomas develop in the pituitary gland at the base of the brain. "
        "They can disrupt hormone regulation, leading to a variety of systemic symptoms, or cause vision problems by pressing on the optic nerves. "
        "Most are benign and treatable."
    ),
    "No Tumor": "The analysis does not show any obvious signs of a brain tumor in this specific scan. However, please consult with a neurologist for a complete evaluation."
}

LUNG_CANCER_INFO = {
    "Adenocarcinoma": (
        "Adenocarcinoma is the most common form of lung cancer, often found in the outer regions of the lung. "
        "It starts in mucus-producing glandular cells. It is common in current/former smokers but is also the most common lung cancer in non-smokers. "
        "Early detection significantly improves prognosis."
    ),
    "Large Cell Carcinoma": (
        "Large cell carcinoma is a subtype of non-small cell lung cancer that can appear in any part of the lung. "
        "It tends to grow and spread quickly, making it harder to treat. "
        "Immediate oncology consultation is recommended."
    ),
    "Squamous Cell Carcinoma": (
        "Squamous cell carcinoma develops in the flat cells lining the airways, usually in the central part of the lungs. "
        "It is strongly linked to a history of cigarette smoking and may cause symptoms like coughing up blood (hemoptysis). "
    ),
    "Normal": "The scan appears normal with no visible signs of lung masses or nodules. Please continue monitoring your respiratory health."
}

SKIN_INFO = {
    "Acne": (
        "Acne is a common skin condition that occurs when hair follicles become plugged with oil and dead skin cells. "
        "It commonly causes whiteheads, blackheads, or pimples, usually appearing on the face, forehead, chest, upper back, and shoulders. "
        "Treatment often involves topical creams, specialized cleansers, or prescription medications."
    ),
    "Eczema": (
        "Eczema (atopic dermatitis) is a condition that makes your skin red and itchy. "
        "It's common in children but can occur at any age. It is long lasting (chronic) and tends to flare periodically. "
        "Management includes avoiding harsh soaps, moisturizing regularly, and applying medicated creams during flare-ups."
    ),
    "Psoriasis": (
        "Psoriasis is a skin disease that causes a rash with itchy, scaly patches, most commonly on the knees, elbows, trunk and scalp. "
        "It is a chronic autoimmune condition that goes through cycles, flaring for a few weeks or months, then subsiding. "
        "Treatments range from specialized ointments to light therapy or systemic medications."
    ),
    "Malignant": (
        "⚠️ This scan shows features consistent with a malignant skin lesion (such as Melanoma). "
        "Melanoma is a serious form of skin cancer that begins in cells known as melanocytes. "
        "**It is crucial that you schedule an appointment with a dermatologist or oncologist immediately** for a physical biopsy and assessment."
    )
}

# Temporary Advice (Safe, General)
ADVICE = {
    "General": [
        "Rest in a quiet, low-lit room to reduce sensory overload.",
        "Stay hydrated and avoid skipping meals.",
        "Keep a symptom diary: note when symptoms start, how long they last, and what triggers them.",
        "Avoid alcohol and tobacco."
    ],
    "Headache": [
        "Apply a cold or warm compress to your head/neck for 15 minutes.",
        "Dim the lights and minimize screen time.",
        "Try guided breathing or relaxation techniques to reduce tension."
    ],
    "Seizure": [
        "**Safety First**: Ensure the person is lying on their side to keep the airway open.",
        "Remove clear hazards (glasses, sharp objects).",
        "Do **NOT** restrain them or put anything in their mouth.",
        "Time the seizure; call emergency services if it lasts > 5 minutes."
    ],
    "Emergency": [
        "🛑 **SEEK IMMEDIATE MEDICAL HELP** 🛑",
        "Go to the nearest Emergency Room immediately.",
        "Do not drive yourself.",
        "Call 911 (or your local emergency number)."
    ],
    "Lung_Issue": [
        "Avoid exposure to smoke, dust, and chemical fumes.",
        "Use a humidifier if the air is dry to ease breathing.",
        "Sleep with your head elevated to help open airways.",
        "Monitor your oxygen levels if you have a pulse oximeter."
    ],
    "Skin_Issue": [
        "Avoid scratching or picking at the affected area to prevent infection.",
        "Wash gently with a mild, fragrance-free cleanser and lukewarm water.",
        "Apply a gentle moisturizer if the skin is dry and flaky.",
        "Protect the area from direct sunlight by covering it or using sunscreen if appropriate."
    ]
}

# Extensive Follow-up Questions for Triage
FOLLOW_UP_QUESTIONS = {
    "general": {
        "general_checkup": [
            "How long have you been experiencing these symptoms?",
            "Have you noticed any unintentional weight loss recently?",
            "Do you have a family history of brain, lung, or skin conditions?",
            "Are you currently taking any medications?"
        ],
        "risk_factors": [
            "Do you currently smoke or have a history of smoking?",
            "Have you been exposed to asbestos or industrial chemicals?",
            "Have you previously been treated for any type of cancer?",
            "Have you had any unexplained weight loss (>10 lbs) recently?"
        ]
    },
    "brain": {
        "headache": [
            "Is the pain localized to one specific area, or is it all over?",
            "How would you describe the pain: throbbing, sharp, or dull pressure?",
            "Is the headache worse in the morning or when you cough/sneeze?",
            "Do you experience nausea or sensitivity to light/sound with the headache?"
        ],
        "seizure": [
            "How often do these episodes occur?",
            "Do you lose consciousness or awareness during the episode?",
            "Do you experience any 'aura' (strange smells, lights, or feelings) before it happens?",
            "Have you had any recent head injuries?"
        ],
        "vision loss": [
            "Is the vision change in one eye or both?",
            "Are you seeing double, or is it more like a blurry patch?",
            "Did this vision change happen suddenly or gradually?",
            "Do you see flashing lights or floating spots?"
        ],
        "dizziness": [
            "Does the room feel like it's spinning (vertigo), or do you feel lightheaded?",
            "Does changing your head position make it worse?",
            "Have you experienced any ringing in your ears (tinnitus)?",
            "Do you feel unsteady when walking?"
        ]
    },
    "lung": {
        "cough": [
            "How long have you had this cough?",
            "Is the cough dry, or are you bringing up phlegm/mucus?",
            "Have you ever coughed up blood? If so, how much?",
            "Is the cough worse at night or after physical activity?"
        ],
        "shortness of breath": [
            "Do you feel short of breath even when resting?",
            "Does it get worse when you lie flat in bed?",
            "Do you explicitly hear a wheezing sound when you breathe?",
            "Have you noticed any swelling in your ankles or legs?"
        ],
        "chest pain": [
            "does the pain get worse when you take a deep breath or cough?",
            "Is the pain sharp/stabbing or a heavy pressure?",
            "Does the pain radiate to your shoulder or back?",
            "Have you had any recent respiratory infections?"
        ]
    },
    "skin": {
        "rash": [
            "Is the rash itchy, painful, or just red?",
            "Has the rash been spreading to other parts of your body?",
            "Do you have any other symptoms like a fever or joint pain?",
            "Have you started using any new soaps, lotions, or medications recently?"
        ],
        "skin": [
            "Is the affected area itchy, painful, or just red?",
            "Has the skin issue been spreading to other parts of your body?",
            "Do you have any other symptoms like a fever or joint pain?",
            "Have you started using any new soaps, lotions, or medications recently?"
        ],
        "red": [
            "Is the red area itchy, painful, or raised?",
            "Has the redness been spreading to other parts of your body?",
            "Do you have any other symptoms like a fever or joint pain?",
            "Have you started using any new soaps, lotions, or medications recently?"
        ],
        "itch": [
            "Is the itching worse at night or after a hot shower?",
            "Can you see any bumps, blisters, or redness where it itches?",
            "Have you been exposed to any new plants, pets, or outdoor environments?",
            "Is the itching affecting your sleep?"
        ],
        "mole": [
            "Has the mole recently changed in size, shape, or color?",
            "Is the mole bleeding, oozing, or scabbing?",
            "Are the edges of the mole ragged or blurry?",
            "Is the mole larger than the size of a pencil eraser?"
        ],
        "acne": [
            "Is your acne primarily whiteheads/blackheads, or deeper painful cysts?",
            "Have you noticed it gets worse during certain times of the month or after certain foods?",
            "Are you currently using any over-the-counter acne treatments?",
            "Is the acne leaving scars or dark spots?"
        ]
    }
}

# Interim Care (Precautions before doctor visit)
INTERIM_CARE = {
    "High_Risk": [
        "**CRITICAL**: Do not ignore these symptoms.",
        "Have a family member or friend stay with you.",
        "Prepare a list of your current medications.",
        "Do not eat or drink if surgery might be required."
    ],
    "Seizure_Care": [
        "Clear the area of hard/sharp objects.",
        "Place something soft under the head.",
        "Turn gently to one side.",
        "Time the event."
    ],
    "Headache_Care": [
        "Hydrate immediately.",
        "Rest in a dark room.",
        "Avoid caffeine and alcohol."
    ]
}

# Doctor Language Mapping (Patient Term -> Clinical Term)
CLINICAL_TERMS = {
    "headache": "Cephalalgia",
    "vomit": "Emesis",
    "vomiting": "Emesis",
    "dizzy": "Vertigo/Dizziness",
    "dizziness": "Vertigo",
    "pain": "Acute Pain",
    "weakness": "Asthenia",
    "blur": "Visual Disturbances",
    "seizure": "Seizure Activity",
    "nausea": "Nausea",
    "rash": "Dermatitis/Erythema",
    "itch": "Pruritus",
    "mole": "Nevus",
    "acne": "Acne Vulgaris"
}
