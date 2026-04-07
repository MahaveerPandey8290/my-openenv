"""
Synthetic patient data factory.
All patients are entirely fictional and procedurally generated from templates.
Each task has its own patient pool with known ground-truth answers.
"""
import random
from typing import Any, Dict, List

# ─────────────────────────────────────────────
# TASK: vital_signs_triage  (EASY)
# Agent sees vitals + complaint only.
# Must classify triage level correctly.
# ─────────────────────────────────────────────
VITAL_SIGNS_PATIENTS: List[Dict[str, Any]] = [
    {
        "complaint": "I feel dizzy and my heart is racing really fast.",
        "vitals": {"HR": "148 bpm", "BP": "88/54 mmHg", "SpO2": "96%", "Temp": "37.1°C", "RR": "18/min"},
        "initial_symptoms": ["palpitations", "dizziness", "diaphoresis"],
        "extra_symptoms": ["chest tightness"],
        "history": ["no prior cardiac history", "takes no medications"],
        "triage_level": "immediate",
        "condition": "supraventricular tachycardia with hypotension",
        "condition_category": "cardiac",
        "relevant_tests": ["ECG", "troponin", "electrolytes"],
    },
    {
        "complaint": "I cut my finger while cooking, it won't stop bleeding.",
        "vitals": {"HR": "82 bpm", "BP": "118/74 mmHg", "SpO2": "99%", "Temp": "36.8°C", "RR": "14/min"},
        "initial_symptoms": ["laceration right index finger", "minor bleeding"],
        "extra_symptoms": ["wound approximately 2cm"],
        "history": ["no blood thinners", "tetanus up to date"],
        "triage_level": "non_urgent",
        "condition": "minor laceration",
        "condition_category": "trauma",
        "relevant_tests": ["wound assessment"],
    },
    {
        "complaint": "Severe headache started suddenly an hour ago, worst of my life.",
        "vitals": {"HR": "90 bpm", "BP": "178/102 mmHg", "SpO2": "98%", "Temp": "37.0°C", "RR": "16/min"},
        "initial_symptoms": ["thunderclap headache", "neck stiffness", "photophobia"],
        "extra_symptoms": ["nausea", "vomiting"],
        "history": ["no prior headache history", "non-smoker"],
        "triage_level": "immediate",
        "condition": "subarachnoid haemorrhage",
        "condition_category": "neurological",
        "relevant_tests": ["CT head", "lumbar puncture", "blood pressure monitoring"],
    },
    {
        "complaint": "My ankle hurts after I twisted it playing football.",
        "vitals": {"HR": "76 bpm", "BP": "122/78 mmHg", "SpO2": "99%", "Temp": "36.9°C", "RR": "15/min"},
        "initial_symptoms": ["right ankle swelling", "tenderness on palpation", "able to weight-bear"],
        "extra_symptoms": ["bruising lateral malleolus"],
        "history": ["no prior injuries", "fit and healthy 22-year-old"],
        "triage_level": "less_urgent",
        "condition": "lateral ankle sprain",
        "condition_category": "musculoskeletal",
        "relevant_tests": ["Ottawa ankle rules assessment", "X-ray if indicated"],
    },
    {
        "complaint": "Chest pain for 20 minutes, spreading to my left arm.",
        "vitals": {"HR": "102 bpm", "BP": "142/88 mmHg", "SpO2": "95%", "Temp": "37.2°C", "RR": "20/min"},
        "initial_symptoms": ["crushing chest pain", "left arm radiation", "diaphoresis"],
        "extra_symptoms": ["nausea", "shortness of breath"],
        "history": ["hypertension", "smoker", "family history of MI"],
        "triage_level": "immediate",
        "condition": "acute myocardial infarction",
        "condition_category": "cardiac",
        "relevant_tests": ["ECG", "troponin", "aspirin", "chest X-ray"],
    },
    {
        "complaint": "I have a sore throat and mild fever for two days.",
        "vitals": {"HR": "88 bpm", "BP": "116/72 mmHg", "SpO2": "98%", "Temp": "38.1°C", "RR": "15/min"},
        "initial_symptoms": ["pharyngitis", "low-grade fever", "mild odynophagia"],
        "extra_symptoms": ["no stridor", "no drooling"],
        "history": ["no immunocompromise", "no penicillin allergy"],
        "triage_level": "non_urgent",
        "condition": "viral pharyngitis",
        "condition_category": "infectious",
        "relevant_tests": ["throat swab", "FBC if persists"],
    },
]

# ─────────────────────────────────────────────
# TASK: differential_diagnosis  (MEDIUM)
# Agent sees vitals + symptoms + partial history.
# Must identify condition AND recommend 2+ relevant tests.
# ─────────────────────────────────────────────
DIFFERENTIAL_PATIENTS: List[Dict[str, Any]] = [
    {
        "complaint": "Sharp right lower abdominal pain, started 6 hours ago, getting worse.",
        "vitals": {"HR": "96 bpm", "BP": "124/80 mmHg", "SpO2": "98%", "Temp": "38.3°C", "RR": "18/min"},
        "initial_symptoms": ["RLQ pain", "rebound tenderness", "guarding", "anorexia"],
        "extra_symptoms": ["Rovsing's sign positive", "psoas sign positive"],
        "history": ["no prior abdominal surgery", "last bowel movement yesterday", "male 24yo"],
        "triage_level": "urgent",
        "condition": "acute appendicitis",
        "condition_category": "appendic",
        "relevant_tests": ["FBC", "CRP", "ultrasound abdomen", "CT abdomen"],
    },
    {
        "complaint": "Sudden onset shortness of breath, right sided chest pain, recent long-haul flight.",
        "vitals": {"HR": "118 bpm", "BP": "104/66 mmHg", "SpO2": "91%", "Temp": "37.4°C", "RR": "26/min"},
        "initial_symptoms": ["dyspnoea", "pleuritic chest pain", "tachycardia", "tachypnoea"],
        "extra_symptoms": ["right calf swelling", "leg tenderness"],
        "history": ["14-hour flight 3 days ago", "on oral contraceptive pill", "no prior VTE"],
        "triage_level": "immediate",
        "condition": "pulmonary embolism",
        "condition_category": "pulmonary embol",
        "relevant_tests": ["D-dimer", "CT pulmonary angiogram", "ABG", "ECG", "leg Doppler"],
    },
    {
        "complaint": "Cannot pass urine, severe lower abdominal pain.",
        "vitals": {"HR": "94 bpm", "BP": "148/92 mmHg", "SpO2": "97%", "Temp": "37.0°C", "RR": "17/min"},
        "initial_symptoms": ["urinary retention", "suprapubic pain", "distended bladder on palpation"],
        "extra_symptoms": ["nocturia x3 prior weeks", "weak stream lately"],
        "history": ["male 67yo", "known BPH", "started new antihistamine recently"],
        "triage_level": "urgent",
        "condition": "acute urinary retention secondary to BPH",
        "condition_category": "urin",
        "relevant_tests": ["bladder scan", "urinalysis", "PSA", "renal function", "catheterisation"],
    },
    {
        "complaint": "Confusion and high fever in my elderly father.",
        "vitals": {"HR": "106 bpm", "BP": "98/60 mmHg", "SpO2": "94%", "Temp": "39.6°C", "RR": "24/min"},
        "initial_symptoms": ["acute confusion", "fever", "hypotension", "tachycardia", "tachypnoea"],
        "extra_symptoms": ["not producing much urine", "cool peripheries"],
        "history": ["78yo male", "type 2 diabetes", "indwelling urinary catheter", "UTI 2 months ago"],
        "triage_level": "immediate",
        "condition": "urosepsis with septic shock",
        "condition_category": "sepsis",
        "relevant_tests": ["blood cultures", "urine culture", "FBC", "lactate", "IV antibiotics", "fluid resuscitation"],
    },
]

# ─────────────────────────────────────────────
# TASK: polytrauma_cascade  (HARD)
# Symptoms are revealed across 5 steps.
# Agent must update triage level and diagnosis as new info arrives.
# Correct answer only achievable by integrating ALL steps.
# ─────────────────────────────────────────────
POLYTRAUMA_PATIENTS: List[Dict[str, Any]] = [
    {
        "complaint": "RTA — motorcyclist vs car, brought in by ambulance.",
        "vitals": {"HR": "124 bpm", "BP": "96/58 mmHg", "SpO2": "93%", "Temp": "36.5°C", "RR": "28/min"},
        "initial_symptoms": ["mechanism: high-speed RTA", "GCS 12", "visible right femur deformity"],
        "extra_symptoms": [
            "step 2: decreased air entry left base, trachea deviated right",
            "step 3: distended neck veins, muffled heart sounds",
            "step 4: abdomen rigid, pelvis unstable on spring test",
            "step 5: GCS now 8, pupils unequal",
        ],
        "history": [
            "step 1: no helmet worn",
            "step 2: on warfarin for AF",
            "step 3: splenectomy 2 years ago",
            "step 4: blood glucose 2.1 (hypoglycaemic)",
            "step 5: wife reports prior head injury 1 year ago",
        ],
        "triage_level": "immediate",
        "condition": "polytrauma: tension pneumothorax + cardiac tamponade + traumatic brain injury",
        "condition_category": "trauma",
        "relevant_tests": [
            "FAST ultrasound", "chest X-ray", "CT trauma series",
            "massive transfusion protocol", "needle decompression", "pericardiocentesis"
        ],
    },
    {
        "complaint": "Found collapsed at home by neighbour, unknown downtime.",
        "vitals": {"HR": "42 bpm", "BP": "80/40 mmHg", "SpO2": "88%", "Temp": "34.1°C", "RR": "8/min"},
        "initial_symptoms": ["GCS 6", "bradycardia", "hypothermia", "hypoxia"],
        "extra_symptoms": [
            "step 2: pupils 2mm bilateral, rigidity",
            "step 3: empty bottle of metoprolol found",
            "step 4: glucose 1.8, potassium 6.8 on bloods",
            "step 5: ECG shows complete heart block",
        ],
        "history": [
            "step 1: elderly female, lives alone",
            "step 2: known hypothyroidism, non-compliant",
            "step 3: recent GP visit for depression",
            "step 4: no known drug allergies",
            "step 5: previous intentional overdose 2 years ago",
        ],
        "triage_level": "immediate",
        "condition": "beta-blocker overdose with complete heart block and myxoedema features",
        "condition_category": "toxicolog",
        "relevant_tests": [
            "glucagon IV", "calcium chloride", "high-dose insulin therapy",
            "atropine", "transcutaneous pacing", "thyroid function tests", "toxicology screen"
        ],
    },
]


TASK_PATIENT_MAP = {
    "vital_signs_triage": VITAL_SIGNS_PATIENTS,
    "differential_diagnosis": DIFFERENTIAL_PATIENTS,
    "polytrauma_cascade": POLYTRAUMA_PATIENTS,
}


def generate_patient(task_name: str) -> Dict[str, Any]:
    """
    Select a random patient for the given task.
    Returns a dict with all patient data including ground truth.
    """
    pool = TASK_PATIENT_MAP.get(task_name)
    if not pool:
        raise ValueError(f"Unknown task: {task_name}. Must be one of {list(TASK_PATIENT_MAP.keys())}")
    return dict(random.choice(pool))  # shallow copy to avoid mutation
