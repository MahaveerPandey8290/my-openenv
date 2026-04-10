"""
Procedural patient generator — produces thousands of unique patients.
Uses seeds for reproducibility. Never memorises — always generates fresh.
Inspired by Reasoning Gym's procedural generation approach.
"""
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional


# ── Building blocks for procedural generation ────────────────────────────────

COMPLAINTS = {
    "cardiac": [
        "crushing chest pain radiating to my left arm for {mins} minutes",
        "sudden chest tightness and shortness of breath since {mins} minutes ago",
        "my heart is racing and I feel faint",
        "sharp chest pain, worse on breathing",
        "palpitations and dizziness started {mins} minutes ago",
    ],
    "neurological": [
        "worst headache of my life, started suddenly {mins} minutes ago",
        "sudden severe headache and my neck feels stiff",
        "I can't see properly and my head is pounding",
        "thunderclap headache, photophobia, feels like head is exploding",
    ],
    "pulmonary_embol": [
        "sudden shortness of breath and right sided chest pain",
        "I can't catch my breath, came on suddenly {mins} minutes ago",
        "pleuritic chest pain and leg swelling after long flight",
        "sudden breathlessness, SpO2 dropping",
    ],
    "appendic": [
        "right lower abdominal pain getting worse over {hrs} hours",
        "started around my belly button now moved to the right side",
        "severe stomach pain, I can't stand up straight",
        "sharp pain in my lower right abdomen, nausea, no appetite",
    ],
    "sepsis": [
        "my elderly father is confused and has a high fever",
        "fever, shaking, I feel really unwell, can't stay awake",
        "high temperature and I can barely stand, very weak",
        "rigors and confusion, temp {temp} degrees",
    ],
    "trauma": [
        "motorcycle accident, I was thrown from the bike",
        "hit by a car, I can't move my leg",
        "fell from height onto concrete",
        "RTA, airbag deployed, chest pain",
    ],
    "musculoskeletal": [
        "twisted my ankle playing sport",
        "fell and hurt my wrist",
        "knee pain after a tackle during football",
        "lower back pain after lifting at work",
    ],
    "infectious": [
        "sore throat and mild fever for {days} days",
        "cough, runny nose and temperature for a few days",
        "ear pain and fever since yesterday",
        "body aches, headache, feeling flu-like",
    ],
}

VITALS_PROFILES = {
    "immediate": {
        "HR": lambda r: f"{r.randint(110,160)} bpm",
        "BP": lambda r: f"{r.randint(70,95)}/{r.randint(40,60)} mmHg",
        "SpO2": lambda r: f"{r.randint(88,94)}%",
        "Temp": lambda r: f"{round(r.uniform(36.0,39.8),1)}°C",
        "RR": lambda r: f"{r.randint(22,32)}/min",
    },
    "urgent": {
        "HR": lambda r: f"{r.randint(95,115)} bpm",
        "BP": lambda r: f"{r.randint(90,140)}/{r.randint(55,90)} mmHg",
        "SpO2": lambda r: f"{r.randint(93,97)}%",
        "Temp": lambda r: f"{round(r.uniform(36.5,39.2),1)}°C",
        "RR": lambda r: f"{r.randint(18,24)}/min",
    },
    "less_urgent": {
        "HR": lambda r: f"{r.randint(78,95)} bpm",
        "BP": lambda r: f"{r.randint(110,145)}/{r.randint(65,88)} mmHg",
        "SpO2": lambda r: f"{r.randint(96,99)}%",
        "Temp": lambda r: f"{round(r.uniform(36.5,38.2),1)}°C",
        "RR": lambda r: f"{r.randint(14,18)}/min",
    },
    "non_urgent": {
        "HR": lambda r: f"{r.randint(65,85)} bpm",
        "BP": lambda r: f"{r.randint(110,135)}/{r.randint(65,82)} mmHg",
        "SpO2": lambda r: f"{r.randint(97,100)}%",
        "Temp": lambda r: f"{round(r.uniform(36.4,37.5),1)}°C",
        "RR": lambda r: f"{r.randint(12,16)}/min",
    },
}

SYMPTOMS_POOL = {
    "cardiac": [
        ["crushing chest pain", "diaphoresis", "left arm radiation"],
        ["palpitations", "syncope", "dizziness"],
        ["chest tightness", "nausea", "shortness of breath"],
        ["pleuritic chest pain", "tachycardia", "diaphoresis"],
    ],
    "neurological": [
        ["thunderclap headache", "neck stiffness", "photophobia"],
        ["sudden severe headache", "vomiting", "photophobia"],
        ["headache", "neck rigidity", "altered consciousness"],
    ],
    "pulmonary_embol": [
        ["sudden dyspnoea", "pleuritic chest pain", "tachycardia"],
        ["shortness of breath", "right leg swelling", "tachypnoea"],
        ["hypoxia", "tachycardia", "pleuritic pain"],
    ],
    "appendic": [
        ["RLQ pain", "rebound tenderness", "guarding"],
        ["periumbilical pain migrated RLQ", "anorexia", "nausea"],
        ["RLQ tenderness", "Rovsing positive", "fever"],
    ],
    "sepsis": [
        ["fever", "rigors", "confusion", "hypotension"],
        ["high temperature", "tachycardia", "reduced consciousness"],
        ["fever", "cool peripheries", "mottled skin"],
    ],
    "trauma": [
        ["mechanism: RTA high speed", "deformity right femur", "GCS 12"],
        ["chest wall tenderness", "reduced air entry", "tracheal deviation"],
        ["visible deformity", "haemorrhage", "mechanism: fall from height"],
    ],
    "musculoskeletal": [
        ["ankle swelling", "lateral tenderness", "able to weight-bear"],
        ["wrist pain", "swelling", "mechanism: FOOSH"],
        ["knee effusion", "medial tenderness", "mechanism: tackle"],
    ],
    "infectious": [
        ["pharyngitis", "low-grade fever", "mild odynophagia"],
        ["rhinorrhoea", "mild cough", "myalgia"],
        ["otalgia", "fever", "ear discharge"],
    ],
}

HISTORY_POOL = {
    "cardiac": [
        "known hypertension on amlodipine",
        "smoker 20 pack-years",
        "family history MI father age 52",
        "hypercholesterolaemia on statin",
        "previous NSTEMI 3 years ago",
        "type 2 diabetes on metformin",
        "no significant cardiac history",
    ],
    "neurological": [
        "no prior headache history",
        "previously well",
        "migraines but this is different",
        "hypertension poorly controlled",
        "no anticoagulation",
        "non-smoker",
    ],
    "pulmonary_embol": [
        "long-haul flight 3 days ago",
        "on oral contraceptive pill",
        "previous DVT right leg",
        "recent surgery 2 weeks ago",
        "active malignancy",
        "no prior VTE",
        "immobilised 2 weeks post orthopaedic surgery",
    ],
    "appendic": [
        "no prior abdominal surgery",
        "last bowel movement yesterday",
        "no blood thinners",
        "fit and well",
        "appendicitis in sibling",
    ],
    "sepsis": [
        "indwelling urinary catheter",
        "type 2 diabetes",
        "recent UTI 2 months ago",
        "immunocompromised on steroids",
        "nursing home resident",
        "chronic kidney disease",
    ],
    "trauma": [
        "no helmet worn",
        "on warfarin for AF",
        "splenectomy 2 years ago",
        "previously fit and well",
        "alcohol intoxicated at scene",
    ],
    "musculoskeletal": [
        "no prior injuries",
        "fit and well",
        "tetanus up to date",
        "regular sport participation",
    ],
    "infectious": [
        "no immunocompromise",
        "no penicillin allergy",
        "smoker",
        "no significant PMH",
        "asthma well-controlled",
    ],
}

RELEVANT_TESTS = {
    "cardiac": ["ECG", "troponin", "chest X-ray", "echocardiogram", "electrolytes"],
    "neurological": ["CT head", "lumbar puncture", "blood pressure monitoring", "coagulation"],
    "pulmonary_embol": ["D-dimer", "CT pulmonary angiogram", "ABG", "ECG", "leg Doppler"],
    "appendic": ["FBC", "CRP", "ultrasound abdomen", "CT abdomen"],
    "sepsis": ["blood cultures", "urine culture", "FBC", "lactate", "CRP"],
    "trauma": ["FAST ultrasound", "chest X-ray", "CT trauma series", "FBC", "coagulation"],
    "musculoskeletal": ["X-ray ankle", "X-ray wrist", "FBC"],
    "infectious": ["throat swab", "FBC", "CRP", "monospot"],
}

TRIAGE_MAP = {
    "cardiac": "immediate",
    "neurological": "immediate",
    "pulmonary_embol": "immediate",
    "appendic": "urgent",
    "sepsis": "immediate",
    "trauma": "immediate",
    "musculoskeletal": "less_urgent",
    "infectious": "non_urgent",
}


def generate_patient(task_name: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a unique synthetic patient using procedural randomness.
    seed=None -> fully random (training). seed=int -> reproducible (evaluation).
    """
    rng = random.Random(seed)

    # Pick category based on task difficulty
    if task_name == "vital_signs_triage":
        categories = list(TRIAGE_MAP.keys())
    elif task_name == "differential_diagnosis":
        categories = ["cardiac", "pulmonary_embol", "appendic", "sepsis", "neurological"]
    else:  # polytrauma_cascade
        categories = ["trauma", "sepsis", "cardiac", "neurological"]

    category = rng.choice(categories)
    triage_level = TRIAGE_MAP[category]

    # Build complaint from template
    complaint_template = rng.choice(COMPLAINTS[category])
    complaint = complaint_template.format(
        mins=rng.randint(10, 120),
        hrs=rng.randint(2, 24),
        days=rng.randint(1, 5),
        temp=round(rng.uniform(38.2, 40.1), 1),
    )

    # Build vitals
    vitals_profile = VITALS_PROFILES[triage_level]
    vitals = {k: v(rng) for k, v in vitals_profile.items()}

    # Build symptoms — pick one cluster + shuffle
    symptom_clusters = SYMPTOMS_POOL[category]
    symptoms = list(rng.choice(symptom_clusters))
    rng.shuffle(symptoms)

    # Build history — pick 2-3 items
    history_pool = HISTORY_POOL[category]
    n_history = rng.randint(2, min(4, len(history_pool)))
    history = rng.sample(history_pool, n_history)

    # For hard task: tag history items with step reveals
    if task_name == "polytrauma_cascade":
        history = [f"step {i+1}: {h}" for i, h in enumerate(history)]

    # Extra symptoms revealed over steps
    extra_symptom_pool = [
        "diaphoresis", "pallor", "peripheral shutdown",
        "altered GCS", "tachypnoea worsening", "mottled skin",
        "reduced urine output", "hypoxia worsening",
    ]
    extra_symptoms = rng.sample(extra_symptom_pool, min(3, len(extra_symptom_pool)))

    return {
        "complaint": complaint,
        "vitals": vitals,
        "initial_symptoms": symptoms[:2],
        "extra_symptoms": extra_symptoms,
        "history": history,
        "triage_level": triage_level,
        "condition": f"{category.replace('_', ' ')} presentation",
        "condition_category": category,
        "relevant_tests": RELEVANT_TESTS[category],
        "seed": seed,
    }
