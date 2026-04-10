"""
Diagnostic test result bank.
Maps (condition_category, test_name) -> realistic result string.
Relevant tests return abnormal findings; irrelevant tests return normal.
"""
from typing import Tuple

TEST_RESULTS = {
    "cardiac": {
        "ecg": "ST elevation leads II III aVF rate 102bpm. Inferior STEMI pattern.",
        "troponin": "Troponin I: 2.8 ng/mL ELEVATED (normal <0.04 ng/mL).",
        "chest x-ray": "Mild cardiomegaly. No pulmonary oedema. Normal mediastinum.",
        "electrolytes": "Na 138 K 3.9 Cl 102 HCO3 24. All within normal limits.",
        "echocardiogram": "Regional wall motion abnormality inferior wall. EF 45%.",
        "fbc": "WBC 12.1 Hb 138 Plt 220. Mild leukocytosis.",
        "abg": "pH 7.38 PaO2 88 PaCO2 38 SpO2 95%. Mild hypoxia.",
    },
    "neurological": {
        "ct head": "URGENT: Hyperdense lesion basal cisterns — subarachnoid haemorrhage.",
        "lumbar puncture": "Xanthochromia present. RBC 12000/mm3. Opening pressure 28cmH2O.",
        "blood pressure monitoring": "BP 178/102 on admission 182/108 at 20min. Persistently elevated.",
        "coagulation": "PT 12.1s APTT 28s INR 1.0. Normal clotting.",
        "fbc": "WBC 9.2 Hb 142. Normal.",
        "mri brain": "CT preferred acutely. MRI arranged.",
    },
    "pulmonary_embol": {
        "d-dimer": "D-dimer 4.8 mcg/mL FEU ELEVATED (normal <0.5). High sensitivity positive.",
        "ct pulmonary angiogram": "BILATERAL pulmonary emboli. Saddle embolus at bifurcation. RV strain.",
        "abg": "pH 7.46 PaO2 61 PaCO2 32. Type 1 respiratory failure.",
        "ecg": "Sinus tachycardia 118bpm. S1Q3T3. Right heart strain pattern.",
        "leg doppler": "Extensive DVT right femoral and popliteal veins confirmed.",
        "chest x-ray": "Wedge opacity right lower lobe. Oligaemia right lung.",
        "fbc": "WBC 11.4 Normal Hb and platelets.",
    },
    "appendic": {
        "fbc": "WBC 16.8 ELEVATED. Neutrophils 14.2. Left shift. Acute infection.",
        "crp": "CRP 88 mg/L ELEVATED (normal <5). Acute inflammation.",
        "ultrasound abdomen": "Non-compressible appendix 9mm. Periappendiceal fat stranding.",
        "ct abdomen": "Thickened appendix with periappendiceal inflammation. No perforation.",
        "urinalysis": "Trace blood only. No infection. Normal.",
    },
    "sepsis": {
        "blood cultures": "POSITIVE: E.coli bacteraemia preliminary. Sensitivities pending.",
        "urine culture": "POSITIVE: E.coli >10^5 CFU/mL. Pan-sensitive.",
        "fbc": "WBC 21.4 ELEVATED. Bands 18%. Toxic granulation.",
        "lactate": "Serum lactate 4.2 mmol/L CRITICALLY ELEVATED (septic shock >2).",
        "crp": "CRP 312 mg/L SEVERELY ELEVATED.",
        "electrolytes": "Na 131 low K 5.8 HIGH Creatinine 224 AKI.",
    },
    "trauma": {
        "fast ultrasound": "POSITIVE: Free fluid hepatorenal space and pelvis. Haemoperitoneum.",
        "chest x-ray": "LEFT tension pneumothorax. Tracheal deviation right. Absent BS left.",
        "ct trauma series": "Multiple rib fractures. Splenic laceration grade III. Pelvic disruption.",
        "fbc": "Hb 68 CRITICALLY LOW. WBC 18. Plt 88 low.",
        "coagulation": "INR 2.8 HIGH warfarin effect. APTT 58s prolonged. DIC picture.",
        "glucose": "Blood glucose 2.1 mmol/L HYPOGLYCAEMIC critical.",
    },
    "musculoskeletal": {
        "x-ray ankle": "No fracture Ottawa criteria. Soft tissue swelling lateral malleolus.",
        "x-ray wrist": "No fracture. Soft tissue swelling dorsum wrist.",
        "fbc": "WBC 7.2 Normal Hb Normal platelets.",
        "crp": "CRP 4 mg/L Normal. No systemic inflammation.",
    },
    "infectious": {
        "throat swab": "Negative rapid strep. Viral culture pending.",
        "fbc": "WBC 9.1 Lymphocyte predominance. No left shift. Viral pattern.",
        "crp": "CRP 8 mg/L mildly elevated. Consistent with viral illness.",
        "monospot": "Negative.",
    },
}

DEFAULT_NORMAL = "Result: Within normal limits. No acute abnormality detected for this presentation."


def get_test_result(condition_category: str, test_name: str) -> Tuple[str, bool]:
    """Return (result_string, is_relevant)."""
    cat_bank = TEST_RESULTS.get(condition_category, {})
    key = test_name.lower().strip()

    for bank_key, result in cat_bank.items():
        if bank_key in key or key in bank_key:
            return result, True

    for other_cat, other_bank in TEST_RESULTS.items():
        if other_cat == condition_category:
            continue
        for bank_key in other_bank:
            if bank_key in key or key in bank_key:
                return f"{test_name}: No significant abnormality for this presentation.", False

    return DEFAULT_NORMAL, False
