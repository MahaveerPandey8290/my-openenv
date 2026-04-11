#!/usr/bin/env python3
"""
Test runner for Clinical Triage Model
Run this to check if your model meets the test conditions
"""

import json
import os
import requests
from typing import Dict, List, Tuple
from pathlib import Path
from openai import OpenAI

# Load .env file if it exists
env_path = Path(".env")
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

BASE_URL = "http://localhost:7860"

# Initialize LLM client (same as inference.py)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY", os.getenv("HF_TOKEN"))

llm_client = None
if API_KEY:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are a clinical triage AI assistant in an emergency department.

Respond ONLY with a JSON object. No markdown, no preamble, no explanation.

Required JSON format:
{
  "triage_level": "immediate" | "urgent" | "less_urgent" | "non_urgent",
  "suspected_condition": "<your primary diagnosis>",
  "recommended_tests": ["<test1>", "<test2>"],
  "reasoning": "<at least 2 sentences of clinical reasoning>"
}"""


def get_model_action(observation: Dict) -> Dict:
    """Call your LLM model to get triage action"""
    if not llm_client:
        # Model not configured - use placeholder
        return {
            "triage_level": "urgent",
            "suspected_condition": "Unknown condition",
            "recommended_tests": [],
            "reasoning": "No model available - using placeholder"
        }
    
    # Format observation for the prompt
    obs_text = f"""
Chief Complaint: {observation.get('chief_complaint', 'N/A')}

Vitals:
{json.dumps(observation.get('vitals', {}), indent=2)}

Symptoms: {', '.join(observation.get('visible_symptoms', []))}

History: {', '.join(observation.get('history_revealed', []))}
"""
    
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text}
            ],
            temperature=0.3,
            max_tokens=400,
        )
        
        response_text = response.choices[0].message.content
        action = json.loads(response_text)
        return action
    except Exception as e:
        print(f"    ⚠️  Model error: {e}")
        return {
            "triage_level": "urgent",
            "suspected_condition": "Error calling model",
            "recommended_tests": [],
            "reasoning": f"Model error: {str(e)}"
        }


# Test case definitions
TEST_CASES = {
    "vital_signs_triage": [
        {
            "name": "1.1_AMI",
            "observation": {
                "chief_complaint": "Severe chest pain and shortness of breath",
                "vitals": {"HR": "125 bpm", "BP": "92/58 mmHg", "SpO2": "88%", "Temp": "37.1°C", "RR": "28/min"},
                "visible_symptoms": ["chest pain", "diaphoresis", "anxiety"],
            },
            "expected": {
                "triage_level": "immediate",
                "condition_keywords": ["cardiac", "ami", "coronary", "acs"],
                "required_tests": ["ecg", "troponin"],
                "min_reasoning_length": 50,
            }
        },
        {
            "name": "1.2_Migraine",
            "observation": {
                "chief_complaint": "Headache for 2 hours, throbbing pain",
                "vitals": {"HR": "78 bpm", "BP": "118/76 mmHg", "SpO2": "98%", "Temp": "37.0°C", "RR": "16/min"},
                "visible_symptoms": ["photophobia", "nausea"],
                "history_revealed": ["history of similar migraines"],
            },
            "expected": {
                "triage_level": "non_urgent",
                "condition_keywords": ["migraine"],
                "required_tests": [],
                "min_reasoning_length": 30,
            }
        },
        {
            "name": "1.3_DKA",
            "observation": {
                "chief_complaint": "Severe nausea, vomiting, confusion for 6 hours",
                "vitals": {"HR": "115 bpm", "BP": "105/65 mmHg", "SpO2": "94%", "Temp": "37.8°C", "RR": "24/min"},
                "visible_symptoms": ["altered mental status", "Kussmaul breathing", "fruity breath odor"],
                "history_revealed": ["Type 1 diabetes"],
            },
            "expected": {
                "triage_level": "urgent",
                "condition_keywords": ["dka", "ketoacidosis"],
                "required_tests": ["blood glucose", "blood gas", "electrolytes"],
                "min_reasoning_length": 50,
            }
        },
        {
            "name": "1.4_AnkleSprain",
            "observation": {
                "chief_complaint": "Twisted ankle 1 hour ago, mild pain",
                "vitals": {"HR": "82 bpm", "BP": "120/78 mmHg", "SpO2": "99%", "Temp": "37.0°C", "RR": "16/min"},
                "visible_symptoms": ["mild ankle swelling", "able to walk"],
                "history_revealed": [],
            },
            "expected": {
                "triage_level": "less_urgent",
                "condition_keywords": ["sprain", "musculoskeletal"],
                "required_tests": ["x-ray"],
                "min_reasoning_length": 30,
            }
        },
    ],
    "differential_diagnosis": [
        {
            "name": "2.1_Appendicitis",
            "observations": [
                {
                    "chief_complaint": "Abdominal pain started this morning",
                    "vitals": {"HR": "92 bpm", "BP": "125/80 mmHg", "SpO2": "97%", "Temp": "38.5°C", "RR": "18/min"},
                    "visible_symptoms": ["right lower quadrant tender"],
                    "history_revealed": [],
                    "step_number": 1,
                }
            ],
            "expected": {
                "triage_level": "urgent",
                "condition_keywords": ["appendicitis"],
                "required_tests": ["cbc", "cmp", "imaging"],
            }
        },
        {
            "name": "2.2_Pneumonia",
            "observations": [
                {
                    "chief_complaint": "Cough and fever for 3 days",
                    "vitals": {"HR": "98 bpm", "BP": "130/82 mmHg", "SpO2": "92%", "Temp": "39.2°C", "RR": "22/min"},
                    "visible_symptoms": ["productive cough", "dyspnea", "crackles left lower lobe"],
                    "history_revealed": ["smoker", "recent upper respiratory infection"],
                    "step_number": 1,
                }
            ],
            "expected": {
                "triage_level": "urgent",
                "condition_keywords": ["pneumonia", "infection"],
                "required_tests": ["x-ray", "blood culture"],
            }
        },
        {
            "name": "2.3_UTI",
            "observations": [
                {
                    "chief_complaint": "Burning with urination and lower abdominal discomfort",
                    "vitals": {"HR": "88 bpm", "BP": "122/76 mmHg", "SpO2": "98%", "Temp": "37.5°C", "RR": "16/min"},
                    "visible_symptoms": ["dysuria", "urinary frequency", "suprapubic tenderness"],
                    "history_revealed": ["female", "no fever"],
                    "step_number": 1,
                }
            ],
            "expected": {
                "triage_level": "less_urgent",
                "condition_keywords": ["uti", "infection"],
                "required_tests": ["urinalysis"],
            }
        },
    ],
    "polytrauma_cascade": [
        {
            "name": "3.1_MVA",
            "observations": [
                {
                    "chief_complaint": "Motor vehicle collision - high speed impact",
                    "vitals": {"HR": "130 bpm", "BP": "88/52 mmHg", "SpO2": "91%", "Temp": "36.2°C", "RR": "30/min"},
                    "visible_symptoms": ["obvious left arm deformity", "lacerations", "abdominal bruising"],
                    "history_revealed": ["ejected from vehicle"],
                    "step_number": 1,
                },
                {
                    "chief_complaint": "Motor vehicle collision - follow up",
                    "vitals": {"HR": "145 bpm", "BP": "75/48 mmHg", "SpO2": "85%", "Temp": "36.2°C", "RR": "32/min"},
                    "visible_symptoms": ["obvious left arm deformity", "lacerations", "abdominal bruising"],
                    "history_revealed": ["ejected from vehicle", "Decreased breath sounds left side", "JVD noted"],
                    "step_number": 2,
                },
            ],
            "expected": {
                "triage_level": "immediate",
                "condition_keywords": ["polytrauma", "trauma", "pneumothorax"],
                "never_under_triage": True,
            }
        },
    ],
}


def check_condition(keyword: str, text: str) -> bool:
    """Case-insensitive keyword check"""
    return keyword.lower() in text.lower()


def evaluate_action(action: Dict, expected: Dict, case_name: str) -> Tuple[bool, List[str]]:
    """Evaluate model action against expected criteria"""
    issues = []
    
    # Check triage level
    if action.get("triage_level") != expected["triage_level"]:
        issues.append(f"❌ Triage level: expected '{expected['triage_level']}', got '{action.get('triage_level')}'")
        if expected.get("never_under_triage"):
            order = {"non_urgent": 0, "less_urgent": 1, "urgent": 2, "immediate": 3}
            if order.get(action.get("triage_level"), -1) < order.get(expected["triage_level"], 4):
                issues.append("🚨 CRITICAL: UNDER-TRIAGE DETECTED - This is dangerous!")
    else:
        issues.append(f"✅ Triage level correct: {expected['triage_level']}")
    
    # Check condition keywords
    condition = action.get("suspected_condition", "").lower()
    matched = any(check_condition(kw, condition) for kw in expected.get("condition_keywords", []))
    if expected.get("condition_keywords") and not matched:
        issues.append(f"⚠️  Condition: expected keywords {expected['condition_keywords']}, got '{action.get('suspected_condition')}'")
    else:
        issues.append(f"✅ Condition recognized: {action.get('suspected_condition')}")
    
    # Check required tests
    recommended = [t.lower() for t in action.get("recommended_tests", [])]
    for test in expected.get("required_tests", []):
        test_lower = test.lower()
        found = any(test_lower in r or r in test_lower for r in recommended)
        if not found:
            issues.append(f"⚠️  Missing test: {test}")
        else:
            issues.append(f"✅ Test recommended: {test}")
    
    # Check reasoning length
    reasoning = action.get("reasoning", "")
    min_len = expected.get("min_reasoning_length", 0)
    if len(reasoning) < min_len:
        issues.append(f"⚠️  Reasoning too short: {len(reasoning)} chars (expected >{min_len})")
    else:
        issues.append(f"✅ Reasoning length: {len(reasoning)} chars")
    
    passed = not any(line.startswith("❌") or line.startswith("🚨") for line in issues)
    return passed, issues


def test_task(task_name: str, cases: List[Dict]) -> Dict:
    """Test all cases for a task"""
    print(f"\n{'='*70}")
    print(f"Testing Task: {task_name}")
    print(f"{'='*70}\n")
    
    results = {"passed": 0, "failed": 0, "details": []}
    
    for case in cases:
        print(f"Case: {case['name']}")
        print("-" * 70)
        
        try:
            # Reset environment
            reset_resp = requests.post(f"{BASE_URL}/reset", json={"task_name": task_name})
            if reset_resp.status_code != 200:
                print(f"❌ Reset failed: {reset_resp.text}")
                results["failed"] += 1
                continue
            
            # For multi-step observations, test each one
            observations = case.get("observations", [case.get("observation")])
            passed = True
            
            for obs in observations:
                # Get model prediction
                action = get_model_action(obs)
                
                step_resp = requests.post(f"{BASE_URL}/step", json={"action": action})
                if step_resp.status_code != 200:
                    print(f"❌ Step failed: {step_resp.text}")
                    passed = False
                    break
                
                response = step_resp.json()
                observation = response.get("observation", {})
                feedback = response.get("feedback", "")
                
                # Evaluate
                action_passed, issues = evaluate_action(action, case["expected"], case["name"])
                for issue in issues:
                    print(f"  {issue}")
                
                if not action_passed:
                    passed = False
                
                print(f"  Feedback: {feedback}")
                print()
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            results["details"].append({"case": case["name"], "passed": passed})
            
        except Exception as e:
            print(f"❌ Error: {e}\n")
            results["failed"] += 1
    
    return results


def main():
    print("\n🏥 Clinical Triage Model Test Suite")
    print("=" * 70)
    print(f"Testing against: {BASE_URL}\n")
    
    # Check if model is configured
    if not API_KEY:
        print("⚠️  WARNING: API_KEY or HF_TOKEN not set. Install your model API credentials:\n")
        print("  Windows (PowerShell):")
        print("    $env:API_KEY='your-token-here'")
        print("    python test_model.py\n")
        print("  Windows (CMD):")
        print("    set API_KEY=your-token-here")
        print("    python test_model.py\n")
        print("  Linux/Mac:")
        print("    export API_KEY='your-token-here'")
        print("    python test_model.py\n")
        return
    
    # Check server is running
    try:
        health = requests.get(f"{BASE_URL}/")
        print(f"✅ Server running: {health.json()['name']} v{health.json()['version']}")
        print(f"✅ Model: {MODEL_NAME}")
        print(f"✅ API Base: {API_BASE_URL}\n")
    except:
        print("❌ ERROR: Server not running at", BASE_URL)
        print("   Start server with: uvicorn clinical_triage_env.server.app:app --port 7860")
        return
    
    all_results = {}
    total_passed = 0
    total_failed = 0
    
    # Run tests for each task
    for task_name, cases in TEST_CASES.items():
        results = test_task(task_name, cases)
        all_results[task_name] = results
        total_passed += results["passed"]
        total_failed += results["failed"]
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for task_name, results in all_results.items():
        total = results["passed"] + results["failed"]
        pct = (results["passed"] / total * 100) if total > 0 else 0
        status = "✅ PASS" if pct >= 75 else "❌ FAIL"
        print(f"{task_name:30} {results['passed']}/{total} ({pct:.0f}%) {status}")
    
    overall = total_passed / (total_passed + total_failed) * 100 if (total_passed + total_failed) > 0 else 0
    print(f"\n{'='*70}")
    print(f"Overall: {total_passed}/{total_passed + total_failed} ({overall:.0f}%)")
    if overall >= 75:
        print("✅ MODEL READY FOR SUBMISSION")
    else:
        print("❌ MODEL NEEDS MORE WORK")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
