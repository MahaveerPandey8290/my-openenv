"""
Baseline inference for ClinicalTriageEnv 3D.
Uses VLM (vision-language model) to process base64 image + text.
Strictly follows Meta hackathon stdout spec.
"""
import base64, json, os, sys, time, traceback
from typing import List, Optional
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("CLINICAL_TRIAGE_BASE_URL", "http://localhost:7860")
BENCHMARK = "clinical_triage_3d"
MAX_STEPS = 20

if not HF_TOKEN:
    raise ValueError("HF_TOKEN required")

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
TASKS = ["single_patient_rescue", "ward_prioritisation", "mass_casualty_incident"]

SYSTEM = """You are an embodied medical AI navigating a 3D emergency department.
You receive a camera image of the ward plus text context.

Respond ONLY with valid JSON. Choose ONE action:

MOVE: {"action_type": "move_to", "target": "bed_1|bed_2|bed_3|bed_4|nurses_station|equipment_cart|exit", "reasoning": "..."}
EXAMINE: {"action_type": "examine_patient", "bed_id": "bed_1|...", "exam_type": "visual|vitals|auscultation|palpation", "reasoning": "..."}
ORDER TEST: {"action_type": "order_test", "bed_id": "bed_1|...", "test_name": "<test>", "reasoning": "..."}
INTERVENE: {"action_type": "intervene", "bed_id": "bed_1|...", "intervention": "oxygen_mask|iv_access|defib_pads|cervical_collar|tourniquet|bag_valve_mask", "reasoning": "..."}
TRIAGE: {"action_type": "submit_triage", "triage_assignments": {"bed_1": "immediate|urgent|less_urgent|non_urgent", ...}, "reasoning": "..."}

Look at the image: RED patients are critical. YELLOW are urgent. GREEN are stable.
Move toward red patients first. Submit triage when confident. Time costs reward."""


def _clamp(v):
    return round(min(max(float(v), 0.01), 0.99), 2)


def call_vlm(image_b64: str, text_context: str) -> dict:
    """Call VLM with image + text context."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}",
                "detail": "low",
            }},
            {"type": "text", "text": text_context},
        ]},
    ]
    r = llm.chat.completions.create(
        model=MODEL_NAME, max_tokens=300, temperature=0.2, messages=messages,
    )
    raw = r.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw.strip())


def build_text_context(obs_dict: dict) -> str:
    beds = obs_dict.get("beds_summary", {})
    nearby = obs_dict.get("nearby_beds", [])
    lines = [
        f"Location: {obs_dict.get('agent_location','?')}",
        f"Time remaining: {obs_dict.get('time_remaining_seconds', 0):.0f}s",
        f"Nearby beds: {', '.join(nearby) or 'none'}",
        "", "Beds overview:",
    ]
    for bid, bdata in beds.items():
        lines.append(
            f"  {bid}: alert={bdata.get('alert_level','?')} "
            f"dist={bdata.get('distance_metres','?')}m"
        )
        if bdata.get("chief_complaint"):
            lines.append(f"    complaint: {bdata['chief_complaint'][:50]}")
        if bdata.get("test_results"):
            for t, r in bdata["test_results"].items():
                lines.append(f"    {t}: {r[:60]}")
    lines += ["", f"Available actions: {', '.join(obs_dict.get('available_actions',[]))}"]
    return "\n".join(lines)


def run_task(task_name: str):
    import requests
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    rewards, step, success = [], 0, False

    try:
        r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_name": task_name}, timeout=30)
        data = r.json()
        obs = data["observation"]
        done = obs.get("done", False)

        while not done and step < MAX_STEPS:
            err = None
            try:
                parsed = call_vlm(obs["image_base64"], build_text_context(obs))
                atype = parsed.get("action_type", "submit_triage")
                astr = f"{atype}({parsed.get('target', parsed.get('bed_id', str(parsed.get('triage_assignments', '')))[:30])})"
            except Exception as e:
                err = str(e)[:80]
                parsed = {"action_type": "submit_triage",
                         "triage_assignments": {f"bed_{i+1}": "urgent" for i in range(4)},
                         "reasoning": "fallback"}
                astr = "submit_triage(fallback)"

            sr = requests.post(f"{ENV_BASE_URL}/step", json={"action": parsed}, timeout=30)
            sd = sr.json()
            obs = sd["observation"]
            step += 1
            done = sd.get("done", False)
            reward = _clamp(sd.get("reward", 0.5))
            rewards.append(reward)
            print(f"[STEP] step={step} action={astr} reward={reward:.2f} done={str(done).lower()} error={err or 'null'}", flush=True)

        score = _clamp(sum(rewards) / max(len(rewards), 1))
        success = score >= 0.5
    except Exception as ex:
        print(f"[STEP] step={step+1} action=none reward=0.50 done=true error={str(ex)[:80]}", flush=True)
        score = 0.5

    print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards) or '0.50'}", flush=True)


if __name__ == "__main__":
    for t in TASKS:
        run_task(t)
        time.sleep(2)
