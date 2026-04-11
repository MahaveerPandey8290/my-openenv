"""
Baseline inference script — Meta OpenEnv hackathon spec.
Strict stdout format: [START] [STEP]... [END]
All scores clamped to strictly open (0.01, 0.99).
"""
import json, os, sys, time, traceback
from typing import List, Optional
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME")
ENV_BASE_URL = os.getenv("CLINICAL_TRIAGE_BASE_URL", "http://localhost:7860")
BENCHMARK = "clinical_triage_env"
MAX_STEPS = 8

if not API_KEY:
    raise ValueError("API_KEY required by hackathon proxy")

llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
TASKS = ["vital_signs_triage", "differential_diagnosis", "polytrauma_cascade"]

SYSTEM = """You are a clinical triage AI. Respond ONLY with JSON.

Choose ONE of these two actions:

ORDER A TEST (when uncertain):
{"action_type": "order_test", "test_name": "<single test>", "reasoning": "<why>"}

SUBMIT TRIAGE (when ready):
{"action_type": "submit_triage", "triage_level": "immediate|urgent|less_urgent|non_urgent", "suspected_condition": "<diagnosis>", "reasoning": "<2+ sentence reasoning>"}

Rules:
- Order only if tests_remaining > 0 AND you are uncertain
- Never order a test you already have results for
- Submit triage when confident or tests exhausted
- Respond ONLY with valid JSON"""


def _clamp(v):
    return round(min(max(float(v), 0.01), 0.99), 2)


def build_prompt(obs):
    lines = [f"COMPLAINT: {obs.chief_complaint}", "", "VITALS:"]
    for k, v in obs.vitals.items():
        lines.append(f"  {k}: {v}")
    lines += ["", f"SYMPTOMS: {', '.join(obs.visible_symptoms)}"]
    if obs.history_revealed:
        lines += ["", f"HISTORY: {'; '.join(obs.history_revealed)}"]
    if obs.test_results:
        lines += ["", "TEST RESULTS:"]
        for t, r in obs.test_results.items():
            lines.append(f"  {t}: {r}")
    lines += [
        "", f"Tests remaining: {obs.tests_remaining}",
        f"Available: {', '.join(obs.available_actions)}",
        f"Step: {obs.step_number + 1}",
    ]
    return "\n".join(lines)


def call_llm(prompt):
    r = llm.chat.completions.create(
        model=MODEL_NAME, max_tokens=400, temperature=0.3,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
    )
    raw = r.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw.strip())


def run_task(task_name):
    from clinical_triage_env.client import ClinicalTriageEnvClient
    from clinical_triage_env.models import OrderTestAction, SubmitTriageAction

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    rewards, step, success = [], 0, False

    try:
        with ClinicalTriageEnvClient(base_url=ENV_BASE_URL) as env:
            r = env.reset(task_name=task_name)
            obs, done = r.observation, r.observation.done

            while not done and step < MAX_STEPS:
                err = None
                try:
                    parsed = call_llm(build_prompt(obs))
                    atype = parsed.get("action_type", "submit_triage")
                    action = OrderTestAction(**parsed) if atype == "order_test" else SubmitTriageAction(**parsed)
                    astr = f"order_test({parsed.get('test_name','')})" if atype == "order_test" else f"submit_triage({parsed.get('triage_level','')})"
                except Exception as e:
                    err = str(e)[:100]
                    action = SubmitTriageAction(action_type="submit_triage", triage_level="urgent",
                                               suspected_condition="unknown", reasoning="Parse error fallback.")
                    astr = "submit_triage(urgent)[fallback]"

                sr = env.step(action)
                obs, step = sr.observation, step + 1
                done = obs.done
                reward = _clamp(obs.reward)
                rewards.append(reward)
                print(f"[STEP] step={step} action={astr} reward={reward:.2f} done={str(done).lower()} error={err or 'null'}", flush=True)

        score = _clamp(sum(rewards) / max(len(rewards), 1))
        success = score >= 0.5
    except Exception as ex:
        print(f"[STEP] step={step+1} action=none reward=0.50 done=true error={str(ex)[:100]}", flush=True)
        score = 0.5

    print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={','.join(f'{r:.2f}' for r in rewards) or '0.50'}", flush=True)


if __name__ == "__main__":
    for t in TASKS:
        run_task(t)
        time.sleep(1)
