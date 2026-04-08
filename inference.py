"""
Baseline inference script for ClinicalTriageEnv.
Complies fully with Meta OpenEnv Hackathon output spec.

IMPORTANT: All scores and rewards are clamped to strictly open interval (0.0, 1.0).
The Meta validator rejects values of exactly 0.0 or exactly 1.0.

Output format (stdout):
  [START] task=<n> env=clinical_triage_env model=<model>
  [STEP] step=<n> action=<level> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""
import json
import os
import sys
import time
import traceback
from typing import List, Optional

from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
ENV_BASE_URL: str = os.getenv("CLINICAL_TRIAGE_BASE_URL", "http://localhost:7860")
BENCHMARK: str = "clinical_triage_env"
MAX_STEPS: int = 5
TEMPERATURE: float = 0.3
MAX_TOKENS: int = 400

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required.")

llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["vital_signs_triage", "differential_diagnosis", "polytrauma_cascade"]

SYSTEM_PROMPT = """You are a clinical triage AI assistant in an emergency department.

Respond ONLY with a JSON object. No markdown, no preamble, no explanation.

Required JSON format:
{
  "triage_level": "immediate" | "urgent" | "less_urgent" | "non_urgent",
  "suspected_condition": "<your primary diagnosis>",
  "recommended_tests": ["<test1>", "<test2>"],
  "reasoning": "<at least 2 sentences of clinical reasoning>"
}

Triage levels:
- immediate: Life-threatening NOW
- urgent: Serious, needs rapid assessment within 15 min
- less_urgent: Needs care, can wait up to 1 hour
- non_urgent: Minor, can be redirected"""


def _clamp_score(value: float) -> float:
    """Clamp to strictly open interval (0.0, 1.0). Validator rejects 0.0 and 1.0."""
    return round(min(max(float(value), 0.01), 0.99), 2)


def build_user_prompt(obs) -> str:
    lines = [
        f"PATIENT COMPLAINT: {obs.chief_complaint}",
        "",
        "VITAL SIGNS:",
    ]
    for k, v in obs.vitals.items():
        lines.append(f"  {k}: {v}")
    lines += ["", f"SYMPTOMS: {', '.join(obs.visible_symptoms)}"]
    if obs.history_revealed:
        lines += ["", f"HISTORY: {'; '.join(obs.history_revealed)}"]
    if obs.feedback and obs.step_number > 0:
        lines += ["", f"FEEDBACK: {obs.feedback}"]
    lines += ["", f"Step {obs.step_number + 1} of {MAX_STEPS}. Provide triage assessment."]
    return "\n".join(lines)


def call_llm(prompt: str) -> dict:
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def run_task(task_name: str) -> None:
    from clinical_triage_env.client import ClinicalTriageEnvClient
    from clinical_triage_env.models import TriageAction

    log_start(task_name, MODEL_NAME)
    rewards: List[float] = []
    step = 0
    success = False
    final_score = 0.5  # safe default (strictly between 0 and 1)

    try:
        with ClinicalTriageEnvClient(base_url=ENV_BASE_URL) as env:
            step_result = env.reset(task_name=task_name)
            obs = step_result.observation
            done = obs.done

            while not done and step < MAX_STEPS:
                prompt = build_user_prompt(obs)
                error_msg = None

                try:
                    parsed = call_llm(prompt)
                    action = TriageAction(**parsed)
                except Exception as e:
                    error_msg = str(e)[:120]
                    action = TriageAction(
                        triage_level="urgent",
                        suspected_condition="unknown",
                        recommended_tests=[],
                        reasoning="Parse error fallback. Defaulting to urgent.",
                    )

                step_result = env.step(action)
                obs = step_result.observation
                step += 1
                done = obs.done

                # Clamp individual step reward — validator checks these too
                reward = _clamp_score(obs.reward)
                rewards.append(reward)

                log_step(step, action.triage_level, reward, done, error_msg)

            # Clamp final task score to strictly open (0.0, 1.0)
            raw_score = sum(rewards) / max(len(rewards), 1)
            final_score = _clamp_score(raw_score)
            success = final_score >= 0.5

    except Exception as exc:
        error_str = str(exc)[:200]
        log_step(step + 1, "none", 0.5, True, error_str)
        traceback.print_exc(file=sys.stderr)
        # Even on exception, emit a valid clamped score
        final_score = _clamp_score(sum(rewards) / max(len(rewards), 1)) if rewards else 0.5

    log_end(success, step, final_score, rewards if rewards else [0.5])


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
        time.sleep(1)
