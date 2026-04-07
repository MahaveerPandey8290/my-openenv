"""
Baseline inference script for ClinicalTriageEnv.
Complies fully with Meta OpenEnv Hackathon output spec.

Output format (stdout):
  [START] task=<name> env=clinical_triage_env model=<model>
  [STEP] step=<n> action=<level> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Environment variables:
  API_BASE_URL  - LLM API base (default: HuggingFace router)
  MODEL_NAME    - Model identifier (default: Qwen2.5-72B-Instruct)
  HF_TOKEN      - API key (REQUIRED, no default)
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

# ── LLM client ────────────────────────────────────────────────────────────────
llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["vital_signs_triage", "differential_diagnosis", "polytrauma_cascade"]

SYSTEM_PROMPT = """You are a clinical triage AI assistant operating in an emergency department.

You will receive patient information incrementally. Your job is to assess the patient and respond with a JSON object ONLY — no preamble, no markdown, no explanation outside the JSON.

Required JSON format:
{
  "triage_level": "immediate" | "urgent" | "less_urgent" | "non_urgent",
  "suspected_condition": "<your primary diagnosis as a string>",
  "recommended_tests": ["<test1>", "<test2>", ...],
  "reasoning": "<your clinical reasoning, at least 2 sentences>"
}

Triage level definitions:
- immediate: Life-threatening, requires intervention NOW (e.g. cardiac arrest, tension pneumothorax, major haemorrhage)
- urgent: Serious condition, stable but needs rapid assessment within 15 minutes
- less_urgent: Needs care but can wait up to 1 hour
- non_urgent: Minor, can wait or be redirected to GP/walk-in

Respond ONLY with valid JSON. No other text."""


def build_user_prompt(obs) -> str:
    """Convert observation to natural language prompt for the LLM."""
    lines = [
        f"PATIENT COMPLAINT: {obs.chief_complaint}",
        "",
        "VITAL SIGNS:",
    ]
    for k, v in obs.vitals.items():
        lines.append(f"  {k}: {v}")
    lines += [
        "",
        f"CURRENT SYMPTOMS: {', '.join(obs.visible_symptoms)}",
    ]
    if obs.history_revealed:
        lines += ["", f"HISTORY (revealed so far): {'; '.join(obs.history_revealed)}"]
    if obs.feedback and obs.step_number > 0:
        lines += ["", f"PREVIOUS FEEDBACK: {obs.feedback}"]
    lines += ["", f"Step {obs.step_number + 1} of {MAX_STEPS}. Provide your triage assessment."]
    return "\n".join(lines)


def call_llm(prompt: str) -> dict:
    """Call LLM and parse JSON response."""
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
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    done_str = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def run_task(task_name: str) -> None:
    """Run one full episode for a task using the local environment server."""
    from clinical_triage_env.client import ClinicalTriageEnvClient
    from clinical_triage_env.models import TriageAction

    log_start(task_name, MODEL_NAME)
    rewards: List[float] = []
    step = 0
    success = False

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
                    # Fallback action on parse error
                    action = TriageAction(
                        triage_level="urgent",
                        suspected_condition="unknown",
                        recommended_tests=[],
                        reasoning="Parse error fallback.",
                    )

                step_result = env.step(action)
                obs = step_result.observation
                step += 1
                reward = obs.reward
                done = obs.done
                rewards.append(reward)

                log_step(step, action.triage_level, reward, done, error_msg)

            final_score = obs.cumulative_reward / max(step, 1)
            success = final_score >= 0.5

    except Exception as exc:
        error_str = str(exc)[:200]
        log_step(step + 1, "none", 0.0, True, error_str)
        traceback.print_exc(file=sys.stderr)

    log_end(success, step, sum(rewards) / max(len(rewards), 1), rewards)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
        time.sleep(1)
