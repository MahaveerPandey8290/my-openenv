---
title: Clinical Triage Env
emoji: 🏥
colorFrom: red
colorTo: white
sdk: docker
pinned: true
tags:
  - openenv
---

# 🏥 ClinicalTriageEnv — OpenEnv RL Environment

A clinical decision-support RL environment where AI agents perform emergency
triage on synthetic patients. Submitted to the **Meta PyTorch OpenEnv Hackathon**.

## Overview

Agents must assess synthetic patient presentations and:
1. Classify urgency level (`immediate` / `urgent` / `less_urgent` / `non_urgent`)
2. Identify the most likely condition
3. Recommend appropriate diagnostic tests

Patient information is revealed **incrementally** — making this a sequential
decision-making problem, not just a classification task.

## Action Space

```json
{
  "triage_level": "immediate | urgent | less_urgent | non_urgent",
  "suspected_condition": "string",
  "recommended_tests": ["ECG", "troponin", ...],
  "reasoning": "chain-of-thought string"
}
```

## Observation Space

```json
{
  "patient_id": "uuid",
  "chief_complaint": "string",
  "vitals": {"HR": "...", "BP": "...", "SpO2": "...", "Temp": "...", "RR": "..."},
  "visible_symptoms": ["symptom1", ...],
  "history_revealed": ["history item", ...],
  "step_number": 0,
  "reward": 0.0,
  "done": false,
  "feedback": "string",
  "cumulative_reward": 0.0
}
```

## Tasks

| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| `vital_signs_triage` | Easy | 3 | Classify urgency from vitals + chief complaint |
| `differential_diagnosis` | Medium | 5 | Identify condition + recommend tests |
| `polytrauma_cascade` | Hard | 5 | Multi-system emergency, cascading findings |

## Reward Function

Shaped reward (0.0–1.0) at every step:
- **Triage level accuracy**: +0.4 to +0.6 for correct level
- **Condition match**: +0.2 if suspected category correct  
- **Test quality**: up to +0.25 based on relevant test overlap
- **Efficiency bonus**: +0.1 for early correct answer
- **Under-triage penalty**: −0.25 to −0.4 (dangerous in real triage)

## Setup

```bash
pip install openenv-core
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/clinical-triage-env
cd clinical-triage-env
pip install -e .
uvicorn clinical_triage_env.server.app:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t clinical-triage-env .
docker run -p 7860:7860 clinical-triage-env
```

## Baseline Results

| Task | Baseline Model | Avg Score |
|------|---------------|-----------|
| vital_signs_triage | Qwen2.5-72B | 0.72 |
| differential_diagnosis | Qwen2.5-72B | 0.55 |
| polytrauma_cascade | Qwen2.5-72B | 0.38 |

## Environment Variables

| Variable | Default | Required |
|----------|---------|----------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | No |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | No |
| `HF_TOKEN` | — | Yes |
| `CLINICAL_TRIAGE_BASE_URL` | `http://localhost:7860` | No |
