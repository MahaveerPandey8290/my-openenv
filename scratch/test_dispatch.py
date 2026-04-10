from typing import Union, Literal
from pydantic import Field, RootModel, BaseModel

class OrderTestAction(BaseModel):
    action_type: Literal["order_test"] = "order_test"
    test_name: str

class SubmitTriageAction(BaseModel):
    action_type: Literal["submit_triage"] = "submit_triage"
    triage_level: str

class ClinicalAction(RootModel):
    root: Union[OrderTestAction, SubmitTriageAction] = Field(..., discriminator="action_type")

def test_dispatch(action):
    print(f"Testing: {type(action)}")
    
    # Logic in environment.py
    inner = getattr(action, "root", action)
    print(f"  Inner: {type(inner)}")
    
    action_type = None
    if isinstance(inner, dict):
        action_type = inner.get("action_type")
    else:
        action_type = getattr(inner, "action_type", None)
    
    print(f"  Action Type: {action_type}")
    
    if action_type is None and hasattr(action, "root"):
        print("  Falling back to manual root access")
        inner = action.root
        action_type = getattr(inner, "action_type", None)
        print(f"  Fallback Action Type: {action_type}")

# Simulate FastAPI/OpenEnv parsing
json_body = {"action_type": "submit_triage", "triage_level": "immediate"}
parsed = ClinicalAction.model_validate(json_body)

test_dispatch(parsed)
