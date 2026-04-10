import httpx
import json
import time

BASE_URL = "http://localhost:7860"

def test_reset_no_body():
    print("Testing POST /reset with NO body...")
    try:
        # Use content=b"" to simulate totally empty body
        response = httpx.post(f"{BASE_URL}/reset", content=b"")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS: /reset handled empty body.")
            return True
        else:
            print(f"FAILED: /reset returned {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_full_loop():
    print("\nTesting Full RL Loop...")
    try:
        # 1. Reset
        resp = httpx.post(f"{BASE_URL}/reset", json={"task_name": "vital_signs_triage"})
        if resp.status_code != 200:
            print("Reset failed")
            return False
        obs = resp.json()
        print(f"Initial Observation: {obs['chief_complaint']}")

        # 2. Step (Order Test)
        action = {
            "action": {
                "action_type": "order_test",
                "test_name": "ECG",
                "reasoning": "Checking for cardiac issues."
            }
        }
        resp = httpx.post(f"{BASE_URL}/step", json=action)
        print(f"Step (test) status: {resp.status_code}")
        if resp.status_code != 200:
            print(resp.text)
            return False
        
        # 3. Step (Submit Triage)
        action = {
            "action": {
                "action_type": "submit_triage",
                "triage_level": "immediate",
                "suspected_condition": "cardiac arrest or MI",
                "reasoning": "Vitals are unstable."
            }
        }
        resp = httpx.post(f"{BASE_URL}/step", json=action)
        print(f"Step (submit) status: {resp.status_code}")
        if resp.status_code == 200:
            print("SUCCESS: Full RL loop completed.")
            return True
        else:
            print(resp.text)
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    # Note: Server must be running for this to work
    print("Verifying server robustness...")
    s1 = test_reset_no_body()
    s2 = test_full_loop()
    
    if s1 and s2:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED - Check if server is running on :7860")
