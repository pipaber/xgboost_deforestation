import requests
import json
import time

print("Waiting for API...")
# Simple retry loop
for _ in range(5):
    try:
        resp = requests.post(
            "http://localhost:8000/predict/aggregate",
            json={
                "group_by": "department",
                "mode": "hindcast",
                "overrides": {"YEAR": 2024},
                "hindcast_factors": {"mineria_factor": 1.2},
            },
        )
        if resp.status_code == 200:
            print("SUCCESS: API accepted request with hindcast_factors.")
            print("Total HA:", resp.json().get("total_pred_ha"))
            break
        else:
            print(f"FAILED: Status {resp.status_code}")
            print(resp.text)
            break
    except Exception as e:
        print(f"Connection failed: {e}")
        time.sleep(1)
