import requests
import json

# Test file path
file_path = "sample_pds.xlsx"

# Job requirements (adjust as needed)
job_req = {
    "education": 2,    # Bachelor's degree
    "experience": 2,   # 2 years experience
    "training": 3,     # At least 3 trainings
    "eligibility": 1   # Has eligibility
}

url = "http://localhost:8000/upload-pds"

with open(file_path, "rb") as f:
    files = {"file": f}
    data = job_req
    response = requests.post(url, files=files, data=data)

print("Status Code:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2))