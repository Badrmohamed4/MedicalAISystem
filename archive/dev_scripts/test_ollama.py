import requests
host = "http://localhost:11434"
try:
    response = requests.get(host, timeout=2)
    print(f"Status Code: {response.status_code}")
    print(f"Content: {response.text}")
except Exception as e:
    print(f"Exception: {e}")
