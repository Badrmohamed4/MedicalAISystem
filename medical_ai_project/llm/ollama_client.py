import requests
import json

class OllamaClient:
    def __init__(self, host="http://localhost:11434", model="llama3"):
        self.host = host
        self.model = model
        self.endpoint = f"{self.host}/api/chat"

    def is_online(self):
        try:
            response = requests.get(self.host, timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def stream_chat(self, system_prompt, user_prompt):
        if not self.is_online():
            yield "Error: Ollama server is offline or unreachable. Please start it with 'ollama serve'."
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }

        try:
            with requests.post(self.endpoint, json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]
                        if chunk.get("done"):
                            break
        except requests.exceptions.RequestException as e:
            yield f"\nError communicating with Ollama: {str(e)}"

    def generate_json(self, system_prompt, user_prompt):
        """
        Synchronously requests a JSON response from Ollama.
        Useful for strict intent classification and entity extraction.
        """
        if not self.is_online():
            return {"error": "Ollama server is offline."}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(self.endpoint, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            if "message" in data and "content" in data["message"]:
                content = data["message"]["content"]
                # Try to parse the content as JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"error": "Failed to parse JSON from Ollama", "raw": content}
            return {"error": "Invalid response format from Ollama"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Error communicating with Ollama: {str(e)}"}
