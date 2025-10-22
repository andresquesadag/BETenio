# V1.0 | 2025-8

import requests
import json
import os
import re
from typing import Optional


class LMStudioClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234",
        model: Optional[str] = None,
        file_path: str = None,
    ):
        """
        Initialize the LM Studio client
        Args:
            base_url: The base URL of your LM Studio server (default: http://127.0.0.1:1234)
            model: Optional model name to use
            file_path: Optional path to file to load in memory
        """
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.models_url = f"{self.base_url}/v1/models"
        self.session = requests.Session()  # Reuse session for keep-alive
        self.model = model or self._get_default_model()
        self.file_content = self._load_file(file_path) if file_path else ""

    def _get_default_model(self):
        models = self.get_available_models()
        if not models:
            raise RuntimeError(
                "No models available. Make sure a model is loaded in LM Studio."
            )
        print(f"Using model: {models[0]}")
        return models[0]

    def _load_file(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_available_models(self):
        """Get list of available models from LM Studio"""
        try:
            response = self.session.get(self.models_url, timeout=10)
            response.raise_for_status()
            models_data = response.json()
            if "data" in models_data:
                return [model["id"] for model in models_data["data"]]
            return []
        except Exception as e:
            print(f"Could not fetch models: {e}")
            return []

    def send_message_with_file(
        self, message: str, max_tokens: int = 3000, temperature: float = 0.7
    ) -> Optional[str]:
        """
        Send a text message along with the preloaded file content to the LM Studio model
        """
        if not self.file_content:
            print("Warning: No file content loaded")

        combined_message = f"{message}\n\nFile content:\n{self.file_content}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": combined_message}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        try:
            response = self.session.post(
                self.chat_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=3600,
            )
            response.raise_for_status()
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                # Limpieza de texto eficiente
                text = data["choices"][0]["message"]["content"]
                text = re.sub(r"\s+", " ", text).strip()
                return text
            else:
                print("Error: Unexpected response format")
                print(json.dumps(data, indent=2))
                return None
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to LM Studio.")
            return None
        except requests.exceptions.Timeout:
            print("Error: Request timed out.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error: Request failed - {e}")
            return None
        except json.JSONDecodeError:
            print("Error: Invalid JSON response from server")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


# ================== Ejemplo de uso ==================
client = LMStudioClient(file_path="noticiasIA.csv")

message = (
    "You are a news writer. Task: generate EXACTLY ONE CSV line in the format: 0,0,<Body>,<Title>. "
    "NON-NEGOTIABLE RULES: "
    "(1) Both Body and Title must be written entirely in SPANISH. "
    "(2) Body length MUST be between 2500 and 5000 CHARACTERS (characters, not words). "
    "(3) Body must be continuous prose in paragraphs BUT WITHOUT any line breaks (no \\n and no \\r). Before output, replace any line breaks with a single space. Simulate paragraph changes with DOUBLE SPACES. "
    "(4) No tables, no lists, no bullets, no code—only narrative text. "
    "(5) To avoid breaking CSV fields, DO NOT use commas in Body or Title; use semicolons, colons, or em dashes (—) instead. "
    "(6) No URLs. "
    "(7) Title: short, informative, attractive, in Spanish, and without commas. "
    "(8) Do NOT attribute quotes to real people; explanatory neutral tone. "
    "(9) SELF-CHECK BEFORE OUTPUT: ensure Body length is within 2500–5000 characters AFTER removing any line breaks; ensure there are NO \\n or \\r characters; ensure exactly ONE line. "
    "(10) FINAL OUTPUT: print ONLY the single CSV line exactly as 0,0,<Body>,<Title> with no extra text before or after."
)


result_file_path = "noticiasIAF.csv"
batch_size = 100  # escribir cada 100 respuestas para mayor eficiencia

buffer = []
total_iterations = 1000

with open(result_file_path, "a", encoding="utf-8") as resultCSV:
    for i in range(total_iterations):
        response = client.send_message_with_file(message)
        if response:
            buffer.append(response)

        # Flush cada batch_size iteraciones
        if len(buffer) >= batch_size:
            resultCSV.write("\n".join(buffer) + "\n")
            buffer.clear()

        # Monitoreo
        if i % 100 == 0:
            print(f"Iteración {i} completada")

    # Flush final
    if buffer:
        resultCSV.write("\n".join(buffer) + "\n")

print("Proceso completado.")
