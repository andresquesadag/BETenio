# V2.0 | 2025-12
import requests
import os
import re
import time
import pandas as pd
from typing import Optional

"""
LM STUDIO CONFIGURATION (AUTHORSHIP MODE – IA AS AUTHOR)

Purpose:
- Generate AI-authored news articles using human-written articles as a source of information.
- The AI acts as the author, not as an editor or paraphraser.
- Configuration balances authorial freedom, factual fidelity, and runtime (< ~2 minutes per article).

Model / Context settings:
- Context length: 4096 tokens
- GPU offload: 45 / 48 layers
- CPU thread pool size: 6
- Evaluation batch size: 512
- RoPE frequency base/scale: Auto
- KV cache offloaded to GPU: ON
- Keep model in memory: ON
- Memory mapping (mmap): ON
- Flash Attention: OFF (stability over experimental speedups)
- K/V cache quantization: Experimental (enabled)

Inference / Sampling settings:
- Temperature: 0.4
- Top-K sampling: 50
- Top-P sampling: 0.95
- Repeat penalty: OFF
- Min-P sampling: OFF
- Limit response length: OFF (handled implicitly by prompt + runtime constraints)

Rationale:
- Temperature 0.4 provides sufficient editorial freedom for the AI to act as an author
  while avoiding uncontrolled hallucinations.
- Top-K 50 and Top-P 0.95 prevent generation stalls and ensure timely completion.
- Repeat penalty disabled to allow natural repetition of names, entities, and figures
  common in factual news reporting.
- No explicit stylistic constraints (e.g., punctuation rules) to preserve natural
  authorial signals for authorship attribution experiments.

Status:
- Configuration validated empirically for quality, factual fidelity, and runtime.
- Parameters frozen for large-scale dataset generation.
"""


class LMStudioClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234",
        model: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.models_url = f"{self.base_url}/v1/models"
        self.session = requests.Session()
        self.model = model or self._get_default_model()

    def _get_default_model(self):
        models = self.get_available_models()
        if not models:
            raise RuntimeError("No models available in LM Studio.")
        print(f"Using model: {models[0]}")
        return models[0]

    def get_available_models(self):
        try:
            response = self.session.get(self.models_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            print(f"Model fetch error: {e}")
            return []

    def rewrite_article(
        self,
        article_text: str,
        max_tokens: int = 4000,
        temperature: float = 0.4,
    ) -> Optional[str]:

        prompt = f"""
You are a journalist from Spain.
The user provides a Spanish news article written by a human, which may include image captions or descriptive lines. Use the text only as a source of information.
Write your own news article about the same event.
Rules:
1. Write in Spanish.
2. Preserve all factual information (events, dates, numbers, names).
3. Do NOT add new facts or speculation.
4. You may reorder, regroup, and rephrase information freely.
5. If the source text contains image captions or descriptive elements, incorporate their information naturally into your article.
6. Maintain a neutral, professional journalistic tone.
7. Do NOT include a headline.
8. Write continuous prose in a single line.
9. The article should be similar in length to the original.
10. Output only the article text.

Source text:
{article_text}
"""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
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
            text = data["choices"][0]["message"]["content"]
            # Enforce single-line output
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as e:
            print(f"Generation error: {e}")
            return None


def main():
    INPUT_CSV_PATH = "human_es_news.csv"
    OUTPUT_BASE_DIR = "output"

    # Ask user for row range
    startrow = int(input("Enter start row index (inclusive): ").strip())
    endrow = int(input("Enter end row index (inclusive): ").strip())

    client = LMStudioClient()

    # Prepare output directory based on model name
    model_name = client.model
    safe_model_name = model_name.replace("/", "_")
    model_output_dir = os.path.join(OUTPUT_BASE_DIR, safe_model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(INPUT_CSV_PATH)

    # Select requested slice by original index
    df_slice = df.loc[startrow:endrow]

    print(
        f"\nProcessing rows {startrow} to {endrow} "
        f"({len(df_slice)} articles) using model '{model_name}'\n"
    )

    start_time = time.time()
    processed = 0

    for idx, row in df_slice.iterrows():
        article_text = row["cuerpo"]

        rewritten = client.rewrite_article(article_text)
        if not rewritten:
            print(f"[WARNING] Failed to rewrite row {idx}")
            continue

        filename = f"{idx:06d}.txt"
        filepath = os.path.join(model_output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(rewritten)

        processed += 1
        print(f"Saved: {safe_model_name}/{filename}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PROCESS COMPLETED")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Rows processed: {processed}")
    print(f"Time elapsed: {elapsed / 60:.2f} minutes")
    if processed > 0:
        print(f"Average per article: {elapsed / processed:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
