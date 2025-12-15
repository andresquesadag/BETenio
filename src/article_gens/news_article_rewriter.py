# V2.0 | 2025-12

import requests
import os
import re
import time
import pandas as pd
from typing import Optional


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
You are a professional journalist from Spain.
The user provides a text written in Spanish by a human. The text is a news article describing a real event and contains factual information.
Your task is to rewrite the article into a clear, well-structured news article written in your own words.
Rules:
- The article must be written in Spanish.
- Preserve all factual information exactly.
- Do NOT add new facts, interpretations, or context.
- Improve clarity, structure, and narrative flow.
- Maintain a neutral and objective journalistic tone.
- Do NOT remove or omit any factual detail, even if it seems redundant or editorial.
- If the original text contains image captions, parenthetical notes, or reported quotations, they must be preserved or faithfully paraphrased.
- Do NOT include a headline.
- The rewritten article must be written as continuous prose in a single line.
- Do NOT use line breaks, paragraphs, lists, or bullet points.
- The length of the rewritten article should be approximately similar to the original.
- Do NOT mention that the text has been rewritten or generated.
Output only the rewritten article text.
Original text:
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
