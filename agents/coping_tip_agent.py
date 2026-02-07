from google import genai

class CopingTipAgent:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.0-flash'

    def generate_tip(self, mood_text):
        try:
            prompt = f"Provide a single, specific, and actionable coping tip for someone feeling {mood_text}. Respond with only the tip text, no other information."
            models_to_try = [self.model, "gemini-2.5-flash", "gemini-2.0-flash-lite"]
            for model in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt
                    )
                    return response.text.strip()
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print(f"Coping tip quota exceeded for {model}, trying fallback...")
                        continue
                    raise
            return "Take a deep breath and a moment for yourself."
        except Exception as e:
            print(f"Coping tip error: {e}")
            return "Take a deep breath and a moment for yourself."
