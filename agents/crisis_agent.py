from google import genai

CRISIS_PROMPT = """Analyze the following user message to determine if it indicates a crisis, such as self-harm, suicidal thoughts, or severe distress. Respond with a single word: \"CRISIS\" if it is, or \"NO_CRISIS\" if it is not. User text: \"{text}\""""

class CrisisAgent:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.0-flash'

    def is_crisis(self, text):
        try:
            models_to_try = [self.model, "gemini-2.5-flash", "gemini-2.0-flash-lite"]
            for model in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=CRISIS_PROMPT.format(text=text)
                    )
                    result = response.text.strip().upper()
                    return result.startswith("CRISIS")
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print(f"Crisis detection quota exceeded for {model}, trying fallback...")
                        continue
                    raise
            return False # Default to safe if both fail
        except Exception as e:
            print(f"Crisis detection error: {e}")
            return False
