from google import genai

CRISIS_PROMPT = """Analyze the following user message to determine if it indicates a crisis, such as self-harm, suicidal thoughts, or severe distress. Respond with a single word: \"CRISIS\" if it is, or \"NO_CRISIS\" if it is not. User text: \"{text}\""""

class CrisisAgent:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.0-flash'

    def is_crisis(self, text):
        response = self.client.models.generate_content(
            model=self.model,
            contents=CRISIS_PROMPT.format(text=text)
        )
        result = response.text.strip().upper()
        return result.startswith("CRISIS")
