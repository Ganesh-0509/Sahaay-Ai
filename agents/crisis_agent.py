import google.generativeai as genai

CRISIS_PROMPT = """Analyze the following user message to determine if it indicates a crisis, such as self-harm, suicidal thoughts, or severe distress. Respond with a single word: \"CRISIS\" if it is, or \"NO_CRISIS\" if it is not. User text: \"{text}\""""

class CrisisAgent:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def is_crisis(self, text):
        response = self.model.generate_content(CRISIS_PROMPT.format(text=text))
        result = response.text.strip().upper()
        return result.startswith("CRISIS")
