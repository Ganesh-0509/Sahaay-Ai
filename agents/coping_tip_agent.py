import google.generativeai as genai

class CopingTipAgent:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def generate_tip(self, mood_text):
        prompt = f"Provide a single, specific, and actionable coping tip for someone feeling {mood_text}. Respond with only the tip text, no other information."
        response = self.model.generate_content(prompt)
        return response.text.strip()
