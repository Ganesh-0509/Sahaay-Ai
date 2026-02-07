from google import genai
import json

SYSTEM_PROMPT = """You are Sahaay-AI, a kind, happy, and supportive mental health companion for Indian youth. Your primary role is to be a welcoming and non-judgmental friend. Your friendly and empathetic tone should always shine through.\n\n**INSTRUCTION: Your entire response must be a single JSON object. Do not include any text, conversation, or markdown before or after the JSON. Do not include any explanation.**\n\nThe JSON object must have one key:\n- **response**: Your brief, empathetic, and conversational reply to the user.\n\nExample:\nUser: I had a great day today!\nYour Response:\n{\n  \"response\": \"Oh, that's fantastic! Tell me all about it—what made your day so great?\"\n}\n"""

class GeminiAgent:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API key cannot be None or empty")
            
        # Clean the API key of any whitespace or quotes
        api_key = api_key.strip().strip('"\'')
        
        try:
            # Initialize the client
            self.client = genai.Client(api_key=api_key)
            self.model = "gemini-2.0-flash"  # Using the latest flash model
            
        except Exception as e:
            print(f"Error initializing Gemini client: {str(e)}")
            raise ValueError(f"Failed to initialize Gemini client: {str(e)}")
            
        self.system_prompt = SYSTEM_PROMPT

    def get_response(self, message, history=None, language='en'):
        try:
            message_text = message.get('content', message) if isinstance(message, dict) else message

            lang_instructions = {
                'hi': "\n\nIMPORTANT: Respond in Hindi (हिन्दी).",
                'ta': "\n\nIMPORTANT: Respond in Tamil (தமிழ்).",
                'te': "\n\nIMPORTANT: Respond in Telugu (తెలుగు).",
                'en': ""
            }
            lang_instruction = lang_instructions.get(language, "")

            prompt = f"{self.system_prompt}{lang_instruction}\n\nUser: {message_text}"

            # Try primary model first, then fallback to 2.5-flash, then 2.0-flash-lite
            models_to_try = [self.model, "gemini-2.5-flash", "gemini-2.0-flash-lite"]
            response = None
            last_error = ""

            for model in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt
                    )
                    if response and response.text:
                        break # Success!
                except Exception as e:
                    last_error = str(e)
                    if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                        print(f"Quota exceeded for {model}, trying fallback...")
                        continue
                    else:
                        raise # Rethrow non-quota errors

            if response and response.text:
                response_text = response.text.strip()
                # Clean markdown wrappers
                response_text = response_text.replace("```json", "").replace("```", "").strip()

                try:
                    parsed_json = json.loads(response_text)
                    return json.dumps({"response": parsed_json.get("response", "")})
                except:
                    return json.dumps({"response": response_text})
            else:
                return json.dumps({
                    "response": "I'm currently at my limit for free AI responses. Please try again in a few minutes! 🙏"
                })

        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                return json.dumps({
                    "response": "The AI is a bit busy right now (quota reached). Please try again in a moment! 🌱"
                })
            return json.dumps({
                "response": "Something went wrong. Please try again."
            })
