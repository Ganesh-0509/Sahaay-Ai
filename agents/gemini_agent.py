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
            
            # Test the configuration
            test_response = self.client.models.generate_content(
                model=self.model,
                contents="Test connection"
            )
            if not test_response:
                raise Exception("Failed to generate test response")
                
        except Exception as e:
            print(f"Error initializing Gemini model: {str(e)}")
            raise ValueError(f"Failed to initialize Gemini model: {str(e)}")
            
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

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            if response and response.text:
                response_text = response.text.strip()

                # Clean markdown wrappers
                response_text = response_text.replace("```json", "").replace("```", "").strip()

                try:
                    parsed_json = json.loads(response_text)

                    # ALWAYS return JSON wrapper
                    return json.dumps({
                        "response": parsed_json.get("response", "")
                    })

                except:
                    # Force wrap into JSON
                    return json.dumps({
                        "response": response_text
                    })

            else:
                return json.dumps({
                    "response": "I'm sorry, I'm unable to reply right now."
                })

        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return json.dumps({
                "response": "Something went wrong. Please try again."
            })
