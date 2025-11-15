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

    def get_response(self, message, history=None):
        try:
            # Extract message content
            message_text = message.get('content', message) if isinstance(message, dict) else message
            
            # Create the prompt
            prompt = f"{self.system_prompt}\n\nUser: {message_text}"
            
            # Generate response using the new API format
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            # Get the response text
            if response and response.text:
                response_text = response.text.strip()
                try:
                    # Try to parse as JSON
                    json_response = json.loads(response_text)
                    if isinstance(json_response, dict) and "response" in json_response:
                        return json_response["response"]
                except:
                    # If not valid JSON, wrap in our response format
                    return json.dumps({"response": response_text})
                
                return response_text
            else:
                return json.dumps({"response": "I apologize, but I wasn't able to generate a response. Could you try asking in a different way?"})
            
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return json.dumps({"response": "I apologize, but I encountered an error. Could you please try rephrasing your message?"})
