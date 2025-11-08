from flask import Blueprint, request, jsonify, session, render_template
from flask_login import login_required, current_user
from agents.gemini_agent import GeminiAgent
from agents.crisis_agent import CrisisAgent
from agents.coping_tip_agent import CopingTipAgent
from utils.rate_limit import rate_limited
from utils.sentiment import get_sentiment_score, get_coarse_mood
from utils.emotion_classifier import classify_emotion
from models.user import User, AnonymousUser
import re, json

from translations.translation_utils import get_user_translations

chat_bp = Blueprint('chat', __name__)


# Render the chat UI (GET) and handle chat API (POST)
@chat_bp.route('/chat', methods=['GET'])
@login_required
def chat_page():
    return render_template('chat.html', user=current_user, lang=get_user_translations())

@chat_bp.route("/chat", methods=["POST"])
@login_required
def chat():
    user_id = current_user.id
    if rate_limited(user_id):
        return jsonify({'error': 'Rate limit exceeded. Please wait.'}), 429
    data = request.get_json()
    message = (data.get("message") or "").strip() if isinstance(data, dict) else ""
    lang = data.get("lang", "en") if isinstance(data, dict) else "en"

    if not message:
        return jsonify({"ok": False, "message": "No message provided."}), 400
    if len(message) > 2000:
        return jsonify({"ok": False, "message": "Message too long. Please keep it under 2000 characters."}), 413

    limiter_key = f"user:{user_id}" if current_user.is_authenticated else f"ip:{request.remote_addr}"
    if rate_limited(limiter_key):
        return jsonify({"ok": False, "message": "You are sending messages too quickly. Please wait a moment."}), 429

    # Sentiment and emotion
    sentiment_score = get_sentiment_score(message)
    coarse_mood = get_coarse_mood(sentiment_score)
    fine_mood = classify_emotion(message)
    mood = fine_mood if fine_mood else coarse_mood

    # Get Gemini response
    try:
        from flask import current_app
        api_key = current_app.config.get('GEMINI_API_KEY')
        
        if not api_key:
            return jsonify({"error": "Gemini API key not configured"}), 500
            
        # Initialize the Gemini agent
        try:
            gemini = GeminiAgent(api_key=api_key)
        except ValueError as ve:
            print(f"Gemini initialization error: {str(ve)}")
            return jsonify({"error": "Failed to initialize AI model. Please try again later."}), 500
        except Exception as e:
            print(f"Unexpected error initializing Gemini: {str(e)}")
            return jsonify({"error": "An unexpected error occurred"}), 500
        response_text = gemini.get_response(message)
        
        # Parse response
        if isinstance(response_text, str):
            try:
                response_data = json.loads(response_text)
            except json.JSONDecodeError:
                response_data = {"response": response_text}
        else:
            response_data = response_text if isinstance(response_text, dict) else {"response": str(response_text)}
            
        # Ensure we have a response field
        if "response" not in response_data:
            response_data = {"response": str(response_data)}
    except Exception as e:
        print(f"Error in chat processing: {str(e)}")
        response_data = {"response": "I apologize, but I encountered an error. Could you please try again?"}
    
    try:
        # Parse response as JSON if it isn't already
        if isinstance(response_text, str):
            response_data = json.loads(response_text)
        else:
            response_data = response_text
            
        # Ensure we have a response field
        if "response" not in response_data:
            response_data = {"response": str(response_data)}
            
    except json.JSONDecodeError:
        # If response isn't valid JSON, wrap it
        response_data = {"response": response_text}

    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            response_json = json.loads(match.group(0).replace("```json", "").replace("```", ""))
        except:
            response_json = {"response": response_text}
    else:
        response_json = {"response": response_text}

    # Crisis detection
    crisis_agent = CrisisAgent(api_key="YOUR_GEMINI_API_KEY")
    crisis = crisis_agent.is_crisis(message)

    final_response = {
        "mood": mood,
        "intent": "casual_chat",
        "sentiment": sentiment_score,
        "response": response_json.get("response", "I'm here to listen. Can you tell me more?")
    }
    final_response["crisis_detected"] = crisis
    return jsonify({"ok": True, "data": final_response})
