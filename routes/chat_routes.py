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

    # Rate limit
    if rate_limited(user_id):
        return jsonify({'ok': False, 'message': 'Rate limit exceeded. Please wait.'}), 429

    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    lang = data.get("lang", "en")

    if not message:
        return jsonify({"ok": False, "message": "No message provided."}), 400

    if len(message) > 2000:
        return jsonify({"ok": False, "message": "Message too long. Keep it under 2000 chars."}), 413

    # Sentiment & emotion
    sentiment_score = get_sentiment_score(message)
    coarse_mood = get_coarse_mood(sentiment_score)
    
    # GEMINI client for emotion classification
    from flask import current_app
    api_key = current_app.config.get("GEMINI_API_KEY")
    
    if not api_key:
        return jsonify({"ok": False, "message": "AI model not configured."}), 500
    
    # Use LLM-based emotion detection with fallback
    from google import genai
    try:
        gemini_client = genai.Client(api_key=api_key)
        fine_mood = classify_emotion(message, client=gemini_client)
    except:
        fine_mood = classify_emotion(message)  # Fallback to keywords
    
    mood = fine_mood or coarse_mood

    # GEMINI for conversation

    if not api_key:
        return jsonify({"ok": False, "message": "AI model not configured."}), 500

    try:
        gemini = GeminiAgent(api_key=api_key)
        response_text = gemini.get_response(message, language=lang)

    except Exception as e:
        print("Gemini error:", str(e))
        return jsonify({
            "ok": True,
            "data": {
                "response": "I'm having trouble responding right now. Please try again.",
                "mood": mood,
                "sentiment": sentiment_score,
                "crisis_detected": False
            }
        })

    # Ensure JSON parsing and strip triple quotes if present
    cleaned_response = response_text.strip()

    # Strip Python triple quotes
    if cleaned_response.startswith('"""') and cleaned_response.endswith('"""'):
        cleaned_response = cleaned_response[3:-3].strip()

    # Strip Markdown code blocks (```json ... ```)
    cleaned_response = re.sub(r"^```(?:json)?", "", cleaned_response, flags=re.IGNORECASE).strip()
    cleaned_response = re.sub(r"```$", "", cleaned_response).strip()

    try:
        response_data = json.loads(cleaned_response)
    except Exception:
        response_data = {"response": cleaned_response}

    # Crisis detection uses real API key
    crisis_agent = CrisisAgent(api_key=api_key)
    crisis_detected = crisis_agent.is_crisis(message)

    # Generate coping tip
    coping_tip_agent = CopingTipAgent(api_key=api_key)
    coping_tip = coping_tip_agent.generate_tip(mood)

    # Save check-in to Firestore
    from utils.helpers import save_checkin
    from app import db
    
    if db:
        save_checkin(db, user_id, mood, lang, message, sentiment_score, intent="chat", helpful_tip=coping_tip)

    final_response = {
        "response": response_data.get("response", "I'm here for you. Can you tell me more?"),
        "mood": mood,
        "sentiment": sentiment_score,
        "crisis_detected": crisis_detected,
        "coping_tip": coping_tip
    }

    return jsonify({"ok": True, "data": final_response})
