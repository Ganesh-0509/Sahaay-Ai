# app.py (fixed to use google-genai and gemini-2.0-flash)
import os
import json
import re
from google import genai
from flask import Flask, jsonify, request, session, redirect, url_for, render_template, flash
from flask_login import LoginManager, login_required, current_user, logout_user
from collections import deque, Counter
from textblob import TextBlob
from dotenv import load_dotenv

# emotion helpers
from utils.emotion_classifier import classify_emotion, parse_final_mood

from models.user import User, AnonymousUser
from utils.helpers import format_timestamp, firestore_to_datetime, calculate_streak, get_most_frequent_words
from translations.translation_utils import get_user_translations, LANGUAGES, TRANSLATIONS
from flask_cors import CORS
from pushbullet import Pushbullet
from google.cloud import firestore
from google.oauth2 import service_account
from datetime import datetime, timezone, timedelta

# Use timezone.utc constant
UTC = timezone.utc

# Load environment variables
load_dotenv()
print("\nLoading environment variables...")

# Load and sanitize Gemini API key and secret
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip("'\"")
SECRET_KEY = os.getenv("SECRET_KEY", "").strip().strip("'\"")
PUSHBULLET_API_TOKEN = os.getenv("PUSHBULLET_API_TOKEN", "").strip().strip("'\"")

# Validate env presence (we will validate API key by listing models shortly)
if not GEMINI_API_KEY or not SECRET_KEY:
    raise ValueError("GEMINI_API_KEY or SECRET_KEY is missing in .env or invalid")

# Initialize Google GenAI client (new SDK)
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    # Validate API key by listing models (safer than calling a model that may not exist)
    try:
        models = client.models.list()
        model_names = [m.name for m in models]
        print("Available models (sample):", model_names[:10])
    except Exception as e:
        print("Warning: Unable to list models (API key may be invalid or network issue):", e)
        # Let the application continue — runtime errors will surface if API key invalid
except Exception as e:
    print("Failed to initialize genai.Client:", e)
    raise

# Choose model to use app-wide (user selected)
MODEL_NAME = "gemini-2.0-flash"  # Full model path for v1 API

# Firestore initialization (explicit service account file recommended)
try:
    cred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sahaay-ai-7fd925852862.json')
    if not os.path.exists(cred_path):
        # allow default discovery if no explicit file
        db = firestore.Client()
    else:
        creds = service_account.Credentials.from_service_account_file(
            cred_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        # prefer explicit project in service account file
        project_id = None
        try:
            with open(cred_path, 'r', encoding='utf-8') as _f:
                project_id = json.load(_f).get('project_id')
        except Exception:
            project_id = None
        if project_id:
            db = firestore.Client(project=project_id, credentials=creds)
        else:
            db = firestore.Client(credentials=creds)
    users_ref = db.collection('users')
except Exception as e:
    print("Firestore init failed:", e)
    db = None
    users_ref = None

# Flask setup
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
app.secret_key = SECRET_KEY
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_SECURE'] = True
app.config['REMEMBER_COOKIE_HTTPONLY'] = True
app.config['GEMINI_API_KEY'] = GEMINI_API_KEY
app.config['GEMINI_MODEL'] = MODEL_NAME

pb = Pushbullet(PUSHBULLET_API_TOKEN) if PUSHBULLET_API_TOKEN else None

# Simple in-memory rate limit store (per-process)
RATE_LIMIT_MAX_REQUESTS = 10  # max requests
RATE_LIMIT_WINDOW_SECONDS = 60  # per window seconds
_rate_limit_store = {}


def _rate_limited(key: str, max_requests: int = RATE_LIMIT_MAX_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS) -> bool:
    now = datetime.now(UTC).timestamp()
    dq = _rate_limit_store.get(key)
    if dq is None:
        dq = deque()
        _rate_limit_store[key] = dq
    # drop old timestamps
    cutoff = now - window_seconds
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= max_requests:
        return True
    dq.append(now)
    return False


# Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'
# Disable the default "Please log in to access this page" message
login_manager.login_message = None


@login_manager.user_loader
def load_user(user_id):
    if users_ref:
        user = User.get(user_id, users_ref)
        if user:
            return user
    return AnonymousUser()


CHAT_HISTORY = {}  # key=user_id, value=list of messages

# --- NLP and Crisis Detection ---
SYSTEM_PROMPT = """You are Sahaay-AI, a kind, happy, and supportive mental health companion for Indian youth. Your primary role is to be a welcoming and non-judgmental friend. You are here to listen, understand, and share in their feelings, both the good and the bad. Your friendly and empathetic tone should always shine through.

**INSTRUCTION: Your entire response must be a single JSON object. Do not include any text, conversation, or markdown before or after the JSON. Do not include any explanation.**

The JSON object must have one key:
- **response**: Your brief, empathetic, and conversational reply to the user.

Example:
User: I had a great day today!
Your Response:
{
  "response": "Oh, that's fantastic! Tell me all about it—what made your day so great?"
}
"""

CRISIS_PROMPT = """You are a crisis detection AI. Analyze the following message and respond with either 'CRISIS' if the message indicates a mental health crisis, or 'SAFE' if it does not. Only respond with one word: CRISIS or SAFE.

Message: {text}
Response:"""

CRISIS_EXCLUSION_LIST = ["bye", "goodbye", "later", "cya", "ok", "okay"]


def is_crisis_sentence(text: str) -> bool:
    if not text or not text.strip():
        return False
    try:
        if text.strip().lower() in CRISIS_EXCLUSION_LIST:
            return False
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=CRISIS_PROMPT.format(text=text)
        )
        result = getattr(resp, "text", "").strip().upper()
        token = getattr(current_user, "push_token", None)
        if result.startswith("CRISIS") and pb and token:
            try:
                pb.push_note("Crisis Alert 🚨", f"A user just mentioned: '{text}'")
            except Exception:
                pass
        return result.startswith("CRISIS")
    except Exception as e:
        print(f"Crisis detection error: {e}")
        return False


def process_and_store_message(user_id, message_text, doc_ref):
    """
    Processes a user message by sending it to Gemini for emotion analysis.
    """
    try:
        prompt = f"""Analyze the following text and identify the primary emotions present. If more than one emotion is present, list them. The emotions should be from the following list: happy, sad, angry, anxious, calm, excited, confused, neutral.

Example 1:
Text: "I had a great day today, but I'm a little tired."
Response: happy, tired

Example 2:
Text: "I'm so frustrated with my work, and now I have a huge deadline."
Response: angry, anxious

Example 3:
Text: "The movie was so boring and I just wanted to leave."
Response: sad

Example 4:
Text: "{message_text}"
Response:"""

        # Use client.models.generate_content (no REST)
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        response_text = getattr(resp, "text", "").strip().lower()
        detected_emotions = [e.strip() for e in re.split(r'[,\n]+', response_text) if e.strip()]
        if not detected_emotions:
            final_emotions = ["neutral"]
        else:
            final_emotions = detected_emotions

        message_data = {
            "text": message_text,
            "timestamp": datetime.now(timezone.utc),
            "emotions": final_emotions
        }
        # store into Firestore doc_ref
        try:
            doc_ref.update({"messages": firestore.ArrayUnion([message_data])})
        except Exception:
            # fallback: if doc_ref.update fails, try adding a new doc or ignore
            pass
        return {"emotions": final_emotions, "message": message_text}
    except Exception as e:
        print(f"Emotion analysis API call failed: {e}")
        return {"emotions": ["unknown"], "message": message_text}


# --- Helper functions ---
def save_checkin(user_id, mood, language, text, intent, sentiment, helpful_tip=None):
    """
    Saves a checkin with proper mood structure:

    - mood_list: ["joy", "sadness"]
    - mood_label: "Joy / Sadness"
    - mood_dominant: "joy"
    """
    if not db:
        return

    try:
        # Convert raw mood (like "joy/sadness" or "mixed:joy(...)/sadness(...)") into lists + clean label
        mood_list, mood_label = parse_final_mood(mood)
        dominant = mood_list[0] if mood_list else "neutral"

        today = datetime.now(UTC).date()
        today_str = today.strftime("%Y-%m-%d")
        checkins_ref = db.collection(f"users/{user_id}/checkins")

        docs = list(checkins_ref.where("date", "==", today_str).stream())

        if docs:
            doc = docs[0]
            data = doc.to_dict() or {}

            sentiments = data.get("sentiments", [])
            sentiments.append(sentiment)
            avg_sentiment = sum(sentiments) / len(sentiments)

            old_moods = data.get("mood_list", [])
            new_moods = old_moods + mood_list

            doc.reference.update({
                "sentiments": sentiments,
                "avg_sentiment": avg_sentiment,
                "mood_list": new_moods,
                "mood_label": mood_label,
                "mood_dominant": dominant,
                "language": language,
                "last_text": text,
                "intent": intent,
                "coping_tip": helpful_tip or generate_coping_tip(text),
                "timestamp": firestore.SERVER_TIMESTAMP
            })

        else:
            checkins_ref.add({
                "date": today_str,
                "sentiments": [sentiment],
                "avg_sentiment": sentiment,
                "mood_list": mood_list,
                "mood_label": mood_label,
                "mood_dominant": dominant,
                "language": language,
                "last_text": text,
                "intent": intent,
                "coping_tip": helpful_tip or generate_coping_tip(text),
                "helpful": False,
                "timestamp": firestore.SERVER_TIMESTAMP
            })

    except Exception as e:
        print("Firestore save failed:", e)


def format_timestamp(ts):
    if not ts:
        return 'N/A'
    try:
        return ts.strftime("%b %d, %Y")
    except Exception:
        return 'N/A'


def firestore_to_datetime(ts):
    if ts and hasattr(ts, 'astimezone'):
        return ts
    return None


def calculate_streak(timestamps):
    dates = {firestore_to_datetime(ts).date() for ts in timestamps if firestore_to_datetime(ts)}
    if not dates:
        return 0
    today = datetime.now(UTC).date()
    streak = 0
    while today in dates:
        streak += 1
        today -= timedelta(days=1)
    return streak


def generate_coping_tip(mood_text):
    prompt = f"Provide a single, specific, and actionable coping tip for someone feeling {mood_text}. Respond with only the tip text, no other information."
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        tip_text = getattr(resp, "text", "").strip()
        return tip_text or "Take a deep breath and a moment for yourself."
    except Exception as e:
        print(f"Error generating coping tip: {e}")
        return "Take a deep breath and a moment for yourself."


# --- Routes (kept your original structure) ---
@app.route("/api/fetch_conversation", methods=["GET"])
@login_required
def fetch_conversation():
    user_id = current_user.id
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({"ok": False, "error": "Date parameter is missing"}), 400
    try:
        doc_ref = db.collection(f"users/{user_id}/conversations").document(date_str)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            return jsonify({"ok": True, "messages": data.get("messages", [])})
        else:
            return jsonify({"ok": True, "messages": []})
    except Exception as e:
        print(f"Error fetching conversation: {e}")
        return jsonify({"ok": False, "error": "Failed to fetch conversation history"}), 500


@app.route('/daily_checkin_prompt')
def daily_checkin_prompt():
    lang = request.args.get('lang', 'en')
    prompts = {
        'en': "Hello! How are you feeling today?",
        'ta': "வணக்கம்! இன்று எப்படி உணர்கிறீர்கள்?",
        'es': "¿Hola! ¿Cómo te sientes hoy?",
        'hi': "नमस्ते! आज आप कैसा महसूस कर रहे हैं?"
    }
    prompt_text = prompts.get(lang, prompts['en'])
    return jsonify({"prompt": prompt_text})


@app.route("/api/set_consent", methods=["POST"])
def set_consent():
    try:
        if current_user.is_authenticated and db:
            user_id = current_user.id
            db.collection("users").document(user_id).set({"has_consent": True}, merge=True)
        else:
            session["has_consent"] = True
        return jsonify({"ok": True})
    except Exception as e:
        print("Set consent error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/submit_consent', methods=['POST'])
def submit_consent():
    data = request.get_json()
    consent = data.get('consent')
    if consent is None:
        return jsonify({"ok": False, "error": "Missing consent value"}), 400
    try:
        if current_user.is_authenticated and db is not None:
            db.collection("users").document(current_user.id).set(
                {"has_consent": bool(consent)},
                merge=True
            )
        else:
            session['has_consent'] = bool(consent)
        status_text = "Consent Given" if consent else "Anonymous Mode (No data is saved)"
        return jsonify({"ok": True, "status": status_text})
    except Exception as e:
        print("Set consent error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/consent_status', methods=['GET'])
def consent_status():
    try:
        if current_user.is_authenticated and db is not None:
            doc = db.collection("users").document(current_user.id).get()
            if doc.exists:
                return jsonify({"has_consent": doc.to_dict().get("has_consent", False)})
            return jsonify({"has_consent": False})
        return jsonify({"has_consent": session.get("has_consent", False)})
    except Exception as e:
        print("Consent status error:", e)
        return jsonify({"has_consent": False, "error": str(e)}), 500


@app.route('/api/get_user_info', methods=['GET'])
def get_user_info():
    if current_user.is_authenticated:
        if isinstance(current_user, User):
            return jsonify({
                "user_id": current_user.id,
                "username": current_user.username,
                "is_anonymous": False
            })
        else:
            return jsonify({
                "user_id": session.get('anonymous_id', 'anon'),
                "username": "Anonymous",
                "is_anonymous": True
            })
    return jsonify({"user_id": "guest", "username": "Guest", "is_anonymous": False})


@app.route('/api/reset_consent', methods=['POST'])
def reset_consent():
    try:
        if current_user.is_authenticated and db:
            db.collection("users").document(current_user.id).update(
                {"has_consent": firestore.DELETE_FIELD}
            )
        else:
            session.pop("has_consent", None)
        return jsonify({"ok": True, "has_consent": False})
    except Exception as e:
        print("Reset consent error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/home_data', methods=['GET'])
@login_required
def home_data():
    if not db:
        return jsonify({"error": "Database not initialized"}), 500
    user_id = current_user.id
    period = request.args.get('period', 'last10')
    
    from translations.translation_utils import translate_mood
    user_lang = session.get('language', 'en')
    
    try:
        all_checkins_docs = db.collection(f"users/{user_id}/checkins").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        all_timestamps = [doc.to_dict().get("timestamp") for doc in all_checkins_docs]
        streak = calculate_streak([ts for ts in all_timestamps if ts])

        filtered_query = db.collection(f"users/{user_id}/checkins").order_by("timestamp", direction=firestore.Query.DESCENDING)
        if period == 'last7days':
            seven_days_ago = datetime.now(UTC) - timedelta(days=7)
            filtered_query = filtered_query.where('timestamp', '>=', seven_days_ago)
        elif period != 'all':
            filtered_query = filtered_query.limit(10)

        filtered_docs = filtered_query.stream()
        recent = []
        helpful = []
        for doc in filtered_docs:
            d = doc.to_dict()
            ts = d.get("timestamp")
            mood_raw = d.get("mood_label", d.get("mood_dominant", "N/A"))
            mood_translated = translate_mood(mood_raw, user_lang)
            recent.append({"date": format_timestamp(ts), "mood": mood_translated})
            if d.get("helpful"):
                helpful.append({"date": format_timestamp(ts), "tip": d.get("coping_tip", ""), "mood": mood_translated})

        latest_mood = recent[0]["mood"] if recent else "No data yet."
        result = {"streak": streak, "recent": recent, "helpful": helpful, "quote": "Keep going, you're stronger than you think! 🌱", "mood": latest_mood}
        print(f"Home data for user {user_id}: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"Home data fetch failed for user {user_id}:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to fetch home data"}), 500


@app.route("/api/mood_data", methods=["GET"])
@login_required
def api_mood_data():
    user_id = current_user.id
    entries = []
    try:
        checkins = db.collection(f"users/{user_id}/checkins").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
        for doc in checkins:
            data = doc.to_dict()
            ts = data.get("timestamp")
            entries.append({"mood": data.get("mood_label", data.get("mood_dominant", "neutral")), "date": format_timestamp(ts)})
        return jsonify({"entries": entries})
    except Exception as e:
        print("Fetch mood entries failed:", e)
        return jsonify({"entries": []}), 500


@app.route("/api/add_mood", methods=["POST"])
@login_required
def api_add_mood():
    data = request.get_json()
    mood_text = data.get("mood")
    if not mood_text:
        return jsonify({"ok": False, "error": "No mood provided"}), 400
    user_id = current_user.id
    try:
        db.collection(f"users/{user_id}/checkins").add({
            "mood": mood_text, "text": mood_text, "intent": "manual_entry", "sentiment": 0.0, "timestamp": firestore.SERVER_TIMESTAMP
        })
        return jsonify({"ok": True})
    except Exception as e:
        print("Add mood entry failed:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/tools_data", methods=["GET"])
@login_required
def api_tools_data():
    user_id = current_user.id
    period = request.args.get('period', 'last10')
    try:
        checkins_query = db.collection(f"users/{user_id}/checkins").order_by("timestamp", direction=firestore.Query.DESCENDING)
        if period != 'all':
            checkins_query = checkins_query.limit(10)
        checkins = checkins_query.stream()

        tools = []
        for doc in checkins:
            data = doc.to_dict()
            tip_text = data.get("coping_tip", "").strip()
            tools.append({
                "tip_id": doc.id,
                "description": tip_text,
                "mood": data.get("mood_label", data.get("mood_dominant", "neutral"))
            })
        return jsonify({"tools": tools})
    except Exception as e:
        print("Tools API fetch error:", e)
        return jsonify({"tools": []}), 500


@app.route('/erase', methods=['POST'])
@login_required
def erase_data():
    user_id = current_user.id
    try:
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        deleted = 0
        for doc in checkins_ref.stream():
            doc.reference.delete()
            deleted += 1
        return jsonify({"ok": True, "deleted": deleted})
    except Exception as e:
        print("Erase failed:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/analytics_data")
@login_required
def analytics_data():
    period = request.args.get("period", "last10")
    user_id = current_user.id
    ref = db.collection(f"users/{user_id}/checkins")
    if period == "last10":
        docs_stream = ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
    elif period == "30days":
        cutoff = datetime.now(UTC) - timedelta(days=30)
        docs_stream = ref.where("timestamp", ">=", cutoff).order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    else:
        docs_stream = ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    docs = [d.to_dict() for d in docs_stream]

    # Map each checkin to a single dominant mood (avoid mixed labels in analytics)
    mood_counts = Counter()
    sentiments_by_mood_and_date = {}
    entries_by_date = {}

    def _normalize_mood_name(m: str) -> str:
        if not m:
            return 'neutral'
        m = m.strip().lower()
        mapping = {
            'joy': 'happy',
            'sadness': 'sad',
            'anger': 'angry',
            'fear': 'anxious',
            'mixed': 'normal'
        }
        return mapping.get(m, m)

    for d in docs:
        # collect moods for this entry (split mixed labels into components)
        moods = []
        ml = d.get('mood_list')
        if isinstance(ml, (list, tuple)) and ml:
            moods = [str(x).strip() for x in ml if x]
        else:
            # try mood_label or mood_dominant
            label = d.get('mood_label') or d.get('mood_dominant')
            if isinstance(label, str) and label:
                lab = label.strip()
                # handle patterns like 'mixed:joy(60)/sadness(40)'
                if lab.lower().startswith('mixed:'):
                    parts = lab.split(':', 1)[1]
                    parts = parts.split('/')
                    moods = [re.sub(r"\(.*?\)", '', p).strip() for p in parts if p.strip()]
                elif lab.lower().strip() == 'mixed':
                    # plain 'mixed' label - treat as normal/healthy bucket
                    moods = ['normal']
                elif '/' in lab:
                    parts = lab.split('/')
                    moods = [p.strip() for p in parts if p.strip()]
                else:
                    moods = [lab]

        # normalize and fallback
        moods = [_normalize_mood_name(m.lower()) for m in moods if m]
        if not moods:
            moods = ['neutral']

        avg_sentiment = float(d.get('avg_sentiment') or 0)
        ts = d.get('timestamp')
        # prefer 'date' string if present, else fallback to timestamp
        date_str = d.get('date')
        if not date_str:
            try:
                if ts and hasattr(ts, 'strftime'):
                    date_str = ts.strftime('%Y-%m-%d')
                elif isinstance(ts, datetime):
                    date_str = ts.strftime('%Y-%m-%d')
                else:
                    date_str = 'N/A'
            except Exception:
                date_str = 'N/A'

        # count each mood component (mixed entries contribute to their parts)
        for mood in moods:
            mood_counts[mood] += 1
            sentiments_by_mood_and_date.setdefault(mood, {}).setdefault(date_str, []).append(avg_sentiment)

        # normalize entry for display (use first mood as label)
        entry_copy = dict(d)
        entry_copy['mood_label'] = moods[0].capitalize() if moods else 'Neutral'
        entry_copy['mood_dominant'] = moods[0] if moods else 'neutral'
        entries_by_date.setdefault(date_str, []).append(entry_copy)

    # Build trend only for moods we will show (we'll trim to top 3 below)
    full_trend_by_mood = {
        mood: [
            {"date": date, "sentiment": sum(vals) / len(vals)}
            for date, vals in sorted(dates.items())
        ]
        for mood, dates in sentiments_by_mood_and_date.items()
    }

    # Keep only top 3 moods by count for analytics display
    top_moods = [m for m, _ in mood_counts.most_common(3)]
    # Group all other moods into a 'normal' (Normal/Healthy) bucket
    other_moods = [m for m in mood_counts.keys() if m not in top_moods]
    other_count = sum(mood_counts[m] for m in other_moods)
    filtered_mood_counts = {m: mood_counts[m] for m in top_moods}
    if other_count > 0:
        filtered_mood_counts['normal'] = other_count

    # Build trend for top moods and aggregate others into 'normal'
    trend_by_mood = {m: full_trend_by_mood.get(m, []) for m in top_moods}
    # aggregate other moods per date into normal trend
    normal_dates = {}
    for m in other_moods:
        for date, vals in sentiments_by_mood_and_date.get(m, {}).items():
            normal_dates.setdefault(date, []).extend(vals)
    if normal_dates:
        trend_by_mood['normal'] = [
            {"date": date, "sentiment": (sum(vals) / len(vals)) if vals else 0}
            for date, vals in sorted(normal_dates.items())
        ]

    most_frequent_words, word_moods = get_most_frequent_words(docs)
    top_tips = [
        tip for tip, _ in Counter(
            d.get("coping_tip") for d in docs if d.get("helpful") and d.get("coping_tip")
        ).most_common(3)
    ]
    total_checkins = len(docs)
    avg_sentiment = round(
        sum(d.get("avg_sentiment", 0) for d in docs) / max(total_checkins, 1), 2
    )
    # choose most common mood from filtered counts (includes 'normal' bucket)
    if filtered_mood_counts:
        most_common_key = max(filtered_mood_counts.items(), key=lambda x: x[1])[0]
    else:
        most_common_key = None
    most_common_mood = most_common_key if most_common_key else (mood_counts.most_common(1)[0][0] if mood_counts else "neutral")
    insights = f"You’ve checked in {total_checkins} times."
    if total_checkins > 0:
        insights += f" Your top mood is {most_common_mood}."
    return jsonify({
        "mood_counts": filtered_mood_counts,
        "trend_by_mood": trend_by_mood,
        "most_frequent_words": most_frequent_words,
        "top_tips": top_tips,
        "word_moods": word_moods,
        "entries_by_date": entries_by_date,
        "kpis": {
            "total_checkins": total_checkins,
            "most_common_mood": most_common_mood,
            "avg_sentiment": avg_sentiment
        },
        "insights": insights,
    })


@app.route("/api/previous_day_summary", methods=["GET"])
@login_required
def previous_day_summary():
    user_id = current_user.id
    # Get yesterday's date in UTC
    today = datetime.now(UTC).date()
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    
    try:
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        # Query by date string for consistency with our data model
        docs = checkins_ref.where("date", "==", yesterday_str).stream()
        checkins = [doc.to_dict() for doc in docs]
        
        if not checkins:
            return jsonify({"ok": True, "summary": "No check-ins for the previous day."})
        
        # Concatenate all text fields for summarization
        all_text = "\n".join([c.get("last_text", c.get("text", "")) for c in checkins if c.get("last_text") or c.get("text")])
        if not all_text.strip():
            return jsonify({"ok": True, "summary": "No text data to summarize for the previous day."})
        
        # Use Gemini for summary
        summary = None
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"Summarize the following daily check-in notes in 2-3 sentences:\n{all_text}"
            )
            summary = getattr(response, "text", "").strip()
        except Exception as e:
            print("Gemini summarization failed:", e)
        
        if not summary:
            # Fallback: simple TextBlob summary (first 2 sentences)
            blob = TextBlob(all_text)
            summary = " ".join(str(s) for s in blob.sentences[:2])
        
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        print("Previous day summary error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/mood_insights", methods=["GET"])
@login_required
def mood_insights():
    user_id = current_user.id
    days = int(request.args.get('days', 30))
    
    try:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        docs = checkins_ref.where("timestamp", ">=", cutoff).order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        checkins = [doc.to_dict() for doc in docs]
        
        if not checkins:
            return jsonify({"ok": False, "message": "No check-ins found"})
        
        # Calculate best and worst day
        best_day = None
        worst_day = None
        best_sentiment = -2
        worst_sentiment = 2
        
        for c in checkins:
            sentiment = c.get('avg_sentiment', 0)
            date_str = c.get('date', 'N/A')
            
            if sentiment > best_sentiment:
                best_sentiment = sentiment
                best_day = {"date": date_str, "sentiment": round(sentiment, 2)}
            
            if sentiment < worst_sentiment:
                worst_sentiment = sentiment
                worst_day = {"date": date_str, "sentiment": round(sentiment, 2)}
        
        return jsonify({
            "ok": True,
            "best_day": best_day,
            "worst_day": worst_day
        })
    except Exception as e:
        print("Mood insights error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/mood_weekly_summary", methods=["GET"])
@login_required
def mood_weekly_summary():
    user_id = current_user.id
    
    try:
        # Get last 7 days of check-ins
        cutoff = datetime.now(UTC) - timedelta(days=7)
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        docs = checkins_ref.where("timestamp", ">=", cutoff).order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        checkins = [doc.to_dict() for doc in docs]
        
        if not checkins:
            return jsonify({"ok": True, "summary": "No check-ins for the past week."})
        
        # Concatenate all text for summarization
        all_text = "\n".join([c.get("last_text", c.get("text", "")) for c in checkins if c.get("last_text") or c.get("text")])
        if not all_text.strip():
            return jsonify({"ok": True, "summary": "No text data to summarize for the week."})
        
        # Use Gemini for summary
        summary = None
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=f"Summarize the following weekly check-in notes in 3-4 sentences, highlighting emotional patterns:\n{all_text}"
            )
            summary = getattr(response, "text", "").strip()
        except Exception as e:
            print("Gemini weekly summarization failed:", e)
        
        if not summary:
            # Fallback: simple summary
            blob = TextBlob(all_text)
            summary = " ".join(str(s) for s in blob.sentences[:3])
        
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        print("Weekly summary error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/mood_themes", methods=["GET"])
@login_required
def mood_themes():
    user_id = current_user.id
    days = int(request.args.get('days', 30))
    ai_mode = request.args.get('ai', 'false').lower() == 'true'
    
    try:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        docs = checkins_ref.where("timestamp", ">=", cutoff).order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        checkins = [doc.to_dict() for doc in docs]
        
        if not checkins:
            return jsonify({"ok": False, "message": "No check-ins found"})
        
        if ai_mode:
            # Use Gemini to generate AI themes
            all_text = "\n".join([c.get("last_text", c.get("text", "")) for c in checkins if c.get("last_text") or c.get("text")])
            if not all_text.strip():
                return jsonify({"ok": True, "themes_ai": ["No text data available"]})
            
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=f"Analyze these mental health check-in notes and identify 5-6 recurring themes or patterns. Return only a simple list of themes:\n{all_text}"
                )
                themes_text = getattr(response, "text", "").strip()
                # Parse the response into a list
                themes_ai = [line.strip().lstrip('•-*123456789. ') for line in themes_text.split('\n') if line.strip()]
                return jsonify({"ok": True, "themes_ai": themes_ai[:6]})
            except Exception as e:
                print("AI themes generation failed:", e)
                return jsonify({"ok": False, "error": "AI themes unavailable"})
        else:
            # Extract keyword themes from text
            from collections import Counter
            import re
            
            all_words = []
            mood_by_word = {}
            
            for c in checkins:
                text = c.get("last_text", c.get("text", ""))
                mood = c.get("mood_dominant", "neutral")
                if not text:
                    continue
                
                words = re.findall(r'\b\w{4,}\b', text.lower())
                for word in words:
                    all_words.append(word)
                    if word not in mood_by_word:
                        mood_by_word[word] = []
                    mood_by_word[word].append(mood)
            
            # Get most common words
            word_counts = Counter(all_words)
            # Filter out common stop words
            stop_words = {'that', 'this', 'with', 'have', 'been', 'were', 'will', 'your', 'from', 'they', 'would', 'there', 'their', 'what', 'about', 'which', 'when', 'make', 'like', 'time', 'just', 'know', 'take', 'people', 'into', 'year', 'good', 'some', 'could', 'them', 'than', 'other', 'then', 'more', 'these', 'want', 'many', 'must'}
            
            themes = []
            for word, count in word_counts.most_common(20):
                if word not in stop_words and count > 1:
                    # Find most common mood associated with this word
                    mood_list = mood_by_word.get(word, [])
                    if mood_list:
                        most_common_mood = Counter(mood_list).most_common(1)[0][0]
                        themes.append({"keyword": word, "count": count, "associated_mood": most_common_mood})
            
            return jsonify({"ok": True, "themes": themes[:10]})
    except Exception as e:
        print("Mood themes error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/mark_helpful", methods=["POST"])
@login_required
def mark_helpful():
    data = request.get_json(force=True) or {}
    tip_id = data.get("tip_id")
    if not tip_id:
        return jsonify({"ok": False, "error": "No tip ID provided"}), 400
    user_id = current_user.id
    try:
        tip_ref = db.collection(f"users/{user_id}/checkins").document(tip_id)
        if not tip_ref.get().exists:
            return jsonify({"ok": False, "error": "Tip not found"}), 404
        tip_ref.update({"helpful": True})
        return jsonify({"ok": True, "message": "Tip marked as helpful."})
    except Exception as e:
        print("Mark helpful failed:", e)
        return jsonify({"ok": False, "error": "Failed to mark tip as helpful"}), 500


@app.route('/api/get_settings', methods=['GET'])
@login_required
def get_settings():
    if isinstance(current_user, AnonymousUser):
        return jsonify({"ok": False, "message": "Cannot get settings for anonymous user."}), 403
    if not users_ref:
        return jsonify({"ok": False, "message": "Firestore not configured."}), 500
    user_id = current_user.id
    try:
        user_doc = users_ref.document(user_id).get()
        if not user_doc.exists:
            return jsonify({"ok": False, "message": "User not found."}), 404
        settings = user_doc.to_dict()
        return jsonify({
            "ok": True,
            "username": settings.get("name"),
            "language": settings.get("language", "en"),
            "dailyReminder": settings.get("daily_reminder", False),
            "push_token": settings.get("push_token", "")
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "message": f"Failed to get user settings: {e}"}), 500


@app.route('/api/update_settings', methods=['POST'])
@login_required
def update_settings():
    if isinstance(current_user, AnonymousUser):
        return jsonify({"ok": False, "message": "Cannot update settings for anonymous user."}), 403
    if not users_ref:
        return jsonify({"ok": False, "message": "Firestore not configured."}), 500
    user_id = current_user.id
    data = request.get_json()
    updates = {}
    if "username" in data:
        updates["name"] = data["username"]
    if "language" in data:
        updates["language"] = data["language"]
        # Also update session immediately
        session['language'] = data["language"]
    if "dailyReminder" in data:
        updates["daily_reminder"] = data["dailyReminder"]
    if "pushToken" in data:
        updates["push_token"] = data["pushToken"]
    try:
        users_ref.document(user_id).update(updates)
        return jsonify({"ok": True, "message": "Settings updated successfully."}), 200
    except Exception as e:
        return jsonify({"ok": False, "message": f"Failed to update settings: {e}"}), 500


@app.route('/api/delete_account', methods=['POST'])
@login_required
def delete_account():
    if isinstance(current_user, AnonymousUser):
        return jsonify({"ok": False, "message": "Cannot delete account for anonymous user."}), 403
    if not db or not users_ref:
        return jsonify({"ok": False, "message": "Firestore not configured."}), 500
    user_id = current_user.id
    try:
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        for doc in checkins_ref.stream():
            doc.reference.delete()
        users_ref.document(user_id).delete()
        logout_user()
        return jsonify({"ok": True, "message": "Account deleted."}), 200
    except Exception as e:
        return jsonify({"ok": False, "message": f"Failed to delete account: {e}"}), 500


@app.route('/translations')
def get_translations():
    lang = request.args.get('lang', 'en')
    return jsonify(TRANSLATIONS.get(lang, TRANSLATIONS['en']))


@app.route('/account/audit-log')
@login_required
def audit_log():
    user_id = current_user.id
    logs = []
    try:
        docs = db.collection(f"users/{user_id}/audit_logs").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        for doc in docs:
            logs.append(doc.to_dict())
    except Exception as e:
        print("Audit log fetch failed:", e)
    return render_template('audit_log.html', logs=logs)


@app.route('/account/export-data')
@login_required
def export_data():
    user_id = current_user.id
    try:
        user_doc = db.collection('users').document(user_id).get()
        checkins = db.collection(f"users/{user_id}/checkins").stream()
        conversations = db.collection(f"users/{user_id}/conversations").stream()
        data = {
            'user': user_doc.to_dict(),
            'checkins': [doc.to_dict() for doc in checkins],
            'conversations': [doc.to_dict() for doc in conversations]
        }
        response = app.response_class(
            response=json.dumps(data, default=str),
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment;filename=account_data.json'}
        )
        return response
    except Exception as e:
        print("Data export failed:", e)
        flash('Failed to export data.', 'danger')


# Register blueprints (keep as-is — ensures modular routes)
from routes.auth_routes import auth_bp
from routes.chat_routes import chat_bp
from routes.dashboard_routes import dash_bp
from routes.misc_routes import misc_bp
from routes.api_routes import api_bp
app.register_blueprint(auth_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(dash_bp)
app.register_blueprint(misc_bp)
app.register_blueprint(api_bp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
