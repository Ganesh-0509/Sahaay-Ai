# app.py
import os
import uuid
import random
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, send_from_directory
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from pushbullet import Pushbullet
import google.generativeai as genai
from google.cloud import firestore
import re
from google.cloud.firestore_v1 import _helpers
from collections import Counter
from textblob import TextBlob

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
PUSHBULLET_API_TOKEN = os.getenv("PUSHBULLET_API_TOKEN")
if not GEMINI_API_KEY or not SECRET_KEY:
    raise ValueError("GEMINI_API_KEY or SECRET_KEY is missing in .env")

# Flask setup
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
app.secret_key = SECRET_KEY

# Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Firestore init
try:
    db = firestore.Client()
    users_ref = db.collection('users')
except Exception as e:
    print("Firestore init failed:", e)
    db = None

# Gemini + Pushbullet
genai.configure(api_key=GEMINI_API_KEY)
pb = Pushbullet(PUSHBULLET_API_TOKEN) if PUSHBULLET_API_TOKEN else None

# User model
class User(UserMixin):
    def __init__(self, id, email, username, push_token=None):
        self.id = id
        self.email = email
        self.username = username
        self.push_token = push_token
    
    @staticmethod
    def get(user_id):
        doc = users_ref.document(user_id).get()
        if doc.exists:
            data = doc.to_dict()
            return User(doc.id, data['email'], data['name'], data.get('push_token'))
        return None

    @staticmethod
    def get_by_email(email):
        if not db: return None
        docs = db.collection('users').where('email', '==', email).limit(1).stream()
        for doc in docs:
            data = doc.to_dict()
            return User(doc.id, data.get('email'), data.get('name'), data.get('push_token'))
        return None

    @staticmethod
    def create(email, name, password):
        if not db: return None
        user_id = str(uuid.uuid4())
        password_hash = generate_password_hash(password)
        db.collection('users').document(user_id).set({
            'email': email,
            'name': name,
            'password_hash': password_hash,
            'created_at': firestore.SERVER_TIMESTAMP,
            'push_token': None, # Initialize push_token
            'daily_reminder': False # Initialize daily reminder
        })
        return User(user_id, email, name)

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


# --- Expanded Multilingual Translations Dictionary ---
TRANSLATIONS = {
    'en': {
        'overview': 'Overview', 'mood_journal': 'Mood Journal', 'coping_tools': 'Coping Tools',
        'analytics': 'Analytics', 'settings': 'Settings', 'dashboard_home': 'Dashboard Home',
        'mood_snapshot': 'Mood Snapshot', 'streak_tracker': 'Streak Tracker', 'quote_of_the_day': 'Quote of the Day',
        'recent_checkins': 'Recent Check-ins', 'helpful_tips': 'Helpful Tips', 'loading': 'Loading...',
        'daily_checkin_btn': 'Daily Check-in',
        'recent_entries': 'Recent Entries', 'add_new_entry': 'Add New Entry', 'add_entry_btn': 'Add Entry',
        'avg_sentiment': 'Avg Sentiment', 'most_frequent_mood': 'Most Frequent Mood', 'total_checkins': 'Total Check-ins',
        'mood_distribution': 'Mood Distribution', 'mood_trend': 'Mood Trend Over Time',
        'top_words': 'Top Words in Your Check-ins', 'top_3_tips': 'Top 3 Helpful Tips', 'insights': 'Insights',
        'account_preferences': 'Account Preferences', 'username': 'Username', 'save_changes': 'Save Changes',
        'delete_account': 'Delete Account',
        'delete_account_confirm': 'Permanently delete your account and all associated data. This action cannot be undone.',
        # Chat-related keys (already used by /translations endpoint)
        'chat_title': 'Sahaay-AI Daily Check-in', 'send_button': 'Send', 'input_placeholder': 'Type your message...',
    },
    'hi': {
        'overview': 'अवलोकन', 'mood_journal': 'मूड जर्नल', 'coping_tools': 'सहायता उपकरण',
        'analytics': 'विश्लेषण', 'settings': 'सेटिंग्स', 'dashboard_home': 'डैशबोर्ड होम',
        'mood_snapshot': 'मूड स्नैपशॉट', 'streak_tracker': 'स्ट्रीक ट्रैकर', 'quote_of_the_day': 'दिन का उद्धरण',
        'recent_checkins': 'हाल के चेक-इन', 'helpful_tips': 'उपयोगी टिप्स', 'loading': 'लोड हो रहा है...',
        'daily_checkin_btn': 'रोज़ाना चेक-इन',
        'recent_entries': 'हाल की प्रविष्टियाँ', 'add_new_entry': 'नई प्रविष्टि जोड़ें', 'add_entry_btn': 'प्रविष्टि जोड़ें',
        'avg_sentiment': 'औसत भावना', 'most_frequent_mood': 'सबसे लगातार मूड', 'total_checkins': 'कुल चेक-इन',
        'mood_distribution': 'मूड वितरण', 'mood_trend': 'समय के साथ मूड की प्रवृत्ति',
        'top_words': 'आपके चेक-इन में शीर्ष शब्द', 'top_3_tips': 'शीर्ष 3 उपयोगी टिप्स', 'insights': 'अंतर्दृष्टि',
        'account_preferences': 'खाता प्राथमिकताएं', 'username': 'उपयोगकर्ता नाम', 'save_changes': 'बदलाव सहेजें',
        'delete_account': 'खाता हटाएं',
        'delete_account_confirm': 'अपने खाते और सभी संबंधित डेटा को स्थायी रूप से हटा दें। यह कार्रवाई पूर्ववत नहीं की जा सकती है।',
        'chat_title': 'सहाय-AI दैनिक चेक-इन', 'send_button': 'भेजें', 'input_placeholder': 'अपना संदेश टाइप करें...',
    },
    # Add other languages like 'ta' (Tamil) here...
}

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
CRISIS_PROMPT = """Analyze the following user message to determine if it indicates a crisis, such as self-harm, suicidal thoughts, or severe distress. Respond with a single word: "CRISIS" if it is, or "NO_CRISIS" if it is not. User text: "{text}"
"""

model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=SYSTEM_PROMPT)
crisis_model = genai.GenerativeModel('gemini-1.5-flash')

# --- Sentiment Analysis with TextBlob ---

CRISIS_EXCLUSION_LIST = ["bye", "goodbye", "later", "cya", "ok", "okay"]

def is_crisis_sentence(text: str) -> bool:
    if text.strip().lower() in CRISIS_EXCLUSION_LIST:
        return False
    try:
        response = crisis_model.generate_content(CRISIS_PROMPT.format(text=text))
        result = response.text.strip().upper()
        if result.startswith("CRISIS") and pb:
            try:
                # Here you might want to use the current_user's push_token if available
                pb.push_note("Crisis Alert 🚨", f"A user just mentioned: '{text}'")
            except Exception as e:
                print(f"Pushbullet failed: {e}")
            return True
        return False
    except Exception as e:
        print(f"Crisis detection error: {e}")
        return False

# --- Helper functions ---
def save_checkin(user_id, mood, language, text, intent, sentiment, helpful_tip=None):
    if not db: return
    try:
        # Generate coping tip before saving
        coping_tip = generate_coping_tip(text) # Pass the original user text to the generator

        db.collection(f"users/{user_id}/checkins").add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "mood": mood,
            "language": language,
            "text": text,
            "intent": intent,
            "sentiment": sentiment,
            "coping_tip": coping_tip,
            "helpful": False
        })
    except Exception as e:
        print("Firestore save failed:", e)

def format_timestamp(ts):
    if not ts: return 'N/A'
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
    if not dates: return 0
    today = datetime.utcnow().date()
    streak = 0
    while today in dates:
        streak += 1
        today -= timedelta(days=1)
    return streak

def generate_coping_tip(mood_text):
    prompt = (f"Analyze the following text describing a mood. Provide the single dominant emotion (e.g., 'anxiety', 'sadness', 'joy'). "
              f"Then, suggest a specific, actionable coping tip for that emotion. "
              f"Use the format: Emotion: <emotion>\\nSentiment: <positive/negative/neutral>\\nTip: <tip>. Text: \"{mood_text}\"")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating coping tip: {e}")
        return "Emotion: neutral\nSentiment: neutral\nTip: Take a deep breath and a moment for yourself."

class AnonymousUser(UserMixin):
    def get_id(self):
        return session.get('anonymous_id')

    @property
    def is_authenticated(self):
        return 'anonymous_id' in session and session['anonymous_id'] is not None

def get_most_frequent_words(docs, num_words=30):
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'of', 'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'out', 'on', 'off', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'doing', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'it', 'its', 'so', 'not', 'very', 'just'])
    words_by_mood = {}
    for d in docs:
        mood = d.get('mood', 'neutral').lower()
        text = d.get('text', '')
        if not isinstance(text, str): continue
        found_words = re.findall(r'\b\w+\b', text.lower())
        if mood not in words_by_mood: words_by_mood[mood] = []
        words_by_mood[mood].extend(word for word in found_words if word not in stop_words and len(word) > 1)
    all_words = [word for words in words_by_mood.values() for word in words]
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(num_words)
    word_moods = {}
    for word, _ in most_common_words:
        mood_counts_for_word = Counter()
        for mood, words in words_by_mood.items():
            mood_counts_for_word[mood] = words.count(word)
        if mood_counts_for_word:
            word_moods[word] = mood_counts_for_word.most_common(1)[0][0]
    return most_common_words, word_moods

# --- Translation Helper ---
def get_user_translations():
    user_language = session.get('language', 'en')
    return TRANSLATIONS.get(user_language, TRANSLATIONS['en'])

# --- Routes ---
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard')) # Redirects to dashboard if user is logged in
    return render_template('index.html') # Displays landing page for new or logged-out users

@app.route('/dashboard')
@login_required
def dashboard():
    # Fetch the language dictionary using the correct function
    lang = get_user_translations() # Correct function call
    # Pass the 'lang' variable to the template
    return render_template('home.html', lang=lang)

@app.route("/chat")
@login_required
def chat_page():
    return render_template('chat.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.get_by_email(email)
        if user and check_password_hash(db.collection('users').document(user.id).get().to_dict()['password_hash'], password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard_home'))
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        if User.get_by_email(email):
            flash('Email address already exists.', 'danger')
        else:
            new_user = User.create(email, name, password)
            login_user(new_user)
            flash('Account created successfully!', 'success')
            return redirect(url_for('dashboard_home'))
    return render_template('signup.html')

@app.route('/daily_checkin_prompt')
def daily_checkin_prompt():
    lang = request.args.get('lang', 'en')
    # This is a simple placeholder.
    # You can fetch a real prompt from a database, a JSON file, or a hardcoded list.
    prompts = {
        'en': "Hello! How are you feeling today?",
        'ta': "வணக்கம்! இன்று எப்படி உணர்கிறீர்கள்?",
        'es': "¿Hola! ¿Cómo te sientes hoy?",
        'hi': "नमस्ते! आज आप कैसा महसूस कर रहे हैं?"
    }
    prompt_text = prompts.get(lang, prompts['en'])
    return jsonify({"prompt": prompt_text})

@app.route('/api/consent_status')
@login_required
def consent_status():
    try:
        user_doc = users_ref.document(current_user.id).get()
        if not user_doc.exists:
            return jsonify({"ok": False, "status": "Unknown"}), 404
        data = user_doc.to_dict()
        # Change according to your consent logic
        if data.get("consent") == True:
            status = "Consent Given"
        else:
            status = "Anonymous / Not Given"
        return jsonify({"ok": True, "status": status})
    except Exception as e:
        return jsonify({"ok": False, "status": "Error"}), 500

@app.route('/api/submit_consent', methods=['POST'])
def submit_consent():
    data = request.get_json()
    consent = data.get('consent')

    if consent is None:
        return jsonify({"ok": False, "error": "Missing consent value"}), 400

    # Save consent to session or DB
    session['has_consent'] = consent  # example using session
    status_text = "Consent Given" if consent else "Anonymous Mode (No data is saved)"
    return jsonify({"ok": True, "status": status_text})

# New route to get user info for the frontend
@app.route('/api/get_user_info', methods=['GET'])
def get_user_info():
    if current_user.is_authenticated:
        if isinstance(current_user, User):
            return jsonify({"username": current_user.username, "is_anonymous": False})
        else: # AnonymousUser instance
            return jsonify({"username": "Anonymous", "is_anonymous": True})
    return jsonify({"username": "Guest", "is_anonymous": False})

# --- Chat API ---
@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json()
    message = data.get("message")
    lang = data.get("lang", "en")

    if not message:
        return jsonify({"ok": False, "message": "No message provided."}), 400

    user_id = current_user.id if isinstance(current_user, User) else current_user.get_id()

    try:
        # Use TextBlob for sentiment analysis
        analysis = TextBlob(message)
        sentiment_score = analysis.sentiment.polarity
        
        mood = "neutral"
        if sentiment_score > 0.1:
            mood = "happy"
        elif sentiment_score < -0.1:
            mood = "sad"
        else:
            mood = "neutral"

        # Now, get the conversational response from the Gemini model
        chat_session = model.start_chat(history=[])
        chat_response = chat_session.send_message(message)
        response_text = chat_response.text if hasattr(chat_response, 'text') else str(chat_response)

        # Extract the JSON response from the model's output
        match = re.search(r'\{.*\}', response_text.strip(), re.DOTALL)
        if match:
            json_string = match.group(0)
            response_json = json.loads(json_string.replace('```json', '').replace('```', ''))
        else:
            response_json = {"response": "I'm sorry, I'm having a little trouble with that. Can you tell me more in a different way?"}

        # Handle crisis detection
        crisis = is_crisis_sentence(message)
        
        if db:
            save_checkin(
                user_id=user_id,
                mood=mood,
                language=lang,
                text=message,
                intent="casual_chat",
                sentiment=sentiment_score
            )

        final_response = {
            "mood": mood,
            "intent": "casual_chat",
            "sentiment": sentiment_score,
            "response": response_json.get("response", "I'm here to listen. Can you tell me more?")
        }
        final_response["crisis_detected"] = crisis
        return jsonify({"ok": True, "data": final_response})

    except Exception as e:
        print("Chat error:", e)
        return jsonify({"ok": False, "message": f"Error generating response: {e}"}), 500
    
# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard/mood')
@login_required
def dashboard_mood():
    return render_template('mood.html', user=current_user, lang=get_user_translations())

@app.route('/dashboard/settings')
@login_required
def dashboard_settings():
    return render_template('settings.html', user=current_user, lang=get_user_translations())

@app.route('/dashboard/analytics')
@login_required
def dashboard_analytics():
    return render_template('analytics.html', user=current_user, lang=get_user_translations())

@app.route('/dashboard/tools')
@login_required
def dashboard_tools():
    return render_template('tools.html', user=current_user, lang=get_user_translations())

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

# --- API Routes ---
@app.route("/api/home_data")
@login_required
def home_data():
    user_id = current_user.id
    period = request.args.get('period', 'last10')
    try:
        all_checkins_docs = db.collection(f"users/{user_id}/checkins").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        all_timestamps = [doc.to_dict().get("timestamp") for doc in all_checkins_docs]
        streak = calculate_streak([ts for ts in all_timestamps if ts])

        filtered_query = db.collection(f"users/{user_id}/checkins").order_by("timestamp", direction=firestore.Query.DESCENDING)
        if period == 'last7days':
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            filtered_query = filtered_query.where('timestamp', '>=', seven_days_ago)
        elif period != 'all':
            filtered_query = filtered_query.limit(10)
        
        filtered_docs = filtered_query.stream()
        recent = []
        helpful = []
        for doc in filtered_docs:
            d = doc.to_dict()
            ts = d.get("timestamp")
            recent.append({"date": format_timestamp(ts), "mood": d.get("mood", "N/A")})
            if d.get("helpful"):
                helpful.append({"date": format_timestamp(ts), "tip": d.get("coping_tip", ""), "mood": d.get("mood", "N/A")})
        
        latest_mood = recent[0]["mood"] if recent else "No data yet."
        return jsonify({"streak": streak, "recent": recent, "helpful": helpful, "quote": "Keep going, you're stronger than you think! 🌱", "mood": latest_mood})
    except Exception as e:
        print("Home data fetch failed:", e)
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
            entries.append({"mood": data.get("mood", "neutral"), "date": format_timestamp(ts)})
        return jsonify({"entries": entries})
    except Exception as e:
        print("Fetch mood entries failed:", e)
        return jsonify({"entries": []}), 500

@app.route("/api/add_mood", methods=["POST"])
@login_required
def api_add_mood():
    data = request.get_json()
    mood_text = data.get("mood")
    if not mood_text: return jsonify({"ok": False, "error": "No mood provided"}), 400
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
        if period != 'all': checkins_query = checkins_query.limit(10)
        checkins = checkins_query.stream()
        tools = []
        for doc in checkins:
            data = doc.to_dict()
            tip_text = data.get("coping_tip", "")
            tip_match = re.search(r'Tip:\s*(.*)', tip_text, re.DOTALL)
            tip_text_cleaned = tip_match.group(1).strip() if tip_match else tip_text
            tools.append({
                "tip_id": doc.id, "description": tip_text_cleaned, "mood": data.get("mood", "neutral")
            })
        return jsonify({"tools": tools})
    except Exception as e:
        print("Tools API fetch error:", e)
        return jsonify({"tools": []}), 500

@app.route("/api/analytics_data")
@login_required
def analytics_data():
    period = request.args.get("period", "last10")
    user_id = current_user.id
    ref = db.collection(f"users/{user_id}/checkins")
    if period == "last10":
        docs_stream = ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
    elif period == "30days":
        cutoff = datetime.utcnow() - timedelta(days=30)
        docs_stream = ref.where("timestamp", ">=", cutoff).order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    else:
        docs_stream = ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    docs = [d.to_dict() for d in docs_stream]
    mood_counts = Counter(d.get("mood", "neutral").lower() for d in docs)
    sentiments_by_mood_and_date = {}
    entries_by_date = {}
    for d in docs:
        mood = d.get("mood", "neutral").lower()
        sentiment = float(d.get("sentiment") or 0)
        ts = d.get("timestamp")
        if ts:
            date_str = ts.strftime("%Y-%m-%d")
            if mood not in sentiments_by_mood_and_date: sentiments_by_mood_and_date[mood] = {}
            sentiments_by_mood_and_date[mood].setdefault(date_str, []).append(sentiment)
            if date_str not in entries_by_date: entries_by_date[date_str] = []
            entries_by_date[date_str].append(d)
    trend_by_mood = {mood: [{"date": date, "sentiment": sum(vals) / len(vals)} for date, vals in sorted(dates.items())] for mood, dates in sentiments_by_mood_and_date.items()}
    most_frequent_words, word_moods = get_most_frequent_words(docs)
    top_tips = [tip for tip, _ in Counter(d.get("coping_tip") for d in docs if d.get("helpful") and d.get("coping_tip")).most_common(3)]
    total_checkins = len(docs)
    avg_sentiment = round(sum(d.get("sentiment", 0) for d in docs) / max(total_checkins, 1), 2)
    most_common_mood = mood_counts.most_common(1)[0][0].capitalize() if mood_counts else "Neutral"
    insights = f"You’ve checked in {total_checkins} times."
    if total_checkins > 0:
        insights += f" Your top mood is {most_common_mood}."
    return jsonify({
        "mood_counts": dict(mood_counts), "trend_by_mood": trend_by_mood, "most_frequent_words": most_frequent_words,
        "top_tips": top_tips, "word_moods": word_moods, "entries_by_date": entries_by_date,
        "kpis": {"total_checkins": total_checkins, "most_common_mood": most_common_mood, "avg_sentiment": avg_sentiment},
        "insights": insights,
    })

@app.route("/api/mark_helpful", methods=["POST"])
@login_required
def mark_helpful():
    data = request.get_json(force=True) or {}
    tip_id = data.get("tip_id")
    if not tip_id: return jsonify({"ok": False, "error": "No tip ID provided"}), 400
    user_id = current_user.id
    try:
        tip_ref = db.collection(f"users/{user_id}/checkins").document(tip_id)
        if not tip_ref.get().exists: return jsonify({"ok": False, "error": "Tip not found"}), 404
        tip_ref.update({"helpful": True})
        return jsonify({"ok": True, "message": "Tip marked as helpful."})
    except Exception as e:
        print("Mark helpful failed:", e)
        return jsonify({"ok": False, "error": "Failed to mark tip as helpful"}), 500

# Settings routes (still require a real login)
@app.route('/api/get_settings', methods=['GET'])
@login_required
def get_settings():
    if isinstance(current_user, AnonymousUser):
        return jsonify({"ok": False, "message": "Cannot get settings for anonymous user."}), 403
    user_id = current_user.id
    try:
        user_doc = users_ref.document(user_id).get()
        if not user_doc.exists:
            return jsonify({"ok": False, "message": "User not found."}), 404
        settings = user_doc.to_dict()
        return jsonify({
            "ok": True,
            "username": settings.get("name"),
            "dailyReminder": settings.get("daily_reminder", False)
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "message": f"Failed to get user settings: {e}"}), 500

@app.route('/api/update_settings', methods=['POST'])
@login_required
def update_settings():
    if isinstance(current_user, AnonymousUser):
        return jsonify({"ok": False, "message": "Cannot update settings for anonymous user."}), 403
    user_id = current_user.id
    data = request.get_json()
    updates = {}
    if "username" in data: updates["name"] = data["username"]
    if "dailyReminder" in data: updates["daily_reminder"] = data["dailyReminder"]
    if "pushToken" in data: updates["push_token"] = data["pushToken"]
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
    user_id = current_user.id
    try:
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        for doc in checkins_ref.stream(): doc.reference.delete()
        users_ref.document(user_id).delete()
        logout_user()
        return jsonify({"ok": True, "message": "Account deleted."}), 200
    except Exception as e:
        return jsonify({"ok": False, "message": f"Failed to delete account: {e}"}), 500

@app.route('/translations')
def get_translations():
    lang = request.args.get('lang', 'en')
    return jsonify(TRANSLATIONS.get(lang, TRANSLATIONS['en']))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)