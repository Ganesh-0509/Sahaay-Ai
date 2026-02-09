from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from utils.helpers import format_timestamp, calculate_streak, get_most_frequent_words
from datetime import datetime, timezone, timedelta
import re
from collections import Counter

api_bp = Blueprint('api', __name__)

@api_bp.route('/api/home_data')
@login_required
def home_data():
    """Fetch home dashboard data for the current user"""
    user_id = current_user.id
    
    # DEBUG: Print user info
    print(f"🔍 DEBUG /api/home_data:")
    print(f"   User ID: {user_id}")
    print(f"   Is Authenticated: {current_user.is_authenticated}")
    print(f"   User Object: {current_user}")
    
    period = request.args.get('period', 'last10')
    
    from app import db
    if not db:
        return jsonify({
            "ok": False,
            "mood": "No data",
            "streak": 0,
            "quote": "Start your mental health journey today!",
            "recent": [],
            "helpful": []
        })
    
    from translations.translation_utils import translate_mood
    from flask import session
    user_lang = session.get('language', 'en')
    
    try:
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        
        # Determine query based on period
        if period == 'last10':
            docs_stream = checkins_ref.order_by("date", direction="DESCENDING").limit(10).stream()
        elif period == 'last7days':
            seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
            docs_stream = checkins_ref.where("date", ">=", seven_days_ago).order_by("date", direction="DESCENDING").stream()
        else:  # all
            docs_stream = checkins_ref.order_by("date", direction="DESCENDING").limit(100).stream()
        
        docs = list(docs_stream)
        
        # Process data
        recent_checkins = []
        helpful_tips = []
        mood_counts = {}
        
        for doc in docs:
            data = doc.to_dict()
            if data:
                date_raw = data.get("date", "")
                # Format date for display
                try:
                    date_obj = datetime.strptime(date_raw, "%Y-%m-%d")
                    date_display = date_obj.strftime("%b %d, %Y")
                except:
                    date_display = date_raw

                mood_raw = data.get("mood_dominant", data.get("mood_label", "neutral"))
                mood_translated = translate_mood(mood_raw, user_lang)
                coping_tip = data.get("coping_tip", "")
                is_helpful = data.get("helpful", False)
                
                # Count moods
                mood_counts[mood_translated] = mood_counts.get(mood_translated, 0) + 1
                
                # Add to recent checkins
                recent_checkins.append({
                    "date": date_display,
                    "mood": mood_translated
                })
                
                # Add to helpful tips if marked as helpful
                if is_helpful and coping_tip:
                    helpful_tips.append({
                        "date": date_display,
                        "mood": mood_translated,
                        "tip": coping_tip
                    })
        
        # Fetch all checkins for streak calculation (more accurate)
        all_docs = list(checkins_ref.order_by("date", direction="DESCENDING").limit(100).stream())
        streak = calculate_streak(all_docs)
        
        # Calculate most common mood
        most_common_mood = max(mood_counts.items(), key=lambda x: x[1])[0] if mood_counts else translate_mood("No data yet.", user_lang)
        
        # Get motivational quote
        quotes = [
            "Every day is a fresh start. 🌅",
            "You are stronger than you think. 💪",
            "Progress, not perfection. 🌱",
            "Your mental health matters. 💚",
            "One step at a time. 🚶",
            "You've got this! ⭐",
            "Be kind to yourself. 🤗"
        ]
        import random
        quote = random.choice(quotes)
        
        return jsonify({
            "ok": True,
            "mood": most_common_mood,
            "streak": streak,
            "quote": quote,
            "recent": recent_checkins[:10],
            "helpful": helpful_tips[:5]
        })
    
    except Exception as e:
        print(f"❌ Error fetching home data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "mood": "Error loading data",
            "streak": 0,
            "quote": "Keep going, you're doing great!",
            "recent": [],
            "helpful": []
        })

@api_bp.route('/api/mood_data', methods=['GET'])
@login_required
def api_mood_data():
    """Fetch mood/check-in data for the current user"""
    user_id = current_user.id
    
    from app import db
    if not db:
        return jsonify({"entries": []})
    
    try:
        # Get all check-ins for the user, ordered by date field (more reliable than timestamp)
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        
        # First try ordering by timestamp, if that fails, try date
        try:
            docs = list(checkins_ref.order_by("timestamp", direction="DESCENDING").limit(100).stream())
        except Exception as e:
            print(f"Warning: Could not order by timestamp, trying date field: {e}")
            docs = list(checkins_ref.order_by("date", direction="DESCENDING").limit(100).stream())
        
        entries = []
        for doc in docs:
            data = doc.to_dict()
            if data:
                # Handle Firestore timestamp properly
                timestamp = data.get("timestamp")
                timestamp_str = ""
                
                if timestamp:
                    # Check if it's a Firestore timestamp object
                    if hasattr(timestamp, 'timestamp'):
                        # Convert Firestore timestamp to Python datetime
                        dt = datetime.fromtimestamp(timestamp.timestamp(), tz=timezone.utc)
                        timestamp_str = dt.isoformat()
                    elif isinstance(timestamp, datetime):
                        timestamp_str = timestamp.isoformat()
                    else:
                        timestamp_str = str(timestamp)
                
                # If no timestamp, use date field
                if not timestamp_str and data.get("date"):
                    try:
                        date_obj = datetime.strptime(data.get("date"), "%Y-%m-%d")
                        timestamp_str = date_obj.isoformat()
                    except:
                        timestamp_str = data.get("date", "")
                
                entries.append({
                    "id": doc.id,
                    "date": data.get("date", ""),
                    "timestamp": timestamp_str,
                    "mood": data.get("mood_dominant", data.get("mood_label", "neutral")),
                    "mood_label": data.get("mood_label", ""),
                    "mood_list": data.get("mood_list", []),
                    "sentiment": data.get("avg_sentiment", 0),
                    "message": data.get("last_text", ""),
                    "coping_tip": data.get("coping_tip", ""),
                    "helpful": data.get("helpful", False),
                    "summary": data.get("last_text", "")[:100]  # Add summary field
                })
        
        print(f"✓ Fetched {len(entries)} mood entries for user {user_id}")
        return jsonify({"entries": entries})
    
    except Exception as e:
        print(f"❌ Error fetching mood data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"entries": []})

@api_bp.route('/api/add_mood', methods=['POST'])
@login_required
def api_add_mood():
    """Manually add a mood entry"""
    data = request.get_json() or {}
    mood_text = data.get("mood")
    
    if not mood_text:
        return jsonify({"ok": False, "error": "No mood provided"}), 400
        
    user_id = current_user.id
    from app import db
    if not db:
        return jsonify({"ok": False, "error": "Database not initialized"}), 500
        
    try:
        from google.cloud import firestore
        from utils.emotion_classifier import parse_final_mood
        
        mood_list, mood_label = parse_final_mood(mood_text)
        dominant = mood_list[0] if mood_list else "neutral"
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        db.collection(f"users/{user_id}/checkins").add({
            "date": today_str,
            "mood_dominant": dominant,
            "mood_label": mood_label,
            "mood_list": mood_list,
            "last_text": mood_text,
            "intent": "manual_entry",
            "avg_sentiment": 0.0,
            "sentiments": [0.0],
            "timestamp": firestore.SERVER_TIMESTAMP,
            "helpful": False
        })
        return jsonify({"ok": True})
    except Exception as e:
        print(f"❌ Add mood entry failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

# ================= AI-Recommended Coping Tools =================

@api_bp.route('/api/recommended_tools', methods=['GET'])
@login_required
def api_recommended_tools():
    """Get AI-recommended coping tools based on user's recent mood and needs"""
    user_id = current_user.id
    
    from app import db, client
    if not db or not client:
        # Fallback to generic tools
        return jsonify({
            "ok": False,
            "tools": [],
            "message": "Unable to generate personalized recommendations"
        })
    
    try:
        # Get recent check-ins to understand user's mood
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        recent_docs = list(checkins_ref.order_by("date", direction="DESCENDING").limit(7).stream())
        
        if not recent_docs:
            # No data, return default tools
            return jsonify({
                "ok": True,
                "tools": _get_default_tools(),
                "message": "General wellness tools"
            })
        
        # Analyze recent moods
        moods = []
        sentiments = []
        messages = []
        
        for doc in recent_docs:
            data = doc.to_dict()
            if data:
                moods.append(data.get("mood_dominant", "neutral"))
                sentiments.append(data.get("avg_sentiment", 0))
                last_text = data.get("last_text", "")
                if last_text:
                    messages.append(last_text)
        
        # Count mood frequency
        mood_counts = Counter(moods)
        dominant_mood = mood_counts.most_common(1)[0][0] if mood_counts else "neutral"
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Create AI prompt for personalized recommendations
        prompt = f"""Based on a user's recent mental health check-ins, recommend 3-4 specific coping tools.

Recent mood patterns:
- Dominant mood: {dominant_mood}
- Average sentiment: {avg_sentiment:.2f} (range: -1 to 1)
- Mood distribution: {dict(mood_counts)}

Recent messages: {" | ".join(messages[:3])}

Recommend specific coping tools from this list, and explain WHY each tool would help:
1. Breathing Exercise (4-4-4 pattern)
2. Quick Meditation (5-minute)
3. 5-4-3-2-1 Grounding
4. Gratitude Journal
5. Calming Sounds/Music
6. Positive Affirmations
7. Physical Exercise/Movement
8. Journaling/Reflection

Return ONLY valid JSON in this format (no markdown, no code blocks):
{{
  "recommended": [
    {{
      "id": "breathing",
      "title": "Breathing Exercise",
      "reason": "Specific reason why this helps based on mood pattern",
      "priority": 1
    }}
  ]
}}"""

        # Call Gemini API
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        
        # Parse response
        response_text = response.text.strip()
        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text)
        
        import json
        recommendations = json.loads(response_text)
        
        return jsonify({
            "ok": True,
            "tools": recommendations.get("recommended", []),
            "dominant_mood": dominant_mood,
            "message": f"Personalized for your recent {dominant_mood} mood"
        })
        
    except Exception as e:
        print(f"❌ Error getting recommended tools: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "ok": True,
            "tools": _get_default_tools(),
            "message": "General wellness tools"
        })

def _get_default_tools():
    """Default coping tools when AI recommendations fail"""
    return [
        {
            "id": "breathing",
            "title": "Breathing Exercise",
            "reason": "Helps calm the nervous system and reduce stress",
            "priority": 1
        },
        {
            "id": "meditation",
            "title": "Quick Meditation",
            "reason": "Promotes mindfulness and emotional balance",
            "priority": 2
        },
        {
            "id": "grounding",
            "title": "5-4-3-2-1 Grounding",
            "reason": "Anchors you in the present moment during overwhelm",
            "priority": 3
        },
        {
            "id": "affirmations",
            "title": "Positive Affirmations",
            "reason": "Builds self-confidence and positive mindset",
            "priority": 4
        }
    ]

# ================= Daily Journal & Weekly Reflection =================

def _today_key():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')

def _clean_summary_text(raw: str) -> str:
    if not raw:
        return ''
    s = str(raw)
    # Strip wrapping quotes
    s = s.strip('\n').strip('\r').strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    # Remove obvious JSON artefacts
    s = re.sub(r'"response"\s*:\s*"', '', s)
    s = s.replace('\\n', '\n')
    # Basic HTML strip
    s = re.sub(r'<[^>]+>', '', s)
    # Collapse excessive spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s

@api_bp.route('/api/daily_journal', methods=['GET'])
@login_required
def api_daily_journal():
    """Return today's daily journal summary; create a stub if missing.
    Structure of doc (users/{uid}/daily_journal/{YYYY-MM-DD}):
      raw_summary: original AI or generated summary
      clean_summary: cleaned text
      note: user private note
      created_at: timestamp
      mood: from today's checkin if available
      avg_sentiment: from today's checkin if available
    """
    from app import db
    if not db:
        return jsonify({"ok": False, "error": "Database unavailable"}), 500
    user_id = current_user.id
    day_key = _today_key()
    coll = db.collection(f"users/{user_id}/daily_journal")
    doc_ref = coll.document(day_key)
    doc = doc_ref.get()

    # Fetch today's checkin (for mood & sentiment enrichment)
    mood = None
    avg_sentiment = None
    try:
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        checkin_docs = list(checkins_ref.where('date', '==', day_key).stream())
        if checkin_docs:
            cdata = checkin_docs[0].to_dict() or {}
            mood = cdata.get('mood_dominant') or cdata.get('mood_label') or cdata.get('mood_list', ['neutral'])[0]
            avg_sentiment = cdata.get('avg_sentiment')
    except Exception as e:
        print('Checkin fetch failed for daily journal:', e)

    if not doc.exists:
        # Auto-create stub using last_text from checkin as summary if present
        raw_summary = ''
        if 'cdata' in locals():
            last_text = (cdata or {}).get('last_text') or ''
            if last_text:
                raw_summary = f"Today you felt {mood or 'neutral'}. Notable note: {last_text}"
        clean_summary = _clean_summary_text(raw_summary)
        stub = {
            'raw_summary': raw_summary,
            'clean_summary': clean_summary,
            'note': '',
            'mood': mood or 'neutral',
            'avg_sentiment': avg_sentiment if avg_sentiment is not None else None,
            'created_at': datetime.now(timezone.utc)
        }
        try:
            doc_ref.set(stub)
            return jsonify({"ok": True, "created": True, "entry": {
                'date': day_key,
                'summary': clean_summary,
                'raw_summary': raw_summary,
                'note': '',
                'mood': stub['mood'],
                'avg_sentiment': stub['avg_sentiment']
            }})
        except Exception as e:
            return jsonify({"ok": False, "error": f"Create failed: {e}"}), 500

    data = doc.to_dict() or {}
    return jsonify({"ok": True, "created": False, "entry": {
        'date': day_key,
        'summary': data.get('clean_summary', ''),
        'raw_summary': data.get('raw_summary', ''),
        'note': data.get('note', ''),
        'mood': data.get('mood', mood or 'neutral'),
        'avg_sentiment': data.get('avg_sentiment', avg_sentiment)
    }})

@api_bp.route('/api/daily_journal_list', methods=['GET'])
@login_required
def api_daily_journal_list():
    """Return list of recent daily journal entries (clean summaries)."""
    from app import db
    if not db:
        return jsonify({"ok": False, "error": "Database unavailable", "entries": []}), 500
    user_id = current_user.id
    limit = int(request.args.get('limit', '7') or 7)
    coll = db.collection(f"users/{user_id}/daily_journal")
    try:
        docs = list(coll.order_by('created_at', direction='DESCENDING').limit(limit).stream())
    except Exception:
        # fallback order by raw field 'date'
        docs = list(coll.limit(limit).stream())
    entries = []

    # For mood/sentiment enrichment create a map from checkins
    checkins_map = {}
    try:
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        chk_docs = list(checkins_ref.order_by('date', direction='DESCENDING').limit(30).stream())
        for d in chk_docs:
            cd = d.to_dict() or {}
            checkins_map[cd.get('date')] = cd
    except Exception as e:
        print('Checkins enrichment failed:', e)

    for doc in docs:
        data = doc.to_dict() or {}
        # Derive date string from doc id or created_at
        date_key = doc.id
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_key):
            # attempt from created_at
            ca = data.get('created_at')
            if isinstance(ca, datetime):
                date_key = ca.strftime('%Y-%m-%d')
        cdata = checkins_map.get(date_key, {})
        entries.append({
            'date': date_key,
            'summary': data.get('clean_summary', ''),
            'raw_summary': data.get('raw_summary', ''),
            'note': data.get('note', ''),
            'mood': data.get('mood', cdata.get('mood_dominant') or cdata.get('mood_label') or 'neutral'),
            'avg_sentiment': data.get('avg_sentiment', cdata.get('avg_sentiment'))
        })
    return jsonify({"ok": True, "entries": entries})

@api_bp.route('/api/save_journal_note', methods=['POST'])
@login_required
def api_save_journal_note():
    """Save today's private user note into daily journal doc."""
    from app import db
    if not db:
        return jsonify({"ok": False, "error": "Database unavailable"}), 500
    user_id = current_user.id
    payload = request.get_json(force=True) if request.is_json else {}
    note = (payload.get('note') or '').strip()
    day_key = _today_key()
    doc_ref = db.collection(f"users/{user_id}/daily_journal").document(day_key)
    doc = doc_ref.get()
    if not doc.exists:
        # create stub first
        doc_ref.set({
            'raw_summary': '',
            'clean_summary': '',
            'note': note,
            'mood': 'neutral',
            'avg_sentiment': None,
            'created_at': datetime.now(timezone.utc)
        })
        return jsonify({"ok": True, "created": True})
    try:
        doc_ref.update({'note': note})
        return jsonify({"ok": True, "created": False})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@api_bp.route('/api/generate_weekly_reflection', methods=['POST'])
@login_required
def api_generate_weekly_reflection():
    """Aggregate last 7 clean summaries and produce short reflection text."""
    from app import db
    if not db:
        return jsonify({"ok": False, "error": "Database unavailable"}), 500
    user_id = current_user.id
    coll = db.collection(f"users/{user_id}/daily_journal")
    try:
        docs = list(coll.order_by('created_at', direction='DESCENDING').limit(7).stream())
    except Exception:
        docs = list(coll.limit(7).stream())
    summaries = []
    moods = []
    for d in docs:
        data = d.to_dict() or {}
        cs = data.get('clean_summary') or ''
        if cs:
            summaries.append(cs)
        md = data.get('mood')
        if md:
            moods.append(md)
    if not summaries:
        return jsonify({"ok": False, "error": "No summaries to reflect on"})
    # Simple heuristic reflection
    dominant_mood = None
    if moods:
        freq = Counter(moods)
        dominant_mood = max(freq.items(), key=lambda x: x[1])[0]
    total_len = sum(len(s.split()) for s in summaries)
    avg_len = round(total_len / len(summaries), 1)
    reflection_parts = []
    if dominant_mood:
        reflection_parts.append(f"Your dominant mood this week seemed to be {dominant_mood}.")
    reflection_parts.append(f"You recorded {len(summaries)} days, with an average entry length of {avg_len} words.")
    if len(set(moods)) > 3:
        reflection_parts.append("It was an emotionally varied week—consider what influenced those shifts.")
    else:
        reflection_parts.append("Your moods were relatively consistent—notice what supported that stability.")
    reflection_parts.append("Try celebrating one small win and planning one gentle self-care action for next week.")
    reflection = ' '.join(reflection_parts)
    return jsonify({"ok": True, "reflection": reflection})
