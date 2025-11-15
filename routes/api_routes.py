from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from utils.helpers import format_timestamp, calculate_streak, get_most_frequent_words
from datetime import datetime, timezone, timedelta

api_bp = Blueprint('api', __name__)

@api_bp.route('/api/home_data')
@login_required
def home_data():
    """Fetch home dashboard data for the current user"""
    user_id = current_user.id
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
    
    try:
        checkins_ref = db.collection(f"users/{user_id}/checkins")
        
        # Determine query based on period
        if period == 'last10':
            docs = list(checkins_ref.order_by("date", direction="DESCENDING").limit(10).stream())
        elif period == 'last7days':
            seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
            docs = list(checkins_ref.where("date", ">=", seven_days_ago).order_by("date", direction="DESCENDING").stream())
        else:  # all
            docs = list(checkins_ref.order_by("date", direction="DESCENDING").limit(100).stream())
        
        # Process data
        recent_checkins = []
        helpful_tips = []
        mood_counts = {}
        
        for doc in docs:
            data = doc.to_dict()
            if data:
                date = data.get("date", "")
                mood = data.get("mood_dominant", data.get("mood_label", "neutral"))
                coping_tip = data.get("coping_tip", "")
                is_helpful = data.get("helpful", False)
                
                # Count moods
                mood_counts[mood] = mood_counts.get(mood, 0) + 1
                
                # Add to recent checkins
                recent_checkins.append({
                    "date": date,
                    "mood": mood
                })
                
                # Add to helpful tips if marked as helpful
                if is_helpful and coping_tip:
                    helpful_tips.append({
                        "date": date,
                        "mood": mood,
                        "tip": coping_tip
                    })
        
        # Calculate most common mood
        most_common_mood = max(mood_counts.items(), key=lambda x: x[1])[0] if mood_counts else "No data yet"
        
        # Calculate streak (consecutive days with check-ins)
        streak = calculate_streak(docs)
        
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
                    "helpful": data.get("helpful", False)
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
    # ...implement logic...
    return jsonify({"ok": True})

# Add other API routes as needed
