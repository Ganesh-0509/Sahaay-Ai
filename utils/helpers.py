from datetime import datetime, timezone, timedelta
from collections import Counter
import re

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

def calculate_streak(docs):
    """Calculate consecutive days streak from check-in documents or timestamps"""
    if not docs:
        return 0
    
    dates = set()
    for item in docs:
        # Handle if it's a timestamp object directly
        if hasattr(item, 'date') and hasattr(item, 'year'):
            dates.add(item.date())
            continue
        
        # Handle if it's a Firestore document or dict
        if hasattr(item, 'to_dict'):
            data = item.to_dict()
        elif isinstance(item, dict):
            data = item
        else:
            # Maybe it's a timestamp object that doesn't have .date() (e.g. Firestore Timestamp)
            try:
                if hasattr(item, 'timestamp'):
                    dt = datetime.fromtimestamp(item.timestamp(), tz=timezone.utc)
                    dates.add(dt.date())
                    continue
            except:
                pass
            continue
        
        # Check for 'date' string
        date_str = data.get('date', '')
        if date_str:
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                dates.add(date_obj)
            except ValueError:
                pass
        
        # Check for 'timestamp'
        ts = data.get('timestamp')
        if ts:
            if hasattr(ts, 'date'):
                dates.add(ts.date())
            elif hasattr(ts, 'timestamp'):
                dt = datetime.fromtimestamp(ts.timestamp(), tz=timezone.utc)
                dates.add(dt.date())

    if not dates:
        return 0
    
    today = datetime.now(timezone.utc).date()
    streak = 0
    current_date = today
    
    # If today not in dates, check if yesterday is (streak might still be active if they haven't checked in today yet)
    if today not in dates:
        current_date = today - timedelta(days=1)
        
    while current_date in dates:
        streak += 1
        current_date -= timedelta(days=1)
    
    return streak

def get_most_frequent_words(docs, num_words=30):
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'of', 'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'out', 'on', 'off', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'doing', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'it', 'its', 'so', 'not', 'very', 'just', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'])
    words_by_mood = {}
    for d in docs:
        if hasattr(d, 'to_dict'):
            data = d.to_dict()
        else:
            data = d
            
        mood = (data.get('mood_dominant') or data.get('mood_label') or data.get('mood', 'neutral')).lower()
        text = data.get('last_text') or data.get('text', '')
        
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

def save_checkin(db, user_id, mood, language, text, sentiment, intent="chat", helpful_tip=None):
    """
    Saves or updates a daily check-in for a user in Firestore.
    """
    if not db:
        return False

    try:
        from utils.emotion_classifier import parse_final_mood
        from google.cloud import firestore
        
        mood_list, mood_label = parse_final_mood(mood)
        dominant = mood_list[0] if mood_list else "neutral"

        today = datetime.now(timezone.utc).date()
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
                "coping_tip": helpful_tip,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP
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
                "coping_tip": helpful_tip,
                "helpful": False,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "created_at": firestore.SERVER_TIMESTAMP
            })
        return True
    except Exception as e:
        print(f"Firestore save failed for user {user_id}: {e}")
        return False
