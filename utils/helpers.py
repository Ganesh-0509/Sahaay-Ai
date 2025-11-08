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

def calculate_streak(timestamps):
    dates = {firestore_to_datetime(ts).date() for ts in timestamps if firestore_to_datetime(ts)}
    if not dates: return 0
    today = datetime.utcnow().date()
    streak = 0
    while today in dates:
        streak += 1
        today -= timedelta(days=1)
    return streak

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
