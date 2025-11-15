import os
import json
from flask import session

TRANSLATION_DIR = os.path.join(os.path.dirname(__file__))
LANGUAGES = ['en', 'ta', 'hi', 'te']

# Preload all translations at import for direct access
TRANSLATIONS = {}
for lang in LANGUAGES:
    path = os.path.join(TRANSLATION_DIR, f'{lang}.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            TRANSLATIONS[lang] = json.load(f)
    else:
        # fallback to English if missing
        with open(os.path.join(TRANSLATION_DIR, 'en.json'), encoding='utf-8') as f:
            TRANSLATIONS[lang] = json.load(f)

# Loads translation JSON for a given language code
def load_translation(lang_code):
    path = os.path.join(TRANSLATION_DIR, f'{lang_code}.json')
    if not os.path.exists(path):
        path = os.path.join(TRANSLATION_DIR, 'en.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)

# Gets current user's translation dictionary
def get_user_translations():
    user_language = session.get('language', 'en')
    return load_translation(user_language)

# Translates a mood label to the user's language
def translate_mood(mood_key, lang_code=None):
    if not lang_code:
        lang_code = session.get('language', 'en')
    
    # Load translation for the language
    translations = TRANSLATIONS.get(lang_code, TRANSLATIONS['en'])
    
    # Get mood translations
    moods = translations.get('moods', {})
    
    # Return translated mood or original if not found
    mood_lower = mood_key.lower() if mood_key else "neutral"
    return moods.get(mood_lower, mood_key)
