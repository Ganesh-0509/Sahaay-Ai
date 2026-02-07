def classify_emotion_llm(message: str, client=None):
    """
    LLM-based emotion classifier for nuanced understanding.
    Returns a single emotion or mixed pair (e.g., "relief/anxiety").
    Falls back to keyword-based if LLM fails.
    """
    if not client:
        return classify_emotion_keywords(message)
    
    try:
        prompt = f"""Analyze the emotional content of this message and identify the primary emotion(s).

Choose from: joy, sadness, anger, fear, surprise, disgust, hope, love, relief, anxiety, neutral

Rules:
- If ONE emotion clearly dominates, return just that emotion
- If TWO emotions are equally present, return them as "emotion1/emotion2"
- Never return more than 2 emotions
- Be precise and context-aware

Message: "{message}"

Response (single emotion or emotion1/emotion2):"""

        models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-flash-lite"]
        
        for model in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                result = response.text.strip().lower()
                
                # Clean the response
                result = result.replace("emotion:", "").replace("emotions:", "").strip()
                
                # Validate format
                if "/" in result:
                    parts = result.split("/")
                    if len(parts) == 2:
                        return f"{parts[0].strip()}/{parts[1].strip()}"
                elif result and len(result.split()) <= 2:
                    return result.split()[0]
                    
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    continue
                break
        
        # Fallback to keywords
        return classify_emotion_keywords(message)
        
    except Exception as e:
        print(f"LLM emotion classification error: {e}")
        return classify_emotion_keywords(message)


def classify_emotion_keywords(message: str):
    """
    Keyword-based emotion classifier (fallback).
    Returns:
      - a single emotion string (e.g. 'joy')
      - a mixed pair as: "joy/sadness"
      - None if no keywords found
    """
    lower = message.lower()

    EMOTIONS = {
        'joy': ["happy", "glad", "excited", "grateful", "cheerful", "joy", "celebrate", "amazing"],
        'sadness': ["sad", "down", "lonely", "depressed", "cry", "unhappy", "hurt", "miss"],
        'anger': ["angry", "mad", "furious", "hate", "annoyed", "frustrated", "irritated"],
        'fear': ["scared", "afraid", "anxious", "nervous", "fear", "worried", "panic"],
        'surprise': ["shock", "surprised", "amazed", "unexpected", "wow"],
        'disgust': ["disgust", "gross", "nasty", "yuck", "revolting"],
        'hope': ["hopeful", "optimistic", "looking forward", "hope", "confident"],
        'love': ["love", "caring", "affection", "loving", "adore"],
        'relief': ["relief", "relieved", "calm", "relaxed", "better"],
        'anxiety': ["anxious", "stress", "tense", "uneasy", "restless"]
    }

    counts = {emo: 0 for emo in EMOTIONS}

    for emo, keys in EMOTIONS.items():
        for kw in keys:
            if kw in lower:
                counts[emo] += lower.count(kw)

    total = sum(counts.values())
    if total == 0:
        return None

    # Calculate percentages
    percents = [(emo, (cnt / total) * 100) for emo, cnt in counts.items() if cnt > 0]
    percents.sort(key=lambda x: x[1], reverse=True)

    # Top emotions
    top_emo, top_pct = percents[0]
    second = percents[1] if len(percents) > 1 else None

    # Lowered threshold from 60% to 70% for clearer single emotions
    if top_pct >= 70 or second is None:
        return top_emo

    # Otherwise pick top 2 → clean mixed pair
    second_emo, second_pct = second
    
    # Only mix if second emotion is significant (>25%)
    if second_pct >= 25:
        return f"{top_emo}/{second_emo}"
    
    return top_emo


def classify_emotion(message: str, client=None):
    """
    Main emotion classification function.
    Uses LLM if available, falls back to keywords.
    """
    if client:
        return classify_emotion_llm(message, client)
    return classify_emotion_keywords(message)


def parse_final_mood(raw_mood: str):
    """
    Convert classifier output into clean mood labels for frontend + DB.

    Input:
        "joy" → ["joy"]
        "joy/sadness" → ["joy", "sadness"]
        None → []

    Output:
        list of emotions (for analytics)
        AND display label "Joy / Sadness"
    """
    if raw_mood is None:
        return [], ""

    raw = raw_mood.strip()

    if "/" in raw:
        parts = [p.strip() for p in raw.split("/") if p.strip()]
        clean_display = " / ".join(p.capitalize() for p in parts)
        return parts, clean_display

    # single emotion
    return [raw], raw.capitalize()
