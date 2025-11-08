def classify_emotion(message: str):
    """
    Keyword-based emotion classifier.
    Returns:
      - a single emotion string (e.g. 'joy')
      - a mixed pair as: "joy/sadness"
      - None if no keywords found
    """
    lower = message.lower()

    EMOTIONS = {
        'joy': ["happy", "glad", "excited", "grateful", "cheerful", "joy"],
        'sadness': ["sad", "down", "lonely", "depressed", "cry", "unhappy"],
        'anger': ["angry", "mad", "furious", "hate", "annoyed"],
        'fear': ["scared", "afraid", "anxious", "nervous", "fear"],
        'surprise': ["shock", "surprised", "amazed", "unexpected"],
        'disgust': ["disgust", "gross", "nasty", "yuck"],
        'hope': ["hopeful", "optimistic", "looking forward", "hope"],
        'love': ["love", "caring", "affection", "loving"]
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

    # If one emotion dominates → single emotion
    if top_pct >= 60 or second is None:
        return top_emo

    # Otherwise pick top 2 → clean mixed pair
    second_emo, _ = second
    return f"{top_emo}/{second_emo}"


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
