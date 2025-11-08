from textblob import TextBlob

def get_sentiment_score(message):
    analysis = TextBlob(message)
    return analysis.sentiment.polarity

def get_coarse_mood(sentiment_score):
    if sentiment_score > 0.2:
        return "positive"
    elif sentiment_score < -0.2:
        return "negative"
    else:
        return "mixed"
