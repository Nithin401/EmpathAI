# emotion_analysis.py
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Load models once at startup
vader_analyzer = SentimentIntensityAnalyzer()
# The BERT model for emotion classification
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
print("✅ Emotion classification model loaded successfully.")


# Advice for each emotion (These will mostly serve as fallback/guidance for Gemini now)
emotion_advice_map = {
    "joy": "😊 That’s great! Celebrate this moment or share it with someone you care about.",
    "sadness": "😢 I’m here for you. Maybe try journaling or reaching out to a friend.",
    "anger": "😠 It’s okay to feel angry. Try deep breaths or take a break.",
    "fear": "😨 That sounds scary. You’re safe here. Try writing your fears down to ease them.",
    "surprise": "😲 That’s unexpected! How do you feel about it?",
    "disgust": "🤢 Let’s talk about what bothered you. You can vent here safely.",
    "love": "❤️ That’s beautiful. Let people know how much you care.",
    "guilt": "😔 Feeling guilty is human. Think about what you need to forgive yourself.",
    "shame": "😓 Shame can be heavy. But you’re not defined by your mistakes.",
    "loneliness": "🧸 You’re not alone. I’m always here to talk to you.",
    "grief": "🖤 Grieving takes time. It’s okay to feel what you feel.",
    "hope": "🌅 That’s inspiring. Hold on to it—it helps you heal."
}

# Keywords to strongly indicate negative/sensitive mood, especially related to criticism/discrimination
SENSITIVE_NEGATIVE_KEYWORDS = [
    "criticize", "criticism", "discriminate", "discrimination", "racism", "racist",
    "unfair", "wrong", "judged", "hate", "harass", "bullying", "prejudice", "stereotyped",
    "badly treated", "abuse", "accused", "blame", "fault", "problem", "difficult"
]


def analyze_emotion(text):
    cleaned_text = text.strip().lower()

    if not cleaned_text:
        return {
            "polarity": 0.0,
            "compound": 0.0,
            "mood": "neutral",
            "intent_emotion": "neutral",
            "advice": "I'm here to support you. What's on your mind?"
        }

    try:
        # Step 1: Sentiment scores using TextBlob and VADER
        polarity = TextBlob(cleaned_text).sentiment.polarity
        compound = vader_analyzer.polarity_scores(cleaned_text)['compound']

        # Step 2: Emotion classification with BERT
        result = emotion_classifier(cleaned_text)[0]
        intent_emotion = result['label'].lower()

        # Step 3: Mood classification based on VADER compound score AND sensitive keywords
        mood = "neutral"
        if any(keyword in cleaned_text for keyword in SENSITIVE_NEGATIVE_KEYWORDS) or compound <= -0.4:
            # Force "negative" mood if sensitive keywords are present or strong negative VADER
            mood = "negative"
            intent_emotion = "sadness" if intent_emotion not in ["anger", "disgust", "fear"] else intent_emotion
        elif compound >= 0.05:
            mood = "positive"
        elif compound <= -0.05:
            mood = "negative"
        else:
            mood = "neutral"

        # Refine intent_emotion for negative cases if not strongly detected otherwise
        if mood == "negative" and intent_emotion in ["joy", "surprise", "love", "hope"]:
            intent_emotion = "sadness" # Override if mood is negative but emotion is positive (common misclassification)
        elif mood == "negative" and not any(k in intent_emotion for k in ["sadness", "anger", "fear", "disgust", "guilt", "shame", "loneliness", "grief"]):
            intent_emotion = "sadness" # General negative emotion if no specific one from BERT

        # Step 4: Get advice based on the classified emotion. This advice will serve as a strong hint for Gemini.
        advice = emotion_advice_map.get(intent_emotion, "🤖 I'm here to support you however you're feeling.")

        return {
            "polarity": polarity,
            "compound": compound,
            "mood": mood,
            "intent_emotion": intent_emotion,
            "advice": advice
        }

    except Exception as e:
        print(f"Error during emotion analysis: {e}")
        return {
            "polarity": 0.0,
            "compound": 0.0,
            "mood": "neutral",
            "intent_emotion": "unknown",
            "advice": "🤖 I’m here for you, even if I couldn’t detect your emotion accurately.",
            "error": str(e)
        }

if __name__ == '__main__':
    print("--- Test Cases ---")
    print(f"Text: 'I am so happy today!' -> {analyze_emotion('I am so happy today!')}")
    print(f"Text: 'I feel really sad and down.' -> {analyze_emotion('I feel really sad and down.')}")
    print(f"Text: 'This is the worst day ever.' -> {analyze_emotion('This is the worst day ever.')}")
    print(f"Text: 'I'm feeling okay.' -> {analyze_emotion('I\'m feeling okay.')}") # Corrected: Escaped apostrophe
    print(f"Text: 'What do you think about the weather?' -> {analyze_emotion('What do you think about the weather?')}")
    print(f"Text: 'My head hurts.' -> {analyze_emotion('My head hurts.')}")
    print(f"Text: 'I am just feeling frustrated.' -> {analyze_emotion('I am just feeling frustrated.')}")
    print(f"Text: 'people critize him because he is black is that correct accourding to you' -> {analyze_emotion('people critize him because he is black is that correct accourding to you')}")
    print(f"Text: 'Praneeth is black how can he be white could you help him?' -> {analyze_emotion('Praneeth is black how can he be white could you help him?')}")
    print(f"Text: 'I am experiencing racial discrimination at work.' -> {analyze_emotion('I am experiencing racial discrimination at work.')}")
    print(f"Text: 'Am I wrong to feel criticized?' -> {analyze_emotion('Am I wrong to feel criticized?')}")
    print(f"Text: 'They constantly mock me.' -> {analyze_emotion('They constantly mock me.')}")
    print(f"Text: 'It's unbearable.' -> {analyze_emotion('It\'s unbearable.')}")
