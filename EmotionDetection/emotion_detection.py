import requests

EMOTION_ENDPOINT = (
    'https://sn-watson-emotion.labs.skills.network/'
    'v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}

def _extract_scores(data):
    scores = {"anger": 0.0, "disgust": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.0}

    # recursively search nested dicts and lists
    if isinstance(data, dict):
        for key, value in data.items():
            if key in scores and isinstance(value, (int, float)):
                scores[key] = float(value)
            elif isinstance(value, dict):
                inner_scores = _extract_scores(value)
                for k in scores:
                    if inner_scores[k] > 0:
                        scores[k] = inner_scores[k]
            elif isinstance(value, list):
                for item in value:
                    inner_scores = _extract_scores(item)
                    for k in scores:
                        if inner_scores[k] > 0:
                            scores[k] = inner_scores[k]
    return scores


def _dominant_emotion(scores):
    if not isinstance(scores, dict):
        return "unknown"
    dominant = max(scores.items(), key=lambda item: item[1])
    if dominant[1] <= 0.0:
        return "unknown"
    return dominant[0]

def emotion_detector(text_to_analyze: str):
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        raise ValueError("text_to_analyze must be a non-empty string.")

    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    try:
        response = requests.post(EMOTION_ENDPOINT, json=payload, headers=HEADERS)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Emotion detection request failed: {e}") from e

    data = response.json()

    scores = _extract_scores(data)
    dom = _dominant_emotion(scores)

    return {
        'anger': scores.get('anger', 0.0),
        'disgust': scores.get('disgust', 0.0),
        'fear': scores.get('fear', 0.0),
        'joy': scores.get('joy', 0.0),
        'sadness': scores.get('sadness', 0.0),
        'dominant_emotion': dom
    }