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
        return None
    dominant = max(scores.items(), key=lambda item: item[1])
    if dominant[1] <= 0.0:
        return None
    return dominant[0]


def emotion_detector(text_to_analyze: str):
    # Handle blank input
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(EMOTION_ENDPOINT, json=payload, headers=HEADERS)
        # Handle invalid text / 400 response
        if response.status_code == 400:
            return {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None
            }
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Emotion detection request failed: {e}") from e

    data = response.json()
    scores = _extract_scores(data)
    dom = _dominant_emotion(scores)

    return {
        'anger': scores.get('anger'),
        'disgust': scores.get('disgust'),
        'fear': scores.get('fear'),
        'joy': scores.get('joy'),
        'sadness': scores.get('sadness'),
        'dominant_emotion': dom
    }
