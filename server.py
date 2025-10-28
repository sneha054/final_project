
"""
Flask web server for the Emotion Detection API.
This server exposes an endpoint `/emotionDetector` that analyzes
the emotion of a given text and returns a formatted response.
"""

from flask import Flask, request, jsonify
from emotion_detection import emotion_detector

app = Flask(__name__)

@app.route("/emotionDetector", methods=["GET"])
def emotion_detection_endpoint():
    """
    Endpoint to analyze the emotion of the provided text.
    Returns formatted emotion scores and the dominant emotion.
    Handles blank inputs gracefully.
    """
    text_to_analyze = request.args.get("text", "")
    result = emotion_detector(text_to_analyze)

    if result["dominant_emotion"] is None:
        return jsonify({"error": "Invalid text! Please try again!"}), 400

    response_message = (
        f"For the given statement, the system response is 'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, 'fear': {result['fear']}, "
        f"'joy': {result['joy']} and 'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )
    return response_message


if __name__ == "__main__":
    app.run(host="localhost", port=5000)
